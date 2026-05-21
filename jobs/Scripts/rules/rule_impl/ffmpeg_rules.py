import json
import os
import re
import subprocess

from rules.rule import Rule
import logging

logger = logging.getLogger(__name__)


def _extract_stderr_hint(stderr):
    """Return first 400 chars after the ffmpeg progress marker, or from start if absent."""
    if not stderr:
        return ""
    marker = "Press [q] to stop, [?] for help"
    idx = stderr.find(marker)
    text = stderr[idx + len(marker):].lstrip("\n") if idx != -1 else stderr
    return text[:400].strip()


class ConversionSuccessRule(Rule):
    """
    Checks that the FFMPEG conversion process completed without error.
    Always applied to every test case.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="FFMPEG conversion process returned a non-zero exit code",
            description="Check that ffmpeg exited successfully (return code 0)"
        )

    def should_be_executed(self):
        return True

    def apply(self, context):
        returncode = context.get("returncode")
        if returncode is None or returncode != 0:
            self.add_error(
                f"FFMPEG conversion failed with exit code: {returncode}"
            )
        else:
            logger.info("ConversionSuccessRule: ffmpeg exited successfully")


class MetadataRule(Rule):
    """
    Runs ffprobe on the output video, records metadata in the report,
    then checks that each field in expected_metadata matches (always v:0).
    If "stream" and "expected_metadata_for_stream" are present in the case,
    also runs ffprobe on that stream selector (e.g. "a:0") and checks those fields.
    Reports each mismatched field individually.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Converted video metadata does not match expected values",
            description="Validate ffprobe metadata of output video against test case expected_metadata"
        )
        self.expected        = self.case.get("expected_metadata", {})
        self.stream          = self.case.get("stream")
        self.expected_stream = self.case.get("expected_metadata_for_stream", {})

    def should_be_executed(self):
        return "expected_metadata" in self.case and bool(self.case["expected_metadata"])

    def _get_metadata(self, ffprobe_exe, video_path, stream_selector, fields_dict):
        fields = ",".join(fields_dict.keys())
        cmd = (f'"{ffprobe_exe}" -v error -select_streams {stream_selector}'
               f' -show_entries stream={fields}'
               f' -print_format json "{video_path}"')
        logger.info(f"Running ffprobe metadata ({stream_selector}): {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=60, shell=True)
            if result.stderr:
                logger.warning(f"ffprobe stderr: {result.stderr.strip()}")
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if not streams:
                logger.error(f"No streams found for selector '{stream_selector}' in {video_path}")
                return {}
            stream = streams[0]
            logger.info(f"Metadata received for {stream_selector}")
            return {k: v for k, v in stream.items() if k in fields_dict}
        except Exception as e:
            logger.error(f"ffprobe metadata error: {e}")
            return {}

    def _check_fields(self, actual, expected, label):
        all_match = True
        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if actual_value is None:
                self.add_error(
                    f"Metadata field '{field}' not found in {label}. Expected: {expected_value}"
                )
                all_match = False
            elif str(actual_value) != str(expected_value):
                self.add_error(
                    f"Metadata mismatch for '{field}' in {label}: expected={expected_value}, actual={actual_value}"
                )
                all_match = False
        return all_match

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("MetadataRule: skipped - output video not produced")
            return

        # Always: video stream (v:0) vs expected_metadata
        actual = self._get_metadata(context["ffprobe_exe"], context["output_video"],
                                    "v:0", self.expected)
        self.json_content["metadata"] = actual

        if actual:
            meta_parts = [f"{k}: {v}" for k, v in actual.items()]
            self.json_content["message"].append({
                "issue":       "Video stream metadata: " + ", ".join(meta_parts),
                "description": "ffprobe output (v:0)"
            })

        if self._check_fields(actual, self.expected, "video stream (v:0)"):
            logger.info("MetadataRule: all video stream fields match")

        # Optional: additional stream (e.g. a:0) vs expected_metadata_for_stream
        if self.stream and self.expected_stream:
            actual_stream = self._get_metadata(context["ffprobe_exe"], context["output_video"],
                                               self.stream, self.expected_stream)
            self.json_content["metadata_for_stream"] = actual_stream

            if actual_stream:
                stream_parts = [f"{k}: {v}" for k, v in actual_stream.items()]
                self.json_content["message"].append({
                    "issue":       f"Stream metadata ({self.stream}): " + ", ".join(stream_parts),
                    "description": f"ffprobe output ({self.stream})"
                })

            if self._check_fields(actual_stream, self.expected_stream, f"stream ({self.stream})"):
                logger.info(f"MetadataRule: all {self.stream} stream fields match")


class PSNRRule(Rule):
    """
    Measures PSNR between input and output video using ffmpeg psnr filter.
    Writes per-frame stats to psnr_log (used later by run_tests for worst-frame extraction).
    Skipped when has_reference is False (e.g. lavfi source) or conversion failed.
    Default threshold: 30.0 dB (overridable via case "psnr_threshold" field).
    """

    DEFAULT_THRESHOLD      = 30.0
    PSNR_COMPARE_TOLERANCE = 3.0   # dB: AMF may be this much lower than non-AMF ref before error

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="PSNR value is below acceptable threshold",
            description="Compare PSNR between input and output video using FFMPEG psnr filter"
        )

    def should_be_executed(self):
        return True

    def _measure_psnr(self, ffmpeg_exe, input_video, output_video, log_path):
        log_dir  = os.path.dirname(log_path)
        log_name = os.path.basename(log_path)
        os.makedirs(log_dir, exist_ok=True)
        cmd = (f'"{ffmpeg_exe}" -i "{input_video}" -i "{output_video}"'
               f' -lavfi "psnr=stats_file={log_name}" -f null -')
        logger.info("Measuring PSNR (ffmpeg filter)")
        try:
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True,
                                    timeout=300, shell=True, cwd=log_dir)
            match = re.search(r"average:(\S+)", result.stderr)
            if match:
                val  = match.group(1)
                psnr = float("inf") if val == "inf" else float(val)
                logger.info(f"PSNR average: {psnr}")
                return psnr, ""
            logger.error("Could not parse PSNR")
            return None, _extract_stderr_hint(result.stderr)
        except Exception as e:
            logger.error(f"PSNR error: {e}")
            return None, ""

    def apply(self, context):
        if not context.get("has_reference"):
            logger.info("PSNRRule: skipped - no reference input video (lavfi source)")
            return

        if not context.get("output_exists"):
            logger.info("PSNRRule: skipped - output video not produced")
            return

        if context.get("returncode") != 0:
            logger.info("PSNRRule: skipped - conversion failed, PSNR on corrupt output is not meaningful")
            return

        psnr_log = context["psnr_log"]
        psnr, stderr_hint = self._measure_psnr(
            context["ffmpeg_exe"], context["input_video"], context["output_video"], psnr_log
        )
        self.json_content["psnr"] = psnr

        if os.path.exists(psnr_log):
            results_dir = context.get("results_dir", "")
            self.json_content["psnr_log"] = (
                os.path.relpath(psnr_log, results_dir).replace("\\", "/")
            )

        threshold = self.case.get("psnr_threshold", self.DEFAULT_THRESHOLD)

        if psnr is None:
            msg = "PSNR measurement failed - no value returned by ffmpeg"
            if stderr_hint:
                msg += "\n" + stderr_hint
            self.add_error(msg)
            return

        if psnr == float("inf"):
            logger.info(f"PSNRRule: PSNR is infinite (lossless or identical), threshold={threshold}")
        elif psnr < threshold:
            self.add_error(
                f"Unacceptable PSNR: {psnr:.2f} dB (threshold: {threshold} dB)"
            )
        else:
            logger.info(f"PSNRRule: PSNR={psnr:.2f} dB >= threshold={threshold} dB - OK")

        self.json_content["message"].append({
            "issue":       f"PSNR: {psnr:.2f} dB",
            "description": "Quality metrics"
        })

        if context.get("reference_video"):
            ref_baseline = context.get("reference_input_video") or context["input_video"]
            ref_psnr = self._measure_psnr(
                context["ffmpeg_exe"], ref_baseline,
                context["reference_video"], context["reference_psnr_log"]
            )[0]
            self.json_content["psnr_reference"] = ref_psnr
            if os.path.exists(context["reference_psnr_log"]):
                self.json_content["psnr_reference_log"] = (
                    os.path.relpath(context["reference_psnr_log"],
                                    context.get("results_dir", "")).replace("\\", "/")
                )
            if ref_psnr is None:
                logger.warning("PSNRRule: reference PSNR measurement failed")
            else:
                self.json_content["message"].append({
                    "issue":       f"PSNR reference (non-AMF): {ref_psnr:.2f} dB",
                    "description": "Reference quality metrics"
                })
                tolerance = self.case.get("psnr_reference_tolerance", self.PSNR_COMPARE_TOLERANCE)
                if psnr < ref_psnr - tolerance:
                    self.add_error(
                        f"AMF PSNR ({psnr:.2f} dB) is significantly lower than "
                        f"non-AMF reference ({ref_psnr:.2f} dB), tolerance={tolerance} dB"
                    )
                else:
                    logger.info(
                        f"PSNRRule: AMF PSNR ({psnr:.2f} dB) within tolerance of "
                        f"reference ({ref_psnr:.2f} dB) - OK"
                    )


class SSIMRule(Rule):
    """
    Measures SSIM between input and output video using ffmpeg ssim filter.
    Skipped when has_reference is False (e.g. lavfi source) or conversion failed.
    Range: 0.0 to 1.0. Default threshold: 0.9 (overridable via case "ssim_threshold" field).
    """

    DEFAULT_THRESHOLD      = 0.9
    SSIM_COMPARE_TOLERANCE = 0.01  # AMF may be this much lower than non-AMF ref before error

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="SSIM value is below acceptable threshold",
            description="Compare SSIM between input and output video using FFMPEG ssim filter"
        )

    def should_be_executed(self):
        return True

    def _measure_ssim(self, ffmpeg_exe, input_video, output_video, log_path):
        cmd = (f'"{ffmpeg_exe}" -i "{input_video}" -i "{output_video}"'
               f' -lavfi ssim -f null -')
        logger.info("Measuring SSIM (ffmpeg filter)")
        try:
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True,
                                    timeout=300, shell=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result.stderr)
            match = re.search(r"All:(\S+)", result.stderr)
            if match:
                ssim = float(match.group(1))
                logger.info(f"SSIM All: {ssim}")
                return ssim, ""
            logger.error("Could not parse SSIM")
            return None, _extract_stderr_hint(result.stderr)
        except Exception as e:
            logger.error(f"SSIM error: {e}")
            return None, ""

    def apply(self, context):
        if not context.get("has_reference"):
            logger.info("SSIMRule: skipped - no reference input video (lavfi source)")
            return

        if not context.get("output_exists"):
            logger.info("SSIMRule: skipped - output video not produced")
            return

        if context.get("returncode") != 0:
            logger.info("SSIMRule: skipped - conversion failed, SSIM on corrupt output is not meaningful")
            return

        ssim_log = context["ssim_log"]
        ssim, stderr_hint = self._measure_ssim(
            context["ffmpeg_exe"], context["input_video"], context["output_video"], ssim_log
        )
        self.json_content["ssim"] = ssim

        if os.path.exists(ssim_log):
            results_dir = context.get("results_dir", "")
            self.json_content["ssim_log"] = (
                os.path.relpath(ssim_log, results_dir).replace("\\", "/")
            )

        threshold = self.case.get("ssim_threshold", self.DEFAULT_THRESHOLD)

        if ssim is None:
            msg = "SSIM measurement failed - no value returned by ffmpeg"
            if stderr_hint:
                msg += "\n" + stderr_hint
            self.add_error(msg)
            return

        if ssim < threshold:
            self.add_error(
                f"Unacceptable SSIM: {ssim:.4f} (threshold: {threshold})"
            )
        else:
            logger.info(f"SSIMRule: SSIM={ssim:.4f} >= threshold={threshold} - OK")

        self.json_content["message"].append({
            "issue":       f"SSIM: {ssim:.4f}",
            "description": "Quality metrics"
        })

        if context.get("reference_video"):
            ref_baseline = context.get("reference_input_video") or context["input_video"]
            ref_ssim = self._measure_ssim(
                context["ffmpeg_exe"], ref_baseline,
                context["reference_video"], context["reference_ssim_log"]
            )[0]
            self.json_content["ssim_reference"] = ref_ssim
            if os.path.exists(context["reference_ssim_log"]):
                self.json_content["ssim_reference_log"] = (
                    os.path.relpath(context["reference_ssim_log"],
                                    context.get("results_dir", "")).replace("\\", "/")
                )
            if ref_ssim is None:
                logger.warning("SSIMRule: reference SSIM measurement failed")
            else:
                self.json_content["message"].append({
                    "issue":       f"SSIM reference (non-AMF): {ref_ssim:.4f}",
                    "description": "Reference quality metrics"
                })
                tolerance = self.case.get("ssim_reference_tolerance", self.SSIM_COMPARE_TOLERANCE)
                if ssim < ref_ssim - tolerance:
                    self.add_error(
                        f"AMF SSIM ({ssim:.4f}) is significantly lower than "
                        f"non-AMF reference ({ref_ssim:.4f}), tolerance={tolerance}"
                    )
                else:
                    logger.info(
                        f"SSIMRule: AMF SSIM ({ssim:.4f}) within tolerance of "
                        f"reference ({ref_ssim:.4f}) - OK"
                    )


class DecodeRule(Rule):
    """
    Decodes the output video with ffmpeg -v error and checks that stderr is empty.
    Any output indicates decode errors: corrupted bitstream, broken frames, etc.
    Applied only when listed in case "rules".
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Decode errors detected in output video",
            description="Decode output video with ffmpeg -v error and verify empty stderr"
        )

    def should_be_executed(self):
        return True

    def _decode_check(self, ffmpeg_exe, output_video):
        cmd = (f'"{ffmpeg_exe}" -hide_banner -v error'
               f' -i "{output_video}" -map 0:v:0 -f null -')
        logger.info(f"Running decode check: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, shell=True)
            return result.stderr.strip()
        except Exception as e:
            logger.error(f"Decode check error: {e}")
            return str(e)

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("DecodeRule: skipped - output video not produced")
            return

        decode_errors = self._decode_check(context["ffmpeg_exe"], context["output_video"])
        if decode_errors:
            message = f"Decode errors detected: {decode_errors}"
            self.add_error(message if len(message) <= 1000 else message[:400] + "........." + message[-500:])
        else:
            logger.info("DecodeRule: no decode errors")
            self.json_content["message"].append({
                "issue":       "No decode errors",
                "description": "Decode check passed"
            })


class DecodeAudioRule(Rule):
    """
    Decodes the output audio with ffmpeg -v error and checks that stderr is empty.
    Applied only when listed in case "rules".
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Decode audio errors detected in output video",
            description="Decode output audio with ffmpeg -v error and verify empty stderr"
        )

    def should_be_executed(self):
        return True

    def _decode_check(self, ffmpeg_exe, output_video):
        cmd = (f'"{ffmpeg_exe}" -hide_banner -v error'
               f' -i "{output_video}" -map 0:a:0 -f null -')
        logger.info(f"Running decode audio check: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, shell=True)
            return result.stderr.strip()
        except Exception as e:
            logger.error(f"Decode audio check error: {e}")
            return str(e)

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("DecodeAudioRule: skipped - output video not produced")
            return

        decode_errors = self._decode_check(context["ffmpeg_exe"], context["output_video"])
        if decode_errors:
            message = f"Decode audio errors detected: {decode_errors}"
            self.add_error(message if len(message) <= 1000 else message[:400] + "........." + message[-500:])
        else:
            logger.info("DecodeAudioRule: no audio decode errors")
            self.json_content["message"].append({
                "issue":       "No audio decode errors",
                "description": "Decode audio check passed"
            })


# Maps output_format extension to the exact ffprobe format_name string.
_FORMAT_NAME_MAP = {
    "mp4":  "mov,mp4,m4a,3gp,3g2,mj2",
    "mkv":  "matroska,webm",
    "ts":   "mpegts",
    "webm": "matroska,webm",
}


class FormatRule(Rule):
    """
    Verifies the output container format, duration, and bitrate via ffprobe -show_entries format.
    Format:   maps output_format extension → ffprobe format_name string via _FORMAT_NAME_MAP
              (e.g. "mkv" → "matroska" checked against "matroska,webm").
    Duration: if -frames:v N and rate=R are parseable from keys,
              checks actual duration is within 1 frame (1/R s) of N/R.
    Bitrate:  if -b:v is present, checks actual ± 10% of target; if -maxrate is present,
              checks actual does not exceed it. Both skipped if neither flag is in keys.
    Skipped when output video was not produced.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Output container format or duration does not match expected",
            description="Verify container format_name and duration via ffprobe -show_entries format"
        )

    def should_be_executed(self):
        return True

    def _get_format_info(self, ffprobe_exe, output_video):
        cmd = (f'"{ffprobe_exe}" -v error -show_entries format=format_name,duration,size,bit_rate'
               f' -of default=noprint_wrappers=1 "{output_video}"')
        logger.info(f"Running format info: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=60, shell=True)
            if result.stderr:
                logger.warning(f"ffprobe stderr: {result.stderr.strip()}")
            info = {}
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    info[k.strip()] = v.strip()
            logger.info(f"Format info: {info}")
            return info
        except Exception as e:
            logger.error(f"Format info error: {e}")
            return {}

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("FormatRule: skipped - output video not produced")
            return

        info = self._get_format_info(context["ffprobe_exe"], context["output_video"])
        self.json_content["format_info"] = info

        # Format check
        case_output_format = self.case.get("output_format", "mp4")
        expected_output_format = _FORMAT_NAME_MAP.get(case_output_format, case_output_format)
        actual_output_format = info.get("format_name", "")

        if actual_output_format != expected_output_format:
            self.add_error(f"Format mismatch: expected '{expected_output_format}', got '{actual_output_format}'")
        else:
            logger.info(f"FormatRule: format OK - {actual_output_format}")
            self.json_content["message"].append({
                "issue":       f"Container format: {actual_output_format}",
                "description": "Format check passed"
            })

        # Duration and bitrate check
        keys         = self.case.get("keys", "")
        frames_match = re.search(r"-frames:v\s+(\d+)", keys)
        rate_match   = re.search(r"rate=(\d+(?:\.\d+)?)", keys)
        bitrate_in_keys = (re.search(r"-b:v\s+(\d+)([kKmM]?)", keys))
        mbitrate_in_keys = (re.search(r"-maxrate\s+(\d+)([kKmM]?)", keys))
        bitrate_tolerance = 0
        if bitrate_in_keys:
            value = int(bitrate_in_keys.group(1))
            unit = bitrate_in_keys.group(2).lower()
            multiplier = {
                "": 1,
                "k": 1000,
                "m": 1000000
                }[unit]
            expected_bitrate = value * multiplier
            case_bitrate_tolerance = self.case.get("bitrate_tolerance", 0.5)
            bitrate_tolerance = case_bitrate_tolerance*expected_bitrate
        else:
            expected_bitrate = None
        if mbitrate_in_keys:
            value = int(mbitrate_in_keys.group(1))
            unit = mbitrate_in_keys.group(2).lower()
            multiplier = {
                "": 1,
                "k": 1000,
                "m": 1000000
                }[unit]
            max_bitrate = value * multiplier
        else:
            max_bitrate = None

        try:
            actual_duration = float(info.get("duration", -1))
        except (ValueError, TypeError):
            actual_duration = -1

        try:
            actual_bitrate = float(info.get("bit_rate", -1))
        except (ValueError, TypeError):
            actual_bitrate = -1

        if not (frames_match and rate_match):
            logger.info("FormatRule: duration check skipped - -frames:v or rate not found in keys")
            self.json_content["message"].append({
                "issue":       f"Duration: {actual_duration:.3f}s",
                "description": "Duration check skipped — no -frames:v or rate in keys"
            })
        
        if expected_bitrate is None and max_bitrate is None:
            logger.info("FormatRule: bitrate check skipped - -b:v and -maxrate not found in keys")
            self.json_content["message"].append({
                "issue":       f"Bitrate: {actual_bitrate}",
                "description": "Bitrate check skipped — no -b:v and -maxrate in keys"
            })

        if not (frames_match and rate_match) and expected_bitrate is None and max_bitrate is None:
            return

        if (frames_match and rate_match):
            n    = int(frames_match.group(1))
            rate = float(rate_match.group(1))
            expected_duration = n / rate
            duration_tolerance = 1.0 / rate

            if actual_duration < 0:
                self.add_error("Could not retrieve duration from ffprobe")
            elif abs(actual_duration - expected_duration) > duration_tolerance:
                self.add_error(
                    f"Duration mismatch: expected {expected_duration:.3f}s, "
                    f"actual={actual_duration:.3f}s (tolerance={duration_tolerance:.3f}s)"
                )
            else:
                logger.info(f"FormatRule: duration OK - {actual_duration:.3f}s - {expected_duration:.3f}s")
                self.json_content["message"].append({
                    "issue":       f"Duration: {actual_duration:.3f}s (expected: {expected_duration:.3f}s)",
                    "description": "Duration check passed"
                })

        if (expected_bitrate or max_bitrate):
            if actual_bitrate < 0:
                self.add_error("Could not retrieve bitrate from ffprobe")
            elif expected_bitrate is not None and abs(actual_bitrate - expected_bitrate) > bitrate_tolerance:
                self.add_error(
                    f"Bitrate mismatch: expected bitrate {expected_bitrate}, "
                    f"actual={actual_bitrate} (tolerance={bitrate_tolerance})"
                )
            elif max_bitrate is not None and actual_bitrate > max_bitrate:
                self.add_error(
                    f"Bitrate mismatch: expected max bitrate {max_bitrate}, "
                    f"actual={actual_bitrate}"
                )
            else:
                expected_bitrate_values = []
                if expected_bitrate is not None:
                    expected_bitrate_values.append(f"Bitrate - {expected_bitrate}")
                if max_bitrate is not None:
                    expected_bitrate_values.append(f"Max bitrate - {max_bitrate}")
                logger.info(f"FormatRule: bitrate OK - {actual_bitrate} - {max_bitrate}")
                self.json_content["message"].append({
                    "issue":       f"Bitrate: {actual_bitrate} (expected:{', '.join(expected_bitrate_values)}, tolerance={bitrate_tolerance})",
                    "description": "Bitrate check passed"
                })


class KeyframeRule(Rule):
    """
    Verifies that keyframes appear at the expected GOP interval.
    GOP size is parsed from -g N in case keys — rule is skipped if -g is absent.
    Checks that every inter-keyframe gap is within [gop_size - TOLERANCE, gop_size + TOLERANCE].
    The tail interval (last keyframe → end of video) is not checked.
    Tolerance is overridable per case via "keyframe_tolerance" field.
    """

    TOLERANCE = 3  # frames: allowed deviation from expected GOP interval

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Keyframe interval does not match expected GOP size",
            description="Verify keyframe positions via ffprobe match -g N from keys"
        )
        self._gop_size = None
        match = re.search(r"-g\s+(\d+)", self.case.get("keys", ""))
        if match:
            self._gop_size = int(match.group(1))

    def should_be_executed(self):
        return self._gop_size is not None

    def _get_keyframe_indices(self, ffprobe_exe, output_video):
        cmd = (f'"{ffprobe_exe}" -v error -select_streams v:0'
               f' -show_entries frame=key_frame'
               f' -of csv "{output_video}"')
        logger.info(f"Running keyframe check: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, shell=True)
            indices = []
            for frame_idx, line in enumerate(result.stdout.splitlines()):
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[1].strip() == "1":
                    indices.append(frame_idx)
            return indices
        except Exception as e:
            logger.error(f"Keyframe check error: {e}")
            return None

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("KeyframeRule: skipped - output video not produced")
            return

        gop_size  = self._gop_size
        tolerance = self.case.get("keyframe_tolerance", self.TOLERANCE)

        keyframe_indices = self._get_keyframe_indices(
            context["ffprobe_exe"], context["output_video"]
        )

        if keyframe_indices is None:
            self.add_error("Keyframe check failed - ffprobe error")
            return
        if not keyframe_indices:
            self.add_error("No keyframes found in output video")
            return

        # First frame must always be a keyframe
        if keyframe_indices[0] != 0:
            self.add_error(
                f"First frame is not a keyframe (first keyframe at frame {keyframe_indices[0]})"
            )

        # Check all inter-keyframe intervals (tail not checked)
        violations = []
        for i in range(1, len(keyframe_indices)):
            interval = keyframe_indices[i] - keyframe_indices[i - 1]
            if not (gop_size - tolerance <= interval <= gop_size + tolerance):
                violations.append(
                    f"frame {keyframe_indices[i]}: interval={interval} "
                    f"(expected {gop_size}±{tolerance})"
                )

        if violations:
            summary = f"GOP interval violations ({len(violations)}): " + "; ".join(violations[:5])
            if len(violations) > 5:
                summary += f" ... and {len(violations) - 5} more"
            self.add_error(summary)
        else:
            logger.info(
                f"KeyframeRule: all {len(keyframe_indices)} keyframes within "
                f"GOP={gop_size}±{tolerance} - OK"
            )
            self.json_content["message"].append({
                "issue":       f"Keyframe interval: GOP={gop_size}, "
                               f"keyframes={len(keyframe_indices)}, tolerance=±{tolerance}",
                "description": "Keyframe interval check passed"
            })


class FrameCountRule(Rule):
    """
    Counts decoded frames in the output video with ffprobe -count_frames.
    Compares against the expected count from -frames:v N in the test case keys.
    Detects frame loss, early encoder termination, muxer issues.
    Applied only when listed in case "rules".
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Frame count does not match expected value",
            description="Count frames with ffprobe -count_frames and compare to -frames:v from keys"
        )

    def should_be_executed(self):
        # Cache parsed value so apply() doesn't repeat the regex
        match = re.search(r"-frames:v\s+(\d+)", self.case.get("keys", ""))
        if match:
            self._expected_frames = int(match.group(1))
            return True
        return False

    def _get_frame_count(self, ffprobe_exe, output_video):
        cmd = (f'"{ffprobe_exe}" -v error -count_frames -select_streams v:0'
               f' -show_entries stream=nb_read_frames'
               f' -of default=noprint_wrappers=1 "{output_video}"')
        logger.info(f"Running frame count: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300, shell=True)
            match = re.search(r"nb_read_frames=(\d+)", result.stdout)
            if match:
                count = int(match.group(1))
                logger.info(f"Frame count: {count}")
                return count
            logger.error("Could not parse nb_read_frames")
            return None
        except Exception as e:
            logger.error(f"Frame count error: {e}")
            return None

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("FrameCountRule: skipped - output video not produced")
            return

        actual = self._get_frame_count(context["ffprobe_exe"], context["output_video"])
        if actual is None:
            self.add_error("Frame count could not be determined")
            return

        if actual != self._expected_frames:
            self.add_error(
                f"Frame count mismatch: expected={self._expected_frames}, actual={actual}"
            )
        else:
            logger.info(f"FrameCountRule: {actual} frames - OK")
            self.json_content["message"].append({
                "issue":       f"Frame count: {actual} frames - OK",
                "description": "Frame count matches -frames:v value"
            })
