import json
import os
import re
import subprocess

from rules.rule import Rule
import logging

logger = logging.getLogger(__name__)


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


_STREAM_FIELDS = {"codec_name", "width", "height", "r_frame_rate", "avg_frame_rate", "pix_fmt"}


class MetadataRule(Rule):
    """
    Runs ffprobe on the output video, records metadata in the report,
    then checks that each field in expected_metadata matches.
    Reports each mismatched field individually.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Converted video metadata does not match expected values",
            description="Validate ffprobe metadata of output video against test case expected_metadata"
        )

    def should_be_executed(self):
        return "expected_metadata" in self.case and bool(self.case["expected_metadata"])

    def _get_metadata(self, ffprobe_exe, video_path):
        cmd = (f'"{ffprobe_exe}" -v quiet -select_streams v:0'
               f' -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt'
               f' -print_format json "{video_path}"')
        logger.info(f"Running ffprobe metadata: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=60, shell=True)
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if not streams:
                logger.error(f"No video streams found in {video_path}")
                return {}
            stream = streams[0]
            logger.info(f"Metadata: codec={stream.get('codec_name')}, "
                        f"{stream.get('width')}x{stream.get('height')}")
            return {k: v for k, v in stream.items() if k in _STREAM_FIELDS}
        except Exception as e:
            logger.error(f"ffprobe metadata error: {e}")
            return {}

    def apply(self, context):
        if not context.get("output_exists"):
            logger.info("MetadataRule: skipped — output video not produced")
            return

        actual = self._get_metadata(context["ffprobe_exe"], context["output_video"])
        self.json_content["metadata"] = actual

        if actual:
            meta_parts = [f"{k}: {v}" for k, v in actual.items()]
            self.json_content["message"].append({
                "issue":       "Output video metadata: " + ", ".join(meta_parts),
                "description": "ffprobe output",
            })

        expected = self.case.get("expected_metadata", {})
        all_ok = True
        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if actual_value is None:
                self.add_error(
                    f"Metadata field '{field}' not found in output video. Expected: {expected_value}"
                )
                all_ok = False
            elif str(actual_value) != str(expected_value):
                self.add_error(
                    f"Metadata mismatch for '{field}': expected={expected_value}, actual={actual_value}"
                )
                all_ok = False

        if all_ok:
            logger.info("MetadataRule: all metadata fields match")


class PSNRRule(Rule):
    """
    Measures PSNR between input and output video using ffmpeg psnr filter.
    Writes per-frame stats to psnr_log (used later by run_tests for worst-frame extraction).
    Skipped when has_reference is False (e.g. lavfi source) or conversion failed.
    Default threshold: 30.0 dB (overridable via case "psnr_threshold" field).
    """

    DEFAULT_THRESHOLD = 30.0

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
                return psnr
            logger.error("Could not parse PSNR")
            return None
        except Exception as e:
            logger.error(f"PSNR error: {e}")
            return None

    def apply(self, context):
        if not context.get("has_reference"):
            logger.info("PSNRRule: skipped — no reference input video (lavfi source)")
            return

        if not context.get("output_exists"):
            logger.info("PSNRRule: skipped — output video not produced")
            return

        if context.get("returncode") != 0:
            logger.info("PSNRRule: skipped — conversion failed, PSNR on corrupt output is not meaningful")
            return

        psnr_log = context["psnr_log"]
        psnr = self._measure_psnr(
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
            self.add_error("PSNR measurement failed — no value returned by ffmpeg")
            return

        if psnr == float("inf"):
            logger.info(f"PSNRRule: PSNR is infinite (lossless or identical), threshold={threshold}")
        elif psnr < threshold:
            self.add_error(
                f"Unacceptable PSNR: {psnr:.2f} dB (threshold: {threshold} dB)"
            )
        else:
            logger.info(f"PSNRRule: PSNR={psnr:.2f} dB >= threshold={threshold} dB — OK")

        self.json_content["message"].append({
            "issue":       f"PSNR: {psnr:.2f} dB",
            "description": "Quality metrics",
        })


class SSIMRule(Rule):
    """
    Measures SSIM between input and output video using ffmpeg ssim filter.
    Skipped when has_reference is False (e.g. lavfi source) or conversion failed.
    Range: 0.0 to 1.0. Default threshold: 0.9 (overridable via case "ssim_threshold" field).
    """

    DEFAULT_THRESHOLD = 0.9

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
                return ssim
            logger.error("Could not parse SSIM")
            return None
        except Exception as e:
            logger.error(f"SSIM error: {e}")
            return None

    def apply(self, context):
        if not context.get("has_reference"):
            logger.info("SSIMRule: skipped — no reference input video (lavfi source)")
            return

        if not context.get("output_exists"):
            logger.info("SSIMRule: skipped — output video not produced")
            return

        if context.get("returncode") != 0:
            logger.info("SSIMRule: skipped — conversion failed, SSIM on corrupt output is not meaningful")
            return

        ssim_log = context["ssim_log"]
        ssim = self._measure_ssim(
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
            self.add_error("SSIM measurement failed — no value returned by ffmpeg")
            return

        if ssim < threshold:
            self.add_error(
                f"Unacceptable SSIM: {ssim:.4f} (threshold: {threshold})"
            )
        else:
            logger.info(f"SSIMRule: SSIM={ssim:.4f} >= threshold={threshold} — OK")

        self.json_content["message"].append({
            "issue":       f"SSIM: {ssim:.4f}",
            "description": "Quality metrics",
        })


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
            logger.info("DecodeRule: skipped — output video not produced")
            return

        decode_errors = self._decode_check(context["ffmpeg_exe"], context["output_video"])
        if decode_errors:
            message = f"Decode errors detected: {decode_errors}"
            self.add_error(message if len(message) <= 1000 else message[:400] + "..." + message[-400:])
        else:
            logger.info("DecodeRule: no decode errors")
            self.json_content["message"].append({
                "issue":       "No decode errors",
                "description": "Decode check passed",
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
    Verifies the output container format and duration via ffprobe -show_entries format.
    Format: maps output_format extension → ffprobe format_name token via _FORMAT_NAME_MAP
            (e.g. "mkv" → "matroska" checked against "matroska,webm").
    Duration: if -frames:v N and rate=R are parseable from keys,
              checks actual duration is within 1 frame (1/R s) of N/R.
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
        cmd = (f'"{ffprobe_exe}" -v error -show_entries format=format_name,duration,size'
               f' -of default=noprint_wrappers=1 "{output_video}"')
        logger.info(f"Running format info: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=60, shell=True)
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
            logger.info("FormatRule: skipped — output video not produced")
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
            logger.info(f"FormatRule: format OK — {actual_output_format}")
            self.json_content["message"].append({
                "issue":       f"Container format: {actual_output_format}",
                "description": "Format check passed",
            })

        # Duration check
        keys         = self.case.get("keys", "")
        frames_match = re.search(r"-frames:v\s+(\d+)", keys)
        rate_match   = re.search(r"rate=(\d+(?:\.\d+)?)", keys)

        if not (frames_match and rate_match):
            logger.info("FormatRule: duration check skipped — -frames:v or rate not found in keys")
            return

        n    = int(frames_match.group(1))
        rate = float(rate_match.group(1))
        expected_duration = n / rate
        tolerance         = 1.0 / rate

        try:
            actual_duration = float(info.get("duration", -1))
        except (ValueError, TypeError):
            actual_duration = -1

        if actual_duration < 0:
            self.add_error("Could not retrieve duration from ffprobe")
        elif abs(actual_duration - expected_duration) > tolerance:
            self.add_error(
                f"Duration mismatch: expected {expected_duration:.3f}s, "
                f"actual={actual_duration:.3f}s (tolerance={tolerance:.3f}s)"
            )
        else:
            logger.info(f"FormatRule: duration OK — {actual_duration:.3f}s ≈ {expected_duration:.3f}s")
            self.json_content["message"].append({
                "issue":       f"Duration: {actual_duration:.3f}s (expected: {expected_duration:.3f}s)",
                "description": "Duration check passed",
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
            logger.info("FrameCountRule: skipped — output video not produced")
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
            logger.info(f"FrameCountRule: {actual} frames — OK")
            self.json_content["message"].append({
                "issue":       f"Frame count: {actual} frames — OK",
                "description": "Frame count matches -frames:v value",
            })
