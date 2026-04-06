"""
Utilities for invoking ffmpeg/ffprobe and performing video analysis.

Frame extraction strategy:
  - ffmpeg psnr filter writes per-frame stats to a log file (test artifact)
  - Parse log to find worst N frames by lowest psnr_avg
  - cv2 seeks directly to those frame indices to save quad images
    (input / output / diff_scaled / diff_thresh)
"""

import json
import logging
import os
import re
import subprocess

import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_ffmpeg_path(build_path):
    return os.path.join(build_path, "ffmpeg.exe")


def get_ffprobe_path(build_path):
    return os.path.join(build_path, "ffprobe.exe")


def get_ffmpeg_version(ffmpeg_exe):
    """Return ffmpeg version string (e.g. 'N-117970-g...'). Empty string on failure."""
    cmd = f'"{ffmpeg_exe}" -version'
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=10, shell=True)
        match = re.search(r"ffmpeg version (\S+)", result.stdout + result.stderr)
        return match.group(1) if match else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def build_conversion_command(ffmpeg_exe, input_video, output_video, case):
    """
    Build ffmpeg conversion command string from test case parameters.

    Command structure:
      ffmpeg -hide_banner -y {case["keys"] with <input_video> substituted} {output}

    The "keys" field carries all required flags for the test case, including
    hwaccel selection, -i <input_video>, and encoder.  Example:
      "-hwaccel dxva2 -hwaccel_output_format dxva2_vld -i <input_video> -c:v h264_amf"
    """
    keys = case.get("keys", "").replace("<input_video>", f'"{input_video}"')
    return f'"{ffmpeg_exe}" -hide_banner -y {keys} "{output_video}"'


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def run_conversion(ffmpeg_exe, input_video, output_video, case, log_path):
    """Run FFMPEG conversion. Returns exit code. Saves stdout+stderr to log_path."""
    cmd = build_conversion_command(ffmpeg_exe, input_video, output_video, case)
    logger.info(f"Running conversion: {cmd}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                text=True, shell=True)

    logger.info(f"FFMPEG exit code: {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_video_metadata(ffprobe_exe, video_path):
    """
    Use ffprobe to extract video stream metadata.
    Returns a dict (codec_name, width, height, avg_frame_rate, pix_fmt).
    """
    cmd = (f'"{ffprobe_exe}" -v quiet -select_streams v:0'
           f' -show_entries stream=codec_name,width,height,avg_frame_rate,pix_fmt'
           f' -print_format json "{video_path}"')
    logger.info(f"Running ffprobe metadata: {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=60, shell=True)
        data = json.loads(result.stdout)
        stream = data.get("streams", [])[0]
        logger.info(f"Metadata: codec={stream.get('codec_name')}, "
                    f"{stream.get('width')}x{stream.get('height')}")
        return stream
    except Exception as e:
        logger.error(f"ffprobe metadata error: {e}")
        return {}


# ---------------------------------------------------------------------------
# PSNR / SSIM via ffmpeg filters (whole-video averages)
# ---------------------------------------------------------------------------

def measure_psnr(ffmpeg_exe, input_video, output_video, log_path):
    """
    Measure average PSNR between input and output using ffmpeg psnr filter.
    Per-frame stats are written to log_path as a test artifact.
    Returns float (dB) or None on failure.
    """
    stats_path = log_path.replace("\\", "/")
    cmd = (f'"{ffmpeg_exe}" -i "{input_video}" -i "{output_video}"'
           f' -lavfi "psnr=stats_file={stats_path}" -f null -')
    logger.info("Measuring PSNR (ffmpeg filter)")
    try:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True,
                                timeout=300, shell=True)
        match = re.search(r"average:(\S+)", result.stderr)
        if match:
            val = match.group(1)
            psnr = float("inf") if val == "inf" else float(val)
            logger.info(f"PSNR average: {psnr}")
            return psnr
        logger.error("Could not parse PSNR")
        return None
    except Exception as e:
        logger.error(f"PSNR error: {e}")
        return None


def measure_ssim(ffmpeg_exe, input_video, output_video, log_path):
    """
    Measure average SSIM between input and output using ffmpeg ssim filter.
    Returns float (0.0-1.0) or None on failure.
    """
    cmd = f'"{ffmpeg_exe}" -i "{input_video}" -i "{output_video}" -lavfi ssim -f null -'
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


# ---------------------------------------------------------------------------
# Per-frame worst-frame extraction
# ---------------------------------------------------------------------------

def _parse_psnr_log(log_path):
    """
    Parse ffmpeg psnr stats_file. Each line format:
      n:1 mse_avg:12.34 mse_y:... psnr_avg:38.21 psnr_y:...
    Returns list of {frame_index, mse, psnr} sorted by input order.
    """
    frames = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n     = int(re.search(r"n:(\d+)", line).group(1))
            mse   = float(re.search(r"mse_avg:([\d.]+)", line).group(1))
            p_str = re.search(r"psnr_avg:(\S+)", line).group(1)
            psnr  = float("inf") if p_str == "inf" else float(p_str)
            frames.append({"frame_index": n - 1, "mse": mse, "psnr": psnr})
    return frames


def _save_frame_quad(cap_input, cap_output, frame_idx, out_dir, prefix):
    """
    Save 4 images for a single frame index:
      <prefix>_<N>_input.jpg
      <prefix>_<N>_output.jpg
      <prefix>_<N>_diff_scaled.jpg    (normalized grayscale diff)
      <prefix>_<N>_diff_thresh.jpg    (binary threshold diff)
    Returns dict with image paths.
    """
    cap_input.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap_output.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret1, f1 = cap_input.read()
    ret2, f2 = cap_output.read()

    if not (ret1 and ret2):
        return {}

    if f1.shape != f2.shape:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))

    p_in   = os.path.join(out_dir, f"{prefix}_{frame_idx+1:04d}_input.jpg")
    p_out  = os.path.join(out_dir, f"{prefix}_{frame_idx+1:04d}_output.jpg")
    p_diff = os.path.join(out_dir, f"{prefix}_{frame_idx+1:04d}_diff_scaled.jpg")
    p_thr  = os.path.join(out_dir, f"{prefix}_{frame_idx+1:04d}_diff_thresh.jpg")

    cv2.imwrite(p_in,  f1)
    cv2.imwrite(p_out, f2)

    diff = cv2.absdiff(f1, f2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_norm = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, diff_thresh = cv2.threshold(diff_norm, 30, 255, cv2.THRESH_BINARY)

    cv2.imwrite(p_diff, diff_norm)
    cv2.imwrite(p_thr,  diff_thresh)

    return {
        "input":       p_in,
        "output":      p_out,
        "diff_scaled": p_diff,
        "diff_thresh": p_thr,
    }


def extract_worst_frames(input_video, output_video, out_dir, count=5, psnr_log=None):
    """
    Find the N frames with worst PSNR from the ffmpeg per-frame stats log,
    then use cv2 to seek directly to those frames and save quad images.

    psnr_log — path to the stats file written by measure_psnr (required).

    Returns list of dicts sorted worst→best:
      [{frame_index, frame_number, mse, psnr, images: {input, output, diff_scaled, diff_thresh}}, ...]
    """
    if not psnr_log or not os.path.exists(psnr_log):
        logger.error("PSNR log not available for worst-frame extraction")
        return []

    os.makedirs(out_dir, exist_ok=True)

    per_frame = _parse_psnr_log(psnr_log)
    if not per_frame:
        logger.error("PSNR log is empty or could not be parsed")
        return []

    worst = sorted(per_frame, key=lambda x: x["psnr"])[:count]

    cap_in  = cv2.VideoCapture(input_video)
    cap_out = cv2.VideoCapture(output_video)

    results = []
    for entry in worst:
        idx    = entry["frame_index"]
        images = _save_frame_quad(cap_in, cap_out, idx, out_dir, "frame")
        results.append({
            "frame_index":  idx,
            "frame_number": idx + 1,
            "mse":          round(entry["mse"],  3),
            "psnr":         round(entry["psnr"], 3),
            "images":       images,
        })
        logger.info(f"Worst frame #{idx+1}: MSE={entry['mse']:.3f}, PSNR={entry['psnr']:.3f}")

    cap_in.release()
    cap_out.release()
    return results
