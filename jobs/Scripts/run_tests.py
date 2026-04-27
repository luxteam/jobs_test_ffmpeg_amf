"""
Main test runner for FFMPEG AMF tests.

Writes output in jobs_launcher-compatible format:
  <output>/test_cases.json          - list of all cases with status
  <output>/<case>_RPR.json          - per-case result (list of one object)
  <output>/Color/<case>/            - extracted frames for report
  <output>/report_compare.json      - collected for build_reports.bat

Called directly by run_local.py (local mode) or via Jenkins pipeline (CI mode).
See run_local.py for local usage; see pipelines/amfdev_ffmpeg_amf.groovy for CI usage.

Each test case in the test pack carries its own "input_video" filename.
The runner resolves the full path as: video_samples / case["input_video"].

CLI usage (both modes share the same arguments):
    python run_tests.py
        --build_path     <path to ffmpeg build directory>
        --video_samples  <folder containing input video files>
        --test_pack      <path to test pack JSON file>
        --output         <output directory>
        [--test_cases    <comma-separated case names, empty = all>]
        [--gpu_name      <GPU name string for report>]
"""

import argparse
import json
import logging
import os
import platform
import re
import sys
import traceback
import uuid
from datetime import datetime

import ffmpeg_utils as fu
from rules.rules_processor import RulesProcessor

CASE_REPORT_SUFFIX = "_RPR.json"
FRAMES_DIR = "Color"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir, logger_name=__name__):
    """
    Configure file + stdout logging.
    Safe to call multiple times — adds handlers only once per logger name.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run_tests.log")

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger  # already configured (e.g. called from run_local.py)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# jobs_launcher-compatible result helpers
# ---------------------------------------------------------------------------

def make_case_report(case, output_dir, gpu_name, test_group="", render_version=""):
    """Initial per-case report dict matching jobs_launcher schema."""
    return {
        # --- jobs_launcher required fields ---
        "test_case":                case["case"],
        "test_group":               test_group,
        "test_status":              "error",        # overwritten on success/skip
        "render_device":            gpu_name,
        "tool":                     "FFmpeg AMF",
        "render_version":           render_version,
        "core_version":             "",
        "render_time":              0.0,
        "execution_time":           0.0,
        "sync_time":                0.0,
        "date_time":                datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
        "number_of_tries":          1,
        "message":                  [],
        "group_timeout_exceeded":   False,
        "testcase_timeout_exceeded": False,
        "scene_name":               "",             # set to input video filename
        "render_mode":              "",
        "file_name":                "",             # set to output video filename
        "render_color_path":        "",
        "render_log":               "",
        "error_screen_path":        "",
        "render_start_time":        "",
        "render_end_time":          "",
        "case_functions":           "",
        "testcase_timeout":         0,
        "difference_color":         -0.0,
        "difference_time":          -0.0,
        "difference_color_2":       -0,
        "has_time_diff":            False,
        "script_info":              case.get("description", []),
        "screens_path":             os.path.abspath(os.path.join(output_dir, FRAMES_DIR, case["case"])),
        # --- custom fields (pass through jobs_launcher transparently) ---
        "psnr":                     None,
        "ssim":                     None,
        "metadata":                 {},
        "ffmpeg_command":           "",
        # ffmpeg_keys: the raw "keys" string from the test case (shown in Info column)
        "ffmpeg_keys":              case.get("keys", ""),
        # expected_metadata: from test case (shown in Info column)
        "expected_metadata":        case.get("expected_metadata", {}),
        # screens_collection: populated after frame extraction for the Frames carousel
        "screens_collection":       [],
        # Log paths stored relative to results-data/ (NOT in POSSIBLE_JSON_LOG_KEYS,
        # so not path-rewritten — same pattern as streaming_sdk server_log/client_log).
        # render_log is left empty to avoid its path-rewriting side-effects.
        "ffmpeg_conversion_log":    "",
        "psnr_log":                 "",
        "ssim_log":                 "",
    }


def write_case_report(output_dir, report):
    path = os.path.join(output_dir, report["test_case"] + CASE_REPORT_SUFFIX)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([report], f, indent=4)


def write_test_cases_json(output_dir, cases):
    """Write test_cases.json — jobs_launcher needs this for report generation."""
    path = os.path.join(output_dir, "test_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4)


RESULTS_SUBDIR = "results-data"


def write_report_compare_json(output_dir, reports):
    """
    Write report_compare.json into the results-data/ subdir.
    jobs_launcher's build_local_reports reads report_compare.json from
    session_report's result_path, which must contain at least one dash.
    """
    subdir = os.path.join(output_dir, RESULTS_SUBDIR)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "report_compare.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)


def _get_os_string():
    """Match jobs_launcher core.system_info.get_os() output format."""
    custom = os.getenv("CIS_OS")
    if custom:
        return custom
    if platform.system() == "Windows":
        return "{} {}({})".format(platform.system(), platform.release(), platform.architecture()[0])
    return platform.system()



def write_session_report(output_dir, all_reports, test_group, gpu_name):
    """
    Write session_report.json in jobs_launcher format.
    build_summary_reports() scans for this file to generate summary_report.html.
    """
    os_str = _get_os_string()

    passed   = sum(1 for r in all_reports if r["test_status"] == "passed")
    failed   = sum(1 for r in all_reports if r["test_status"] == "failed")
    errors   = sum(1 for r in all_reports if r["test_status"] == "error")
    skipped  = sum(1 for r in all_reports if r["test_status"] == "skipped")
    observed = sum(1 for r in all_reports if r["test_status"] == "observed")
    total    = passed + failed + errors + skipped + observed
    duration  = sum(r.get("render_time",    0.0) for r in all_reports)
    exec_time = sum(r.get("execution_time", 0.0) for r in all_reports)

    render_version = all_reports[0]["render_version"] if all_reports else ""

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        ram_gb = 0.0

    machine_info = {
        "render_device":  gpu_name,
        "os":             os_str,
        "tool":           "FFmpeg AMF",
        "render_version": render_version,
        "core_version":   "",
        "reporting_date": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
        "host":           platform.node(),
        "driver_version": "",
        "driver":         "",
        "newest_driver":  "",
        # "cpu" omitted intentionally — columns_template.html hides the CPU line
        # when machine_info.cpu is undefined (streaming_sdk report_type branch)
        "cpu_count":      str(os.cpu_count() or 0),
        "ram":            ram_gb,
    }

    render_results = list(all_reports)

    counts = {
        "total": total, "passed": passed, "failed": failed,
        "observed": observed, "error": errors, "skipped": skipped,
        "duration": duration, "render_duration": duration,
        "synchronization_duration": 0.0, "execution_time": exec_time,
    }

    # jobs_launcher hardcodes results[test_package][""] (empty string) as the
    # second-level config key in build_summary_report (line 967-971).
    # result_path must contain at least one dash — build_summary_reports splits
    # it on "-" to populate summary["result_path"] (line 1664-1665).
    session = {
        "machine_info": machine_info,
        "results": {
            test_group: {
                "": dict(
                    result_path=RESULTS_SUBDIR,
                    render_results=render_results,
                    machine_info=machine_info,
                    **counts,
                )
            }
        },
        "guid":         str(uuid.uuid4()),
        "failed_tests": [],
        "summary":      counts,
    }

    path = os.path.join(output_dir, "session_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=4)


def _strip_json_comments(text):
    """Remove // line comments from JSON-with-comments text."""
    return re.sub(r"//[^\n]*", "", text)


def load_test_pack(path):
    """
    Load a test pack JSON file.  Supports // line comments.
    Returns (cases_list, pack_meta_dict).
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    data = json.loads(_strip_json_comments(text))
    if isinstance(data, list):
        return data, {}
    cases = data.get("cases", [])
    meta  = {k: v for k, v in data.items() if k != "cases"}
    return cases, meta


# ---------------------------------------------------------------------------
# Single test case execution
# ---------------------------------------------------------------------------

def run_single_case(case, output_dir, ffmpeg_exe, ffprobe_exe,
                    video_samples_dir, default_input_video,
                    gpu_name, test_group, render_version, logger):
    case_name = case["case"]
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    report = make_case_report(case, output_dir, gpu_name, test_group, render_version)

    # ---- 0. Populate Info column fields ----
    # ffmpeg_keys and expected_metadata are already set in make_case_report.
    # script_info holds the human-readable description lines shown under "Description:".
    report["script_info"] = case.get("description", [])

    # ---- 1. Resolve and verify input video ----
    input_file = case.get("input_video")
    has_reference = bool(input_file) and "<input_video>" in case.get("keys", "")
    input_video_path = os.path.join(video_samples_dir, input_file) if has_reference else None
    report["scene_name"] = input_file or "lavfi"

    if has_reference and not os.path.exists(input_video_path):
        report["test_status"] = "error"
        report["message"].append({
            "issue": f"Input video not found: {input_video_path}",
            "description": "Input video must exist before conversion"
        })
        logger.error(f"[{case_name}] Input video not found: {input_video_path}")
        return report

    # ---- 2. Run FFMPEG conversion ----
    output_video   = os.path.join(case_output_dir, f"{case_name}_output.mp4")
    conversion_log = os.path.join(case_output_dir, f"{case_name}_conversion.log")
    # Store log paths relative to results-data/ (same pattern as streaming_sdk server_log).
    # These fields are NOT in POSSIBLE_JSON_LOG_KEYS so they are never path-rewritten.
    # The HTML report lives at results-data/report.html, so paths like ../hwaccel_1/...
    # resolve correctly from the browser.
    _results_dir = os.path.join(output_dir, RESULTS_SUBDIR)
    report["ffmpeg_conversion_log"] = os.path.relpath(conversion_log, _results_dir).replace("\\", "/")

    cmd = fu.build_conversion_command(ffmpeg_exe, input_video_path, output_video, case)
    report["ffmpeg_command"] = cmd
    report["file_name"]      = os.path.basename(output_video)
    logger.info(f"[{case_name}] Command: {cmd}")

    start_time = datetime.now()
    returncode = fu.run_conversion(
        ffmpeg_exe, input_video_path, output_video, case, conversion_log
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    report["render_time"]    = elapsed
    report["execution_time"] = elapsed
    logger.info(f"[{case_name}] Conversion done in {elapsed:.1f}s, exit={returncode}")

    # ---- 3. Get metadata with ffprobe ----
    metadata = {}
    if os.path.exists(output_video):
        metadata = fu.get_video_metadata(ffprobe_exe, output_video)
        report["metadata"] = metadata

    psnr_log = os.path.join(case_output_dir, f"{case_name}_psnr.log")

    # ---- 4. Measure PSNR ----
    psnr = None
    if has_reference and os.path.exists(output_video):
        psnr = fu.measure_psnr(ffmpeg_exe, input_video_path, output_video, psnr_log)
        report["psnr"] = psnr
        report["psnr_log"] = os.path.relpath(psnr_log, _results_dir).replace("\\", "/")

    # ---- 5. Measure SSIM ----
    ssim = None
    ssim_log_path = os.path.join(case_output_dir, f"{case_name}_ssim.log")
    if has_reference and os.path.exists(output_video):
        ssim = fu.measure_ssim(ffmpeg_exe, input_video_path, output_video, ssim_log_path)
        report["ssim"] = ssim
        report["ssim_log"] = os.path.relpath(ssim_log_path, _results_dir).replace("\\", "/")

    # ---- 6. Extract worst frames for visual comparison ----
    # ffmpeg psnr stats log (written by measure_psnr) is parsed to find worst N
    # frames; cv2 seeks directly to those indices to save quad images.
    frames_dir = os.path.join(output_dir, FRAMES_DIR, case_name)
    if has_reference and os.path.exists(output_video):
        try:
            worst_frames = fu.extract_worst_frames(
                input_video_path, output_video, frames_dir, count=5, psnr_log=psnr_log
            )
            report["worst_frames"] = worst_frames

            # Build screens_collection for the Frames carousel column.
            # Each entry needs: path, thumb256, thumb128, name.
            # We use the same image for path and thumbs (no separate thumbnails generated).
            screens = []
            for wf in worst_frames:
                imgs = wf.get("images", {})
                frame_num = wf.get("frame_number", "?")
                frame_psnr = wf.get("psnr")
                psnr_suffix = f" PSNR={frame_psnr:.2f}" if isinstance(frame_psnr, float) else ""
                for img_key in ("output", "input", "diff_scaled", "diff_thresh"):
                    abs_path = imgs.get(img_key)
                    if abs_path and os.path.exists(abs_path):
                        rel = os.path.relpath(abs_path, _results_dir).replace("\\", "/")
                        screens.append({
                            "path":    rel,
                            "thumb256": rel,
                            "thumb128": rel,
                            "name":    f"Frame #{frame_num} {img_key}{psnr_suffix}",
                        })
            report["screens_collection"] = screens

            # render_color_path: first output image for Compare tab
            if worst_frames:
                first_img = worst_frames[0].get("images", {}).get("output")
                if first_img and os.path.exists(first_img):
                    report["render_color_path"] = os.path.relpath(
                        first_img, _results_dir
                    ).replace("\\", "/")
        except Exception as e:
            logger.warning(f"[{case_name}] Frame extraction failed: {e}")

    # ---- 7. Add ffprobe metadata + quality metrics to Message column ----
    # ffprobe metadata (actual output video properties)
    if metadata:
        meta_parts = []
        for k, v in metadata.items():
            meta_parts.append(f"{k}: {v}")
        if meta_parts:
            report["message"].append({
                "issue":       "Output video metadata: " + ", ".join(meta_parts),
                "description": "ffprobe output",
            })

    # PSNR / SSIM quality metrics
    if psnr is not None or ssim is not None:
        parts = []
        if psnr is not None:
            parts.append(f"PSNR: {psnr:.2f} dB")
        if ssim is not None:
            parts.append(f"SSIM: {ssim:.4f}")
        report["message"].append({
            "issue":       ", ".join(parts),
            "description": "Quality metrics",
        })

    # ---- 9. Apply rules ----
    report["test_status"] = "passed"   # rules will downgrade if needed
    data = {
        "ffmpeg_returncode": returncode,
        "metadata":          metadata,
        "psnr":              psnr,
        "ssim":              ssim,
    }
    processor = RulesProcessor(case, report)
    processor.process(data)

    logger.info(f"[{case_name}] Final status: {report['test_status']}")
    return report


# ---------------------------------------------------------------------------
# Core runner — called by both CLI (main) and run_local.py (run)
# ---------------------------------------------------------------------------

def run(args):
    """
    Execute the full test run given a populated args namespace.
    Returns int exit code (0 = all passed, 1 = failures/errors).
    Usable directly from run_local.py without spawning a subprocess.
    """
    logger = setup_logging(args.output)
    logger.info("=" * 60)
    logger.info("FFMPEG AMF Test Runner started")
    logger.info(f"  Build:          {args.build_path}")
    logger.info(f"  Video samples:  {args.video_samples}")
    logger.info(f"  Test pack:      {args.test_pack}")
    logger.info(f"  Output:         {args.output}")
    logger.info(f"  GPU:            {args.gpu_name}")
    logger.info("=" * 60)

    ffmpeg_exe  = fu.get_ffmpeg_path(args.build_path)
    ffprobe_exe = fu.get_ffprobe_path(args.build_path)

    for exe, name in ((ffmpeg_exe, "ffmpeg.exe"), (ffprobe_exe, "ffprobe.exe")):
        if not os.path.exists(exe):
            logger.error(f"{name} not found at: {exe}")
            return 1

    render_version = fu.get_ffmpeg_version(ffmpeg_exe)
    test_group     = os.path.splitext(os.path.basename(args.test_pack))[0]
    group_dir      = os.path.join(args.output, test_group)
    logger.info(f"  FFmpeg version: {render_version}")
    logger.info(f"  Test group:     {test_group}")

    try:
        cases, pack_meta = load_test_pack(args.test_pack)
    except Exception as e:
        logger.error(f"Failed to load test pack: {e}")
        return 1

    default_input_video = "default_input_video.mp4"

    if getattr(args, "test_cases", ""):
        selected = {c.strip() for c in args.test_cases.split(",") if c.strip()}
        cases    = [c for c in cases if c["case"] in selected]
        logger.info(f"Running selected cases: {sorted(selected)}")

    logger.info(f"Total cases: {len(cases)}")

    all_reports       = []
    cases_with_status = []

    for case in cases:
        case_copy = dict(case)

        if case.get("status") == "skipped":
            logger.info(f"[{case['case']}] Skipped")
            report = make_case_report(case, group_dir, args.gpu_name, test_group, render_version)
            report["test_status"]            = "skipped"
            report["group_timeout_exceeded"] = False
            write_case_report(group_dir, report)
            all_reports.append(report)
            case_copy["status"] = "skipped"
            cases_with_status.append(case_copy)
            continue

        logger.info(f"\n{'-' * 50}\nRunning: {case['case']}")
        try:
            report = run_single_case(
                case, group_dir,
                ffmpeg_exe, ffprobe_exe,
                args.video_samples, default_input_video,
                args.gpu_name, test_group, render_version, logger
            )
        except Exception as e:
            logger.error(f"Case {case['case']} crashed: {e}\n{traceback.format_exc()}")
            report = make_case_report(case, group_dir, args.gpu_name, test_group, render_version)
            report["test_status"] = "error"
            report["message"].append({
                "issue":       f"Unexpected crash: {e}",
                "description": "Unhandled exception in test runner"
            })

        write_case_report(group_dir, report)
        all_reports.append(report)
        case_copy["status"] = report["test_status"]
        cases_with_status.append(case_copy)

    write_test_cases_json(group_dir, cases_with_status)
    write_report_compare_json(group_dir, all_reports)
    write_session_report(group_dir, all_reports, test_group, args.gpu_name)

    passed  = sum(1 for r in all_reports if r["test_status"] == "passed")
    failed  = sum(1 for r in all_reports if r["test_status"] == "failed")
    errors  = sum(1 for r in all_reports if r["test_status"] == "error")
    skipped = sum(1 for r in all_reports if r["test_status"] == "skipped")
    logger.info(
        f"\nSummary: {passed} passed, {failed} failed, {errors} errors,"
        f" {skipped} skipped / {len(all_reports)} total"
    )
    logger.info(f"Results written to: {group_dir}")

    return 0  # test failures are reported via report files; exit 1 is reserved for setup errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FFMPEG AMF test runner")
    parser.add_argument("--build_path",     required=True,
                        help="Path to ffmpeg build directory (ffmpeg.exe + ffprobe.exe)")
    parser.add_argument("--video_samples",  required=True,
                        help="Folder containing input video files.")
    parser.add_argument("--test_pack",   required=True,
                        help="Path to test pack JSON file")
    parser.add_argument("--output",      required=True,
                        help="Output directory for results, logs, frames")
    parser.add_argument("--test_cases",  default="",
                        help="Comma-separated case names to run (empty = all)")
    parser.add_argument("--gpu_name",    default="Unknown GPU",
                        help="GPU name for report (e.g. 'AMD Radeon RX 7900 XTX')")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
