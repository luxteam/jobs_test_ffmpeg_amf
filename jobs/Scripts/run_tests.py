"""
Main test runner for FFMPEG AMF tests.

Writes output in jobs_launcher-compatible format:
  <output>/test_cases.json          - list of all cases with status
  <output>/<case>_RPR.json          - per-case result (list of one object)
  <output>/Color/<case>/            - extracted frames for report
  <output>/report_compare.json      - collected for build_reports.bat

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
    Safe to call multiple times - adds handlers only once per logger name.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run_tests.log")

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

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
        "format_info":              {},
        "ffmpeg_command":           "",
        # ffmpeg_keys: the raw "keys" string from the test case (shown in Info column)
        "ffmpeg_keys":                    case.get("keys", ""),
        "reference_keys":                 case.get("reference_keys", ""),
        "input_video_keys":               case.get("input_video_keys", ""),
        "input_video":                    case.get("input_video", ""),
        "reference_input_video_keys":     case.get("reference_input_video_keys", ""),
        "reference_input_video":          case.get("reference_input_video", ""),
        # expected_metadata: from test case (shown in Info column)
        "expected_metadata":              case.get("expected_metadata", {}),
        # screens_collection: populated after frame extraction for the Frames carousel
        "screens_collection":       [],
        # Log paths stored relative to results-data/ (NOT in POSSIBLE_JSON_LOG_KEYS,
        # so not path-rewritten - same pattern as streaming_sdk server_log/client_log).
        # render_log is left empty to avoid its path-rewriting side-effects.
        "ffmpeg_conversion_log":    "",
        "psnr_log":                 "",
        "ssim_log":                 "",
        "psnr_reference":           None,
        "ssim_reference":           None,
        "psnr_reference_log":       "",
        "ssim_reference_log":       "",
    }


def write_case_report(output_dir, report):
    path = os.path.join(output_dir, report["test_case"] + CASE_REPORT_SUFFIX)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([report], f, indent=4)


def write_test_cases_json(output_dir, cases):
    """Write test_cases.json - jobs_launcher needs this for report generation."""
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
        # "cpu" omitted intentionally - columns_template.html hides the CPU line
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
    # result_path must contain at least one dash - build_summary_reports splits
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
                    video_samples_dir,
                    gpu_name, test_group, render_version, logger,
                    compare_ffmpeg_exe=None):
    # Compare build used for reference_keys ("compare cases") encodes;
    # falls back to the main build when no separate compare build is given.
    compare_ffmpeg_exe = compare_ffmpeg_exe or ffmpeg_exe
    case_name = case["case"]
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    report = make_case_report(case, output_dir, gpu_name, test_group, render_version)

    # ---- 1. Resolve and verify input video ----
    input_video_path     = None
    has_reference        = False
    generated_input_path = None

    if "input_video" in case:
        input_file       = case["input_video"]
        has_reference    = "<input_video>" in case.get("keys", "")
        input_video_path = os.path.join(video_samples_dir, input_file) if has_reference else None
        report["scene_name"] = input_file
        if has_reference and not os.path.exists(input_video_path):
            report["test_status"] = "error"
            report["message"].append({
                "issue": f"Input video not found: {input_video_path}",
                "description": "Input video must exist before conversion"
            })
            logger.error(f"[{case_name}] Input video not found: {input_video_path}")
            return report
    elif "input_video_keys" in case:
        generated_input_path = fu.generate_input_video(
            case["input_video_keys"], ffmpeg_exe, case_output_dir, case_name, logger,
            ext=case.get("input_video_format", "mp4")
        )
        if generated_input_path is None:
            report["test_status"] = "error"
            report["message"].append({
                "issue": "Input video generation failed",
                "description": "generate_input_video returned None - see log for details"
            })
            return report
        input_video_path  = generated_input_path
        has_reference     = "<input_video>" in case.get("keys", "")
        report["scene_name"] = os.path.basename(generated_input_path)
    else:
        report["scene_name"] = "lavfi"

    # ---- 2. Run FFMPEG conversion ----
    output_format = case.get("output_format", "mp4")
    output_video  = os.path.join(case_output_dir, f"{case_name}_output.{output_format}")
    conversion_log = os.path.join(case_output_dir, f"{case_name}_conversion.log")
    _results_dir   = os.path.join(output_dir, RESULTS_SUBDIR)
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

    # ---- 2.5. Generate reference video (non-AMF) for quality comparison ----
    generated_reference_path       = None
    generated_reference_input_path = None
    reference_input_path           = None  # separate PSNR/SSIM baseline for reference encode

    if "reference_keys" in case:
        # Determine input for reference encode — separate source if specified, else same as main.
        # reference_input_video takes priority over reference_input_video_keys.
        ref_input_path = input_video_path

        if "reference_input_video" in case:
            candidate = os.path.join(video_samples_dir, case["reference_input_video"])
            if not os.path.exists(candidate):
                logger.error(f"[{case_name}] Reference input not found: {candidate}")
                report["test_status"] = "failed"
                report["message"].append({
                    "issue": f"Reference input video not found: {candidate}",
                    "description": "reference_keys comparison will be skipped"
                })
                ref_input_path = None
            else:
                ref_input_path       = candidate
                reference_input_path = candidate
        elif "reference_input_video_keys" in case:
            generated_reference_input_path = fu.generate_input_video(
                case["reference_input_video_keys"], ffmpeg_exe, case_output_dir, case_name, logger,
                suffix="_ref_input", ext=case.get("reference_input_video_format", "mp4")
            )
            if generated_reference_input_path is None:
                logger.error(f"[{case_name}] Reference input generation failed")
                report["test_status"] = "failed"
                report["message"].append({
                    "issue": "Reference input video generation failed",
                    "description": "reference_keys comparison will be skipped"
                })
            ref_input_path       = generated_reference_input_path
            reference_input_path = generated_reference_input_path

        if ref_input_path:
            reference_output = os.path.join(case_output_dir, f"{case_name}_reference.mp4")
            ref_case         = {"keys": case["reference_keys"]}
            ref_log          = os.path.join(case_output_dir, f"{case_name}_reference.log")
            ref_returncode   = fu.run_conversion(
                compare_ffmpeg_exe, ref_input_path, reference_output, ref_case, ref_log
            )
            if ref_returncode == 0 and os.path.exists(reference_output):
                generated_reference_path = reference_output
                logger.info(f"[{case_name}] Reference video generated: {reference_output}")
            else:
                logger.error(f"[{case_name}] Reference generation failed (exit {ref_returncode})")
                report["test_status"] = "failed"
                report["message"].append({
                    "issue": f"Reference video generation failed (exit {ref_returncode})",
                    "description": "psnr_rule/ssim_rule reference comparison will be skipped"
                })

    # ---- 3. Build context and apply rules ----
    # Each rule is responsible for its own data collection (ffprobe, psnr filter, etc.)
    # and writes its results directly into the report.
    # psnr_log path is pre-defined here so PSNRRule writes it to a known location,
    # allowing frame extraction below to find it after rules complete.
    context = {
        "ffmpeg_exe":     ffmpeg_exe,
        "ffprobe_exe":    ffprobe_exe,
        # VMAF-capable ffmpeg (libvmaf is not in the AMF build) — used by vmaf_rule.
        "compare_ffmpeg_exe": compare_ffmpeg_exe,
        "input_video":    input_video_path,
        "output_video":   output_video,
        "output_exists":  os.path.exists(output_video),
        "returncode":     returncode,
        "has_reference":  has_reference,
        "psnr_log":            os.path.join(case_output_dir, f"{case_name}_psnr.log"),
        "ssim_log":            os.path.join(case_output_dir, f"{case_name}_ssim.log"),
        "reference_video":       generated_reference_path,
        "reference_input_video": reference_input_path,
        "reference_psnr_log":    os.path.join(case_output_dir, f"{case_name}_psnr_reference.log"),
        "reference_ssim_log":    os.path.join(case_output_dir, f"{case_name}_ssim_reference.log"),
        "results_dir":         _results_dir,
    }

    if report["test_status"] == "error":   # still at init value — promote to passed
        report["test_status"] = "passed"   # rules will downgrade if needed
    processor = RulesProcessor(case, report)
    processor.process(context)

    logger.info(f"[{case_name}] Final status: {report['test_status']}")

    # ---- 4. Extract worst frames for visual comparison ----
    # Requires the PSNR log written by PSNRRule. Skipped if PSNRRule wasn't listed,
    # if there is no reference input, or if the output video was not produced.
    frames_dir = os.path.join(output_dir, FRAMES_DIR, case_name)
    psnr_log   = context["psnr_log"]
    if has_reference and os.path.exists(output_video) and os.path.exists(psnr_log):
        try:
            worst_frames = fu.extract_worst_frames(
                input_video_path, output_video, frames_dir, count=5, psnr_log=psnr_log
            )
            report["worst_frames"] = worst_frames

            screens = []
            for wf in worst_frames:
                imgs = wf.get("images", {})
                frame_num  = wf.get("frame_number", "?")
                frame_psnr = wf.get("psnr")
                psnr_suffix = f" PSNR={frame_psnr:.2f}" if isinstance(frame_psnr, float) else ""
                for img_key in ("output", "input", "diff_scaled", "diff_thresh"):
                    abs_path = imgs.get(img_key)
                    if abs_path and os.path.exists(abs_path):
                        rel = os.path.relpath(abs_path, _results_dir).replace("\\", "/")
                        screens.append({
                            "path":     rel,
                            "thumb256": rel,
                            "thumb128": rel,
                            "name":     f"Frame #{frame_num} {img_key}{psnr_suffix}",
                        })
            report["screens_collection"] = screens

            if worst_frames:
                first_img = worst_frames[0].get("images", {}).get("output")
                if first_img and os.path.exists(first_img):
                    report["render_color_path"] = os.path.relpath(
                        first_img, _results_dir
                    ).replace("\\", "/")
        except Exception as e:
            logger.warning(f"[{case_name}] Frame extraction failed: {e}")

    # ---- 5. Cleanup generated videos ----
    for cleanup_path in (output_video, generated_reference_path,
                         generated_reference_input_path, generated_input_path):
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
                logger.info(f"[{case_name}] Removed: {cleanup_path}")
            except Exception as e:
                logger.warning(f"[{case_name}] Could not remove {cleanup_path}: {e}")

    return report


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(args):
    """
    Execute the full test run given a populated args namespace.
    Returns int exit code (0 = all passed, 1 = failures/errors).
    """
    logger = setup_logging(args.output)

    # Resolve the ffmpeg build directory. Either supplied via --build_path, or
    # produced on the fly by the in-framework build stage when --auto_build is set.
    if not args.build_path:
        if getattr(args, "auto_build", False):
            import build_ffmpeg
            logger.info("No --build_path given; running in-framework build stage (--auto_build)")
            try:
                args.build_path = build_ffmpeg.build(
                    amf_ffmpeg_dir=(args.amf_ffmpeg_dir or None),
                    build_type=args.build_type,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"In-framework ffmpeg build failed: {e}")
                return 1
        else:
            logger.error("No ffmpeg build available: pass --build_path <dir> or use --auto_build")
            return 1

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

    for exe in (ffmpeg_exe, ffprobe_exe):
        if not os.path.exists(exe):
            logger.error(f"{os.path.basename(exe)} not found at: {exe}")
            return 1

    # Compare build for "compare cases" (reference_keys encodes). Defaults to
    # the main build when --compare_build_path is not supplied.
    compare_build      = args.compare_build_path or args.build_path
    compare_ffmpeg_exe = fu.get_ffmpeg_path(compare_build)
    if not os.path.exists(compare_ffmpeg_exe):
        logger.error(f"compare ffmpeg not found at: {compare_ffmpeg_exe}")
        return 1
    if compare_build != args.build_path:
        logger.info(f"  Compare build:  {compare_build} "
                    f"(version {fu.get_ffmpeg_version(compare_ffmpeg_exe)})")

    render_version = fu.get_ffmpeg_version(ffmpeg_exe)
    test_group     = os.path.splitext(os.path.basename(args.test_pack))[0]
    group_dir      = os.path.join(args.output, test_group)
    logger.info(f"  FFmpeg version: {render_version}")
    logger.info(f"  Test group:     {test_group}")

    try:
        cases, _ = load_test_pack(args.test_pack)
    except Exception as e:
        logger.error(f"Failed to load test pack: {e}")
        return 1

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
                args.video_samples,
                args.gpu_name, test_group, render_version, logger,
                compare_ffmpeg_exe=compare_ffmpeg_exe
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
    parser.add_argument("--build_path",     required=False, default="",
                        help="Path to ffmpeg build directory (ffmpeg + ffprobe). "
                             "If omitted, pass --auto_build to build it in-framework.")
    parser.add_argument("--compare_build_path", default="",
                        help="ffmpeg build for compare/reference encodes "
                             "(default: same as --build_path)")
    parser.add_argument("--auto_build",      action="store_true",
                        help="Build ffmpeg+AMF in-framework via build/ scripts when --build_path is omitted")
    parser.add_argument("--amf_ffmpeg_dir",  default="",
                        help="Path to amf-ffmpeg sources/scripts for --auto_build (default: auto-detect)")
    parser.add_argument("--build_type",      default="release", choices=["debug", "release"],
                        help="ffmpeg build type for --auto_build (default: release)")
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
