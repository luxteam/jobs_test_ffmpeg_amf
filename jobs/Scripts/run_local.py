"""
Local runner for FFMPEG AMF tests - no Jenkins, no NAS required.

Mirrors the Jenkins pipeline stages exactly:
  Stage 1  Checkout      →  (not needed locally; scripts are already present)
  Stage 2  Copy build    →  skipped; build_path is read from local_config.json
  Stage 3  Run tests     →  run_tests.run(args)          ← shared with Jenkins
  Stage 4  Generate HTML →  generate_frames_html.generate()  ← shared with Jenkins
  Stage 5  build_reports →  build_reports.bat (optional, requires jobs_launcher)
  Stage 6  Summary       →  printed to console

────────────────────────────────────────────────────────────────────────────
DIFFERENCE TABLE  (Jenkins ↔ Local)
────────────────────────────────────────────────────────────────────────────
 Aspect              Jenkins                        Local
 ─────────────────── ────────────────────────────── ──────────────────────────
 build_path          workspace\\ffmpeg_build         local_config.json "build_path"
                     (xcopy'd from NAS)              (local directory, no copy)
 video_samples       --video_samples NAS\\VideoSamples  local_config.json
                     passed to run_tests.py          "video_samples_path" dir
 input video         per-case "input_video" field    same - each case resolves
                     resolved inside run_tests.py    its own file from video_samples
 output dir          WORKSPACE\\summaryTestResults   output_base\\YYYYMMDD_HHMMSS_<pack>
 jobs_launcher       checked out as submodule        optional; set "jobs_launcher_path"
 report publishing   Jenkins HTML Publisher plugin   open frames_report.html in browser
 artifacts           archiveArtifacts in workspace   all files in output dir
────────────────────────────────────────────────────────────────────────────

Usage (all options are optional - defaults come from local_config.json):

  # Use local_config.json next to this script (or two levels up):
  python run_local.py

  # Override config file location:
  python run_local.py --config C:\\myconfig.json

  # Override individual settings without editing the config:
  python run_local.py --build_path C:\\builds\\ffmpeg --pack FFMPEG_AMF_hwaccel.json

  # Run a single test case:
  python run_local.py --test_cases hwaccel_1

  # Run fully without a config file (all paths explicit):
  python run_local.py
      --build_path     C:\\builds\\ffmpeg
      --video_samples  C:\\Videos\\VideoSamples
      --pack           FFMPEG_AMF_hwaccel.json
      --output         C:\\test_output\\run1
      --gpu_name       "AMD Radeon RX 7900 XTX"
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate repo root and scripts dir regardless of cwd
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", ".."))
TESTS_DIR   = os.path.join(REPO_ROOT, "jobs", "Tests")

# Shared modules live in the same Scripts dir - add to path so imports work
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_tests
import generate_frames_html


# ---------------------------------------------------------------------------
# Console colours (Windows-compatible via ANSI; fall back gracefully)
# ---------------------------------------------------------------------------
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _print_header(text):
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")

def _print_stage(n, text):
    print(f"\n{BOLD}[Stage {n}]{RESET} {text}")

def _ok(text):    print(f"  {GREEN}✔{RESET}  {text}")
def _warn(text):  print(f"  {YELLOW}⚠{RESET}  {text}")
def _fail(text):  print(f"  {RED}✘{RESET}  {text}")
def _info(text):  print(f"  {CYAN}·{RESET}  {text}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _find_default_config():
    """
    Look for local_config.json:
    1. In repo root  (jobs_test_ffmpeg_amf/local_config.json)
    2. Next to this script
    """
    candidates = [
        os.path.join(REPO_ROOT, "local_config.json"),
        os.path.join(SCRIPTS_DIR, "local_config.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_config(config_path):
    """Load and return local_config.json, stripping _comment/_*_note keys."""
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Local runner for FFMPEG AMF tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--config",        default=None,
                   help="Path to local_config.json (auto-detected if omitted)")
    # Overrides - all optional; fall back to config values
    p.add_argument("--build_path",    default=None,
                   help="Path to ffmpeg build dir (overrides local_config.json)")
    p.add_argument("--video_samples", default=None,
                   help="Folder containing input video files (overrides local_config.json)")
    p.add_argument("--pack",          default=None,
                   help="Single test pack filename, e.g. FFMPEG_AMF_hwaccel.json "
                        "(overrides test_packs list in config)")
    p.add_argument("--output",        default=None,
                   help="Full output directory path (overrides output_base + timestamp)")
    p.add_argument("--gpu_name",      default=None,
                   help="GPU name string for report")
    p.add_argument("--test_cases",    default="",
                   help="Comma-separated case names to run (empty = all)")
    p.add_argument("--jobs_launcher", default=None,
                   help="Path to jobs_launcher repo (overrides local_config.json)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_dir(path, label):
    """Return True if path exists as a directory, else print error and return False."""
    if not os.path.isdir(path):
        _fail(f"{label} not found: {path}")
        return False
    return True


def _require_file(path, label):
    if not os.path.isfile(path):
        _fail(f"{label} not found: {path}")
        return False
    return True


# ---------------------------------------------------------------------------
# build_reports.bat wrapper (optional)
# ---------------------------------------------------------------------------

def _run_build_reports(jobs_launcher_path, results_dir, pack_name):
    """
    Call jobs_launcher/build_reports.bat to generate summary_report.html
    and compare_report.html, exactly as the Jenkins pipeline does.

    ┌ JENKINS: jobs_launcher checked out as Git submodule, always present
    └ LOCAL:   optional; set jobs_launcher_path in local_config.json
    """
    bat = os.path.join(jobs_launcher_path, "build_reports.bat")
    if not os.path.isfile(bat):
        _warn(f"build_reports.bat not found in: {jobs_launcher_path} - skipping")
        return False

    cmd = [
        bat,
        results_dir,
        "FFMPEG_AMF",
        "local",          # commit SHA - not meaningful locally
        "local",          # branch
        "Local run",      # commit message
        pack_name,
    ]
    _info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=jobs_launcher_path)
    if result.returncode != 0:
        _warn("build_reports.bat returned non-zero - summary/compare HTML may be incomplete")
        return False
    return True


# ---------------------------------------------------------------------------
# Main local runner
# ---------------------------------------------------------------------------

def main():
    cli = parse_args()

    _print_header("FFMPEG AMF  -  Local Test Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Stage 1: Load config ─────────────────────────────────────────────
    _print_stage(1, "Load configuration")

    config_path = cli.config or _find_default_config()
    config = {}
    if config_path:
        try:
            config = load_config(config_path)
            _ok(f"Config loaded: {config_path}")
        except Exception as e:
            _fail(f"Failed to parse {config_path}: {e}")
            sys.exit(1)
    else:
        _warn("No local_config.json found - relying entirely on CLI arguments")

    # Merge: CLI overrides config
    build_path       = cli.build_path    or config.get("build_path", "")
    video_samples    = cli.video_samples or config.get("video_samples_path", "")
    gpu_name         = cli.gpu_name      or config.get("gpu_name", "Unknown GPU")
    output_base      = cli.output        or config.get("output_base", "")
    jobs_launcher_p  = cli.jobs_launcher or config.get("jobs_launcher_path", "")

    # Build test-pack list from config; --pack overrides the list
    if cli.pack:
        test_packs = [{"pack": cli.pack}]
    else:
        test_packs = config.get("test_packs", [])

    if not test_packs:
        _fail("No test packs defined. Set test_packs in local_config.json or pass --pack")
        sys.exit(1)

    # ── Stage 2: Validate paths ──────────────────────────────────────────
    _print_stage(2, "Validate paths")

    # ┌ DIFFERENCE: Jenkins copies build from NAS via xcopy; locally the build
    # │ directory is used directly (no copy step needed).
    # └──────────────────────────────────────────────────────────────────────
    ok = True
    if not build_path:
        _fail("build_path not set. Edit local_config.json or pass --build_path")
        ok = False
    else:
        ok &= _require_dir(build_path, "build_path")
        ok &= _require_file(os.path.join(build_path, "ffmpeg.exe"),  "ffmpeg.exe")
        ok &= _require_file(os.path.join(build_path, "ffprobe.exe"), "ffprobe.exe")

    if not video_samples:
        _fail("video_samples_path not set. Edit local_config.json or pass --video_samples")
        ok = False
    else:
        ok &= _require_dir(video_samples, "video_samples_path")

    if not output_base:
        _fail("output_base not set. Edit local_config.json or pass --output")
        ok = False

    if not ok:
        sys.exit(1)

    _ok(f"Build:         {build_path}")
    _ok(f"Video samples: {video_samples}")

    # ── Stage 3: Run each test pack ──────────────────────────────────────
    overall_exit = 0

    for tp in test_packs:
        pack_filename = tp.get("pack", "")

        if not pack_filename:
            _warn("Skipping test pack entry with no 'pack' filename")
            continue

        pack_path = os.path.join(TESTS_DIR, pack_filename)
        if not os.path.isfile(pack_path):
            _fail(f"Test pack not found: {pack_path}")
            overall_exit = 1
            continue

        # ┌ DIFFERENCE: Jenkins uses WORKSPACE\summaryTestResults (flat, single run).
        # │ Locally each run gets its own timestamped dir so runs don't overwrite.
        # └──────────────────────────────────────────────────────────────────────
        pack_name   = os.path.splitext(pack_filename)[0]
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = cli.output or os.path.join(output_base, f"{timestamp}_{pack_name}")
        os.makedirs(results_dir, exist_ok=True)

        _print_stage(3, f"Run tests  [{pack_name}]")
        _info(f"Test pack:      {pack_path}")
        _info(f"Video samples:  {video_samples}")
        _info(f"Output dir:     {results_dir}")

        # ┌ DIFFERENCE: Jenkins passes --video_samples NAS\VideoSamples.
        # │ Locally we pass the local VideoSamples directory directly.
        # │ Per-case input resolution (case["input_video"]) is done inside run_tests.
        # └──────────────────────────────────────────────────────────────────────
        run_args = Namespace(
            build_path    = build_path,
            video_samples = video_samples,
            test_pack     = pack_path,
            output        = results_dir,
            test_cases    = cli.test_cases,
            gpu_name      = gpu_name,
        )

        try:
            exit_code = run_tests.run(run_args)
        except Exception as e:
            _fail(f"run_tests.run() raised an exception: {e}")
            _fail(traceback.format_exc())
            exit_code = 1

        if exit_code == 0:
            _ok("All test cases passed")
        else:
            _fail("One or more test cases failed - check run_tests.log")
        overall_exit = max(overall_exit, exit_code)

        # ── Stage 4: Generate frame comparison HTML ──────────────────────
        _print_stage(4, "Generate frames_report.html")
        frames_html_path = os.path.join(results_dir, "frames_report.html")
        try:
            generate_frames_html.generate(results_dir, frames_html_path)
            _ok(f"frames_report.html → {frames_html_path}")
        except Exception as e:
            _warn(f"HTML generation failed: {e}")

        # ── Stage 5: build_reports.bat (optional) ────────────────────────
        # ┌ DIFFERENCE: Jenkins always runs this (jobs_launcher submodule is
        # │ always checked out). Locally it is optional - only if
        # │ jobs_launcher_path is set and the repo exists.
        # └──────────────────────────────────────────────────────────────────
        _print_stage(5, "Generate summary/compare reports (jobs_launcher)")
        if jobs_launcher_p and os.path.isdir(jobs_launcher_p):
            success = _run_build_reports(jobs_launcher_p, results_dir, pack_name)
            if success:
                _ok("summary_report.html + compare_report.html generated")
        else:
            _warn("jobs_launcher_path not set or not found - skipping summary/compare HTML")
            _info("Set 'jobs_launcher_path' in local_config.json to enable this step")

    # ── Stage 6: Final summary ───────────────────────────────────────────
    _print_stage(6, "Summary")
    if overall_exit == 0:
        print(f"\n  {GREEN}{BOLD}ALL PACKS PASSED{RESET}")
    else:
        print(f"\n  {RED}{BOLD}FAILURES DETECTED{RESET}")
    print(f"  Results in: {output_base}\n")

    sys.exit(overall_exit)


if __name__ == "__main__":
    main()
