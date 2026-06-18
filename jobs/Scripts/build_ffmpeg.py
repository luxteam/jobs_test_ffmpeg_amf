#!/usr/bin/env python3
"""
In-framework ffmpeg + AMF build stage.

Drives the vendored build scripts (build/ffmpeg_build.bat on Windows,
build/ffmpeg_build.sh on Linux) — the same flow documented in
Ubuntu_build_instructions.txt — then verifies the AMF encoders and packages a
self-contained, downloadable .zip artifact.

Two ways to use it:
  * standalone stage:  python build_ffmpeg.py [--build_type release] [...]
  * from run_tests.py: build(...) is called when --auto_build is passed

build(...) returns the ffmpeg build directory, suitable for
run_tests.py --build_path.
"""

import argparse
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime

# Make <repo>/build importable regardless of the current working directory.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))         # <repo>/jobs/Scripts
_REPO_ROOT   = os.path.dirname(os.path.dirname(_SCRIPTS_DIR))     # <repo>
_BUILD_DIR   = os.path.join(_REPO_ROOT, "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

import build_config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_logger(logger):
    if logger is not None:
        return logger
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("build_ffmpeg")


def _run(cmd, logger, cwd=None, env=None, shell=False):
    """Run a command, streaming its output into the logger. Returns exit code."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    logger.info(f"$ {printable}")
    proc = subprocess.run(cmd, cwd=cwd, env=env, shell=shell,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            logger.info(f"  | {line}")
    return proc.returncode


def _capture(cmd, cwd=None):
    """Run a command and return (returncode, combined_output)."""
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


# ---------------------------------------------------------------------------
# Sources & dependencies
# ---------------------------------------------------------------------------

def _ensure_sources(amf_dir, logger):
    """Clone FFmpeg / AMF sources into amf_dir if they are missing."""
    ffmpeg_src = os.path.join(amf_dir, cfg.FFMPEG_SRC_DIRNAME)
    amf_src    = os.path.join(amf_dir, cfg.AMF_SRC_DIRNAME)
    os.makedirs(amf_dir, exist_ok=True)

    has_ffmpeg = os.path.exists(os.path.join(ffmpeg_src, "configure"))
    if not has_ffmpeg:
        logger.info(f"FFmpeg sources not found — cloning {cfg.FFMPEG_GIT_URL}")
        if _run(["git", "clone", cfg.FFMPEG_GIT_URL, ffmpeg_src], logger) != 0:
            raise RuntimeError("git clone FFmpeg failed")

    has_amf = os.path.isdir(os.path.join(amf_src, "amf"))
    if not has_amf:
        logger.info(f"AMF sources not found — cloning {cfg.AMF_GIT_URL}")
        if _run(["git", "clone", cfg.AMF_GIT_URL, amf_src], logger) != 0:
            raise RuntimeError("git clone AMF failed")

    return ffmpeg_src, amf_src


def _preflight_deps(logger):
    """Fail fast (or warn) if build dependencies are missing."""
    if platform.system() == "Windows":
        if not os.path.isdir(cfg.MSYS2_ROOT):
            raise RuntimeError(
                f"msys2 not found at {cfg.MSYS2_ROOT}. "
                f"Install msys2 or set MSYS2_ROOT (see Ubuntu_build_instructions.txt / Windows build)."
            )
        dav1d_pc = os.path.join(cfg.WIN_DEPS_ROOT, "dav1d", "lib", "pkgconfig", "dav1d.pc")
        if not os.path.exists(dav1d_pc):
            raise RuntimeError(
                f"dav1d not found at {dav1d_pc}. Build/install dav1d to "
                f"C:\\deps\\dav1d (MSVC, static) — see BUILD_AND_TEST.md."
            )
    else:
        if shutil.which("pkg-config") is None:
            raise RuntimeError("pkg-config not found; sudo apt install pkg-config")
        for spec in cfg.LINUX_DEP_SPECS:
            rc = subprocess.run(["pkg-config", "--exists", spec],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            if rc != 0:
                raise RuntimeError(
                    f"dependency '{spec}' not found via pkg-config. "
                    f"See Ubuntu_build_instructions.txt (dav1d/x264/x265/SVT-AV1 setup)."
                )
        logger.info("Linux dependencies OK: %s", ", ".join(cfg.LINUX_DEP_SPECS))


# ---------------------------------------------------------------------------
# Verification & versioning
# ---------------------------------------------------------------------------

def _verify_amf_encoders(ffmpeg_exe, logger):
    rc, out = _capture([ffmpeg_exe, "-hide_banner", "-encoders"])
    if rc != 0:
        raise RuntimeError(f"ffmpeg -encoders failed (exit {rc})")
    missing = [e for e in cfg.REQUIRED_AMF_ENCODERS if e not in out]
    if missing:
        raise RuntimeError(f"AMF encoders missing from build: {missing}")
    logger.info("Verified AMF encoders present: %s", ", ".join(cfg.REQUIRED_AMF_ENCODERS))


def _ffmpeg_version(ffmpeg_src, ffmpeg_exe):
    """Prefer 'git describe --tags' in the source tree; fall back to -version."""
    rc, out = _capture(["git", "describe", "--tags"], cwd=ffmpeg_src)
    if rc == 0 and out.strip():
        return out.strip()
    rc, out = _capture([ffmpeg_exe, "-version"])
    m = re.search(r"ffmpeg version (\S+)", out)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Self-contained packaging
# ---------------------------------------------------------------------------

def _bundle_windows_dlls(dest_dir, logger):
    copied = []
    for d in cfg.WIN_DLL_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in cfg.WIN_DLL_PATTERNS:
            for f in glob.glob(os.path.join(d, pat)):
                target = os.path.join(dest_dir, os.path.basename(f))
                if not os.path.exists(target):
                    shutil.copy2(f, target)
                    copied.append(target)
    if copied:
        logger.info("Bundled %d Windows DLLs", len(copied))
    else:
        logger.warning("No dependency DLLs bundled — expected if libs are static; "
                       "otherwise add the DLL dir to WIN_DLL_SEARCH_DIRS in build_config.py")
    return copied


def _bundle_linux_libs(exes, lib_dir, logger):
    """Copy non-system shared libs the binaries depend on into lib_dir."""
    needed = set()
    for exe in exes:
        rc, out = _capture(["ldd", exe])
        if rc != 0:
            continue
        for line in out.splitlines():
            if "=>" in line:
                path = line.split("=>", 1)[1].strip().split(" ")[0]
                if path and os.path.exists(path):
                    needed.add(path)
    interesting = ("dav1d", "x264", "x265", "svtav1", "svt-av1", "amf")
    copied = []
    for path in sorted(needed):
        base = os.path.basename(path)
        if path.startswith("/usr/local/") or any(n in base.lower() for n in interesting):
            os.makedirs(lib_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(lib_dir, base))
            copied.append(path)
    if copied:
        logger.info("Bundled %d Linux .so libs into lib/ (run with "
                    "LD_LIBRARY_PATH=$PWD/lib)", len(copied))
    else:
        logger.warning("No non-system .so deps bundled — likely statically linked")
    return copied


def _package_self_contained(ffmpeg_exe, ffprobe_exe, ffmpeg_src,
                            build_type, artifacts_dir, logger):
    os.makedirs(artifacts_dir, exist_ok=True)
    version  = _ffmpeg_version(ffmpeg_src, ffmpeg_exe)
    plat     = "windows" if os.name == "nt" else platform.system().lower()
    safe_ver = re.sub(r"[^\w.\-]", "-", version)
    stem     = f"ffmpeg_amf_{safe_ver}_{plat}_{build_type}"

    stage = os.path.join(artifacts_dir, stem)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)

    shutil.copy2(ffmpeg_exe,  stage)
    shutil.copy2(ffprobe_exe, stage)

    if os.name == "nt":
        bundled = _bundle_windows_dlls(stage, logger)
    else:
        bundled = _bundle_linux_libs([ffmpeg_exe, ffprobe_exe],
                                     os.path.join(stage, "lib"), logger)

    info_lines = [
        "ffmpeg_amf build artifact",
        f"version:    {version}",
        f"platform:   {plat}",
        f"build_type: {build_type}",
        f"date:       {datetime.now().isoformat(timespec='seconds')}",
        f"encoders:   {', '.join(cfg.REQUIRED_AMF_ENCODERS)}",
        f"bundled libs ({len(bundled)}):",
    ] + [f"  - {os.path.basename(b)}" for b in bundled]
    with open(os.path.join(stage, "build_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")

    zip_path = os.path.join(artifacts_dir, stem + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # Flat layout: ffmpeg/ffprobe (+ bundled libs, build_info.txt) at the zip
    # root, matching the existing "build_ffmpeg.zip -> ffmpeg_build" unzip
    # convention used by the pipeline and the NAS builds.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, stage))

    logger.info("Artifact packaged: %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build(amf_ffmpeg_dir=None, build_type="release",
          artifacts_dir=None, package=True, logger=None):
    """
    Build ffmpeg+AMF via the vendored build scripts and (optionally) package a
    self-contained zip artifact.

    Returns the ffmpeg build directory (contains ffmpeg/ffprobe), suitable for
    run_tests.py --build_path.
    """
    logger = _get_logger(logger)
    system = platform.system()
    amf_dir = os.path.abspath(amf_ffmpeg_dir or cfg.AMF_FFMPEG_DIR)
    logger.info("amf-ffmpeg dir: %s", amf_dir)

    ffmpeg_src, amf_src = _ensure_sources(amf_dir, logger)
    _preflight_deps(logger)

    build_dir   = os.environ.get("FFMPEG_BUILD_DIR")   or os.path.join(amf_dir, "build_ffmpeg")
    install_dir = os.environ.get("FFMPEG_INSTALL_DIR") or os.path.join(amf_dir, "ffmpeg_install")
    amf_headers = os.path.join(amf_src, "amf", "public", "include")

    # The build scripts concatenate suffixes directly onto these dir vars
    # ("%FFMPEG_SRC_DIR%configure", "%FFMPEG_BUILD_DIR%extrainclude\"), so each
    # MUST end with a path separator or the names get mashed together
    # (".../ffmpegconfigure", ".../build_ffmpegextrainclude").
    def _with_sep(p):
        return p if p.endswith(os.sep) else p + os.sep

    env = os.environ.copy()
    env.update({
        "FFMPEG_SRC_DIR":     _with_sep(ffmpeg_src),
        "FFMPEG_BUILD_DIR":   _with_sep(build_dir),
        "FFMPEG_INSTALL_DIR": _with_sep(install_dir),
        "AMF_HEADERS_DIR":    _with_sep(amf_headers),
        "FFMPEG_BUILD_TYPE":  build_type,
    })

    if system == "Windows":
        script = cfg.WIN_BUILD_SCRIPT
        configure_cmd = f'"{script}" configure'
        build_cmd     = f'"{script}" build'
        shell = True
    else:
        script = cfg.NIX_BUILD_SCRIPT
        try:
            os.chmod(script, 0o755)
        except OSError:
            pass
        configure_cmd = ["sh", script, "configure"]
        build_cmd     = ["sh", script, "build"]
        shell = False

    logger.info("=== configure ===")
    if _run(configure_cmd, logger, cwd=amf_dir, env=env, shell=shell) != 0:
        raise RuntimeError("ffmpeg configure failed")

    logger.info("=== build ===")
    if _run(build_cmd, logger, cwd=amf_dir, env=env, shell=shell) != 0:
        raise RuntimeError("ffmpeg build failed")

    ffmpeg_exe  = os.path.join(build_dir, "ffmpeg"  + cfg.EXE)
    ffprobe_exe = os.path.join(build_dir, "ffprobe" + cfg.EXE)
    for p in (ffmpeg_exe, ffprobe_exe):
        if not os.path.exists(p):
            raise RuntimeError(f"expected binary not produced: {p}")

    _verify_amf_encoders(ffmpeg_exe, logger)

    if package:
        out_dir = os.path.abspath(artifacts_dir or os.path.join(_REPO_ROOT, cfg.ARTIFACTS_DIRNAME))
        _package_self_contained(ffmpeg_exe, ffprobe_exe, ffmpeg_src,
                                build_type, out_dir, logger)

    logger.info("BUILD_PATH=%s", build_dir)
    return build_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="In-framework ffmpeg+AMF build stage")
    p.add_argument("--amf_ffmpeg_dir", default="",
                   help="amf-ffmpeg sources/scripts dir (default: ../amf-ffmpeg)")
    p.add_argument("--build_type", default="release", choices=["debug", "release"])
    p.add_argument("--artifacts_dir", default="",
                   help="where to write the downloadable zip (default: <repo>/artifacts)")
    p.add_argument("--no_package", action="store_true",
                   help="skip building the downloadable zip artifact")
    args = p.parse_args()

    build_dir = build(
        amf_ffmpeg_dir=(args.amf_ffmpeg_dir or None),
        build_type=args.build_type,
        artifacts_dir=(args.artifacts_dir or None),
        package=not args.no_package,
    )
    print(f"BUILD_PATH={build_dir}")


if __name__ == "__main__":
    main()
