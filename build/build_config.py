"""
Configuration for the in-framework ffmpeg + AMF build stage.

Every value can be overridden by an environment variable of the same name.
Defaults assume the layout produced by the existing amf-ffmpeg build tree
described in Ubuntu_build_instructions.txt.
"""

import os

# ".exe" on Windows, empty elsewhere.
EXE = ".exe" if os.name == "nt" else ""

# --- Locations -------------------------------------------------------------

# This file lives in   <repo>/build/build_config.py
_HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(_HERE)

# amf-ffmpeg working tree (FFmpeg + AMF sources and the documented dep layout).
# Default: sibling of the repo root.
AMF_FFMPEG_DIR = os.environ.get(
    "AMF_FFMPEG_DIR",
    os.path.normpath(os.path.join(REPO_ROOT, "..", "amf-ffmpeg")),
)

# Vendored build scripts (this folder). Preferred over amf-ffmpeg's copies so
# the new version is self-contained — the Linux .sh has the corrected args.
BUILD_SCRIPTS_DIR = _HERE
WIN_BUILD_SCRIPT  = os.path.join(BUILD_SCRIPTS_DIR, "ffmpeg_build.bat")
NIX_BUILD_SCRIPT  = os.path.join(BUILD_SCRIPTS_DIR, "ffmpeg_build.sh")

# Source sub-dir names inside AMF_FFMPEG_DIR.
FFMPEG_SRC_DIRNAME = os.environ.get("FFMPEG_SRC_DIRNAME", "ffmpeg")
AMF_SRC_DIRNAME    = os.environ.get("AMF_SRC_DIRNAME", "AMF")

# Auto-clone URLs (used only when the sources are missing).
FFMPEG_GIT_URL = os.environ.get("FFMPEG_GIT_URL", "https://code.ffmpeg.org/FFmpeg/FFmpeg.git")
AMF_GIT_URL    = os.environ.get("AMF_GIT_URL",    "https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git")

# --- Dependency preflight --------------------------------------------------

MSYS2_ROOT     = os.environ.get("MSYS2_ROOT", r"C:\msys64")
WIN_DEPS_ROOT  = os.environ.get("WIN_DEPS_ROOT", r"C:\deps")
WIN_DEPS_PKGCONFIG = [
    os.path.join(WIN_DEPS_ROOT, d, "lib", "pkgconfig")
    for d in ("dav1d", "x264", "x265", "svtav1")
]
# Linux pkg-config specs that must resolve before configure.
LINUX_DEP_SPECS = ["dav1d >= 1.0.0", "x264", "x265", "SvtAv1Enc"]

# --- Self-contained packaging ----------------------------------------------

# Windows: dirs searched for runtime DLLs to bundle into the zip.
WIN_DLL_SEARCH_DIRS = [
    os.path.join(WIN_DEPS_ROOT, "dav1d",  "bin"),
    os.path.join(WIN_DEPS_ROOT, "x264",   "bin"),
    os.path.join(WIN_DEPS_ROOT, "x265",   "bin"),
    os.path.join(WIN_DEPS_ROOT, "svtav1", "bin"),
    os.path.join(MSYS2_ROOT, "mingw64", "bin"),
]
WIN_DLL_PATTERNS = [
    "dav1d*.dll", "libdav1d*.dll",
    "x264*.dll", "libx264*.dll",
    "x265*.dll", "libx265*.dll",
    "SvtAv1Enc*.dll", "libSvtAv1Enc*.dll",
    "libwinpthread*.dll", "libgcc*.dll", "libstdc++*.dll",
]

# Build is rejected unless all of these encoders are present.
REQUIRED_AMF_ENCODERS = ["h264_amf", "hevc_amf", "av1_amf"]

# Downloadable zip lands in <repo>/<ARTIFACTS_DIRNAME>/.
ARTIFACTS_DIRNAME = os.environ.get("ARTIFACTS_DIRNAME", "artifacts")
