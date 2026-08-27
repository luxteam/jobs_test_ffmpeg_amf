#!/usr/bin/env python3
"""Provision a VMAF-capable ffmpeg (gyan.dev git build) for the VMAF/compare step.

Pattern A: pulls directly from gyan.dev on the node.
Latest-but-pinnable via VMAF_FFMPEG_VERSION (env) or --version:
  - "latest" (default): the always-newest gyan git-master build.
  - "<ver>"  (e.g. 2026-08-20-git-7d77562d2a): a pinned gyan GitHub release asset.

Two output modes:
  - dir mode (default): extract the whole build into --dest. The build framework
    passes --dest as --compare_build_path; run_tests' get_ffmpeg_path() finds
    <dest>/<ver>_build/bin/ffmpeg.exe.
  - single-file mode (--as NAME): copy just bin/ffmpeg.exe to <dest>/NAME. The
    AMFInternal framework uses this to produce ffmpeg_vmaf.exe.

Resilience: a transient upstream failure (HTTP 503 / offline) does NOT discard an
existing cached build - it is reused. A download is only attempted when there is
no cached build or the upstream sha256 has changed.

Windows only for now; on other OSes it exits 0 without acting (Linux VMAF ffmpeg
is provisioned separately). Extraction uses the system 7-Zip.

On genuine failure (no build produced and none cached) it prints an error and
exits non-zero; callers continue WITHOUT a VMAF ffmpeg so VMAF cases skip.
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

GYAN_BASE  = "https://www.gyan.dev/ffmpeg/builds"
GITHUB_REL = "https://github.com/GyanD/codexffmpeg/releases/download"
_7Z_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def _log(msg):
    print("[provision_vmaf] " + msg, flush=True)


def _find_7z(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    for c in _7Z_CANDIDATES:
        if os.path.isfile(c):
            return c
    found = shutil.which("7z")
    if found:
        return found
    raise RuntimeError("7z.exe not found (looked in Program Files and PATH)")


def _fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "vmaf-provisioner"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace").strip()


def _download(url, dest):
    _log("GET " + url)
    req = urllib.request.Request(url, headers={"User-Agent": "vmaf-provisioner"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(variant, version):
    """Return (archive_url, sha256_url_or_None, marker) for latest|pinned."""
    if version in ("", "latest", None):
        url = "%s/ffmpeg-git-%s.7z" % (GYAN_BASE, variant)
        return url, url + ".sha256", "latest"
    fname = "ffmpeg-%s-%s_build.7z" % (version, variant)          # pinned release asset
    return "%s/%s/%s" % (GITHUB_REL, version, fname), None, version


def _has_libvmaf(ffmpeg_exe):
    try:
        out = subprocess.run([ffmpeg_exe, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30)
        return "libvmaf" in (out.stdout + out.stderr)
    except Exception:
        return False


def _ffmpeg_version(ffmpeg_exe):
    try:
        out = subprocess.run([ffmpeg_exe, "-version"],
                             capture_output=True, text=True, timeout=10)
        m = re.search(r"ffmpeg version (\S+)", out.stdout + out.stderr)
        return m.group(1) if m else ""
    except Exception:
        return ""


def _up_to_date(marker_path, remote_key, present):
    if not (remote_key and present and os.path.isfile(marker_path)):
        return False
    try:
        return open(marker_path).read().strip() == remote_key
    except OSError:
        return False


def provision(dest, variant, version, as_name=None, sevenzip=None):
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    marker_path = os.path.join(dest, (as_name + ".marker") if as_name
                               else ".vmaf_ffmpeg.marker")
    url, sha_url, marker = _resolve(variant, version)

    if as_name:
        present = os.path.isfile(os.path.join(dest, as_name))
    else:
        present = bool(glob.glob(os.path.join(dest, "**", "ffmpeg.exe"),
                                 recursive=True))

    # --- cache-gate: only re-download when the build actually changed ---
    remote_key = marker
    if sha_url:
        try:
            remote_key = _fetch_text(sha_url).split()[0]
        except Exception as e:
            # transient upstream issue (HTTP 503 / offline): keep a cached build.
            if present:
                _log("upstream unreachable (%s); using cached build" % e)
                return 0
            _log("WARN: upstream unreachable (%s) and no cached build" % e)
            remote_key = None
    if _up_to_date(marker_path, remote_key, present):
        _log("up-to-date (%s); skipping download" % str(remote_key)[:12])
        return 0

    sevenzip = _find_7z(sevenzip)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "ffmpeg.7z")
            _download(url, archive)
            if sha_url:
                try:
                    want = _fetch_text(sha_url).split()[0].lower()
                    got = _sha256(archive).lower()
                    if want != got:
                        raise RuntimeError("sha256 mismatch: want %s got %s"
                                           % (want[:12], got[:12]))
                    _log("sha256 OK")
                    remote_key = want
                except Exception as e:
                    _log("WARN: sha256 check skipped/failed: %s" % e)

            extract = os.path.join(tmp, "x")
            os.makedirs(extract)
            subprocess.run([sevenzip, "x", "-y", archive, "-o" + extract], check=True)
            ff = next(iter(glob.glob(os.path.join(extract, "**", "bin", "ffmpeg.exe"),
                                     recursive=True)), None)
            if not ff:
                raise RuntimeError("ffmpeg.exe not found after extraction")
            if not _has_libvmaf(ff):
                raise RuntimeError("extracted ffmpeg has no libvmaf filter")
            ver = _ffmpeg_version(ff)

            if as_name:
                shutil.copy2(ff, os.path.join(dest, as_name))
                _log("installed %s (version %s)" % (as_name, ver or "unknown"))
            else:
                for old in glob.glob(os.path.join(dest, "ffmpeg-*")):
                    shutil.rmtree(old, ignore_errors=True)
                top = os.path.dirname(os.path.dirname(ff))   # <ver>_build/
                shutil.move(top, os.path.join(dest, os.path.basename(top)))
                _log("installed build %s (version %s)"
                     % (os.path.basename(top), ver or "unknown"))
    except Exception as e:
        # download/extract failed: keep a cached build if we have one.
        if present:
            _log("refresh failed (%s); using cached build" % e)
            return 0
        _log("ERROR: %s" % e)
        return 2

    with open(marker_path, "w") as f:
        f.write(remote_key or marker)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--variant", default="full", choices=["full", "essentials"])
    ap.add_argument("--version",
                    default=os.environ.get("VMAF_FFMPEG_VERSION", "latest"))
    ap.add_argument("--as", dest="as_name", default=None,
                    help="single-file mode: copy bin/ffmpeg.exe to <dest>/<name>")
    ap.add_argument("--sevenzip", default=os.environ.get("SEVENZIP_EXE"))
    args = ap.parse_args()

    if os.name != "nt":
        _log("non-Windows: skipping gyan provision (Linux handled separately)")
        return 0
    return provision(args.dest, args.variant, args.version,
                     args.as_name, args.sevenzip)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        _log("ERROR: %s" % e)
        sys.exit(1)
