"""
Generates a self-contained HTML frame comparison report for Jenkins HTML Publisher plugin.

For each test case it shows:
  - FFMPEG command, average PSNR/SSIM, render time, metadata
  - Error/warning messages
  - The 5 WORST frames (lowest PSNR) as image quads:
      Input | Output | Diff (scaled) | Diff (threshold)
    with per-frame MSE / PSNR / SSIM values

Image source: reads worst_frames list from *_RPR.json (written by run_tests.py).
Falls back to any *.jpg files found in Color/<case>/ if worst_frames is absent.

Usage:
    python generate_frames_html.py
        --results_dir  <path containing *_RPR.json + Color/ subdir>
        --output_html  <path to write HTML file>
"""

import argparse
import base64
import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# HTML skeleton
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FFMPEG AMF – Frame Comparison Report</title>
<style>
  :root {{
    --bg:      #1a1a1a;
    --bg2:     #252525;
    --bg3:     #1e1e1e;
    --border:  #444;
    --text:    #e0e0e0;
    --muted:   #888;
    --accent:  #ff6600;
    --pass:    #4caf50;
    --fail:    #f44336;
    --warn:    #ff9800;
    --skip:    #9e9e9e;
    --blue:    #82b1ff;
    --green:   #a5d6a7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: var(--bg); color: var(--text);
          padding: 16px; }}
  h1   {{ color: var(--accent); margin-bottom: 4px; }}
  .ts  {{ color: var(--muted); font-size: .8em; margin-bottom: 20px; }}

  /* ---- case block ---- */
  .case  {{ background: var(--bg2); border-radius: 6px; padding: 14px 16px;
            margin-bottom: 22px; border-left: 4px solid var(--border); }}
  .case.passed  {{ border-color: var(--pass); }}
  .case.failed  {{ border-color: var(--fail); }}
  .case.error   {{ border-color: var(--warn); }}
  .case.skipped {{ border-color: var(--skip); }}

  .case-header {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }}
  .case-title  {{ font-size: 1.1em; font-weight: bold; }}
  .badge {{ font-size: .75em; font-weight: bold; padding: 2px 8px;
            border-radius: 3px; letter-spacing: .05em; }}
  .badge.passed  {{ background: var(--pass);  color: #000; }}
  .badge.failed  {{ background: var(--fail);  color: #fff; }}
  .badge.error   {{ background: var(--warn);  color: #000; }}
  .badge.skipped {{ background: var(--skip);  color: #fff; }}

  .cmd {{ font-family: monospace; font-size: .78em; background: var(--bg3);
          padding: 6px 10px; border-radius: 4px; word-break: break-all;
          color: #aaa; margin: 6px 0 10px 0; }}

  /* ---- metrics row ---- */
  .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
  .chip    {{ background: var(--bg3); border: 1px solid var(--border);
              border-radius: 4px; padding: 3px 10px; font-size: .85em; }}
  .chip.bad {{ border-color: var(--fail); color: var(--fail); }}

  /* ---- messages ---- */
  .messages {{ margin: 8px 0; font-size: .85em; }}
  .msg-err  {{ color: var(--fail); margin: 2px 0; }}
  .msg-warn {{ color: var(--warn); margin: 2px 0; }}

  /* ---- metadata table ---- */
  details {{ margin: 8px 0; }}
  summary {{ cursor: pointer; font-size: .85em; color: var(--muted); }}
  .meta-table {{ border-collapse: collapse; font-size: .82em; margin-top: 6px; }}
  .meta-table td, .meta-table th {{
    border: 1px solid var(--border); padding: 3px 10px;
  }}
  .meta-table th {{ background: var(--bg3); }}

  /* ---- frame grid ---- */
  h3.frames-title {{ font-size: .9em; color: var(--muted); margin: 14px 0 8px 0;
                     border-top: 1px solid var(--border); padding-top: 10px; }}
  .frame-row {{ margin-bottom: 18px; }}
  .frame-row-header {{ font-size: .82em; color: var(--muted); margin-bottom: 5px; }}
  .frame-row-header .fi {{ color: var(--text); font-weight: bold; }}
  .frame-row-header .fm {{ color: var(--warn); }}
  .frame-row-header .fp {{ color: var(--green); }}

  .quad {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
  .quad-item {{ display: flex; flex-direction: column; gap: 3px; }}
  .quad-item img {{ width: 100%; border-radius: 3px;
                    border: 1px solid var(--border); }}
  .quad-label {{ font-size: .72em; text-align: center; color: var(--muted); }}
  .quad-label.input  {{ color: var(--blue); }}
  .quad-label.output {{ color: var(--green); }}
  .quad-label.diff   {{ color: var(--warn); }}
  .quad-label.thresh {{ color: #ce93d8; }}

  .no-frames {{ color: var(--muted); font-size: .85em; font-style: italic;
                margin-top: 10px; }}
</style>
</head>
<body>
<h1>FFMPEG AMF – Frame Comparison Report</h1>
<p class="ts">Generated: {generated} &nbsp;|&nbsp; Showing worst-PSNR frames per test case</p>
{cases_html}
</body>
</html>
"""

CASE_TEMPLATE = """\
<div class="case {status}">
  <div class="case-header">
    <span class="case-title">{case_name}</span>
    <span class="badge {status}">{status_upper}</span>
  </div>
  <div class="cmd">{cmd}</div>
  <div class="metrics">{chips}</div>
  {messages_html}
  {metadata_html}
  {frames_html}
</div>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(path):
    """Return base64 data-URI for an image, or empty string if missing."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def _chip(label, value, bad=False):
    cls = ' class="chip bad"' if bad else ' class="chip"'
    return f'<span{cls}>{label}: <b>{value}</b></span>'


def _render_messages(messages):
    if not messages:
        return ""
    items = []
    for m in messages:
        if isinstance(m, dict):
            text = m.get("issue", str(m))
            desc = m.get("description", "")
            is_warn = text.startswith("[Warning]")
            css = "msg-warn" if is_warn else "msg-err"
            suffix = f' <span style="color:#555">- {desc}</span>' if desc else ""
            items.append(f'<div class="{css}">&#9654; {text}{suffix}</div>')
        else:
            items.append(f'<div class="msg-err">&#9654; {m}</div>')
    return f'<div class="messages">{"".join(items)}</div>'


def _render_metadata(metadata):
    if not metadata:
        return ""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(metadata.items())
    )
    table = (f'<table class="meta-table">'
             f'<tr><th>Field</th><th>Value</th></tr>{rows}</table>')
    return f'<details><summary>ffprobe metadata</summary>{table}</details>'


def _img_tag(path, alt):
    src = _b64(path)
    if src:
        return f'<img src="{src}" alt="{alt}">'
    return '<div style="height:90px;background:#333;border-radius:3px;"></div>'


def _render_worst_frames(worst_frames):
    """Render 5 worst-frame quads (input / output / diff_scaled / diff_thresh)."""
    if not worst_frames:
        return '<p class="no-frames">No frame comparison data available.</p>'

    rows = []
    for entry in worst_frames:
        fn    = entry.get("frame_number", "?")
        mse   = entry.get("mse",  "N/A")
        psnr  = entry.get("psnr", "N/A")
        imgs  = entry.get("images", {})

        header = (
            f'<div class="frame-row-header">'
            f'Frame <span class="fi">#{fn}</span> &nbsp;|&nbsp; '
            f'<span class="fm">MSE={mse:.3f}</span> &nbsp; '
            f'<span class="fp">PSNR={psnr:.2f} dB</span>'
            f'</div>'
        ) if isinstance(psnr, float) else (
            f'<div class="frame-row-header">Frame <span class="fi">#{fn}</span></div>'
        )

        quad = (
            f'<div class="quad">'
            f'  <div class="quad-item">{_img_tag(imgs.get("input"),       "input")}'
            f'    <div class="quad-label input">Input</div></div>'
            f'  <div class="quad-item">{_img_tag(imgs.get("output"),      "output")}'
            f'    <div class="quad-label output">Output</div></div>'
            f'  <div class="quad-item">{_img_tag(imgs.get("diff_scaled"), "diff scaled")}'
            f'    <div class="quad-label diff">Diff (scaled)</div></div>'
            f'  <div class="quad-item">{_img_tag(imgs.get("diff_thresh"), "diff threshold")}'
            f'    <div class="quad-label thresh">Diff (threshold)</div></div>'
            f'</div>'
        )
        rows.append(f'<div class="frame-row">{header}{quad}</div>')

    title = '<h3 class="frames-title">Worst frames by PSNR (lowest first)</h3>'
    return title + "\n".join(rows)


# ---------------------------------------------------------------------------
# Per-case HTML builder
# ---------------------------------------------------------------------------

def _build_case_html(report):
    case_name   = report.get("test_case", "unknown")
    status      = report.get("test_status", "error")
    psnr_val    = report.get("psnr")
    ssim_val    = report.get("ssim")
    render_time = report.get("render_time", 0.0)
    messages    = report.get("message", [])
    metadata    = report.get("metadata", {})
    cmd         = report.get("ffmpeg_command", "-")
    worst_frames = report.get("worst_frames", [])

    psnr_str = f"{psnr_val:.2f} dB" if isinstance(psnr_val, (int, float)) else "N/A"
    ssim_str = f"{ssim_val:.4f}"    if isinstance(ssim_val, (int, float)) else "N/A"

    chips = []
    chips.append(_chip("PSNR", psnr_str))
    chips.append(_chip("SSIM", ssim_str))
    chips.append(_chip("Time", f"{render_time:.1f}s"))
    codec = metadata.get("codec_name", "")
    res   = (f"{metadata.get('width')}x{metadata.get('height')}"
             if metadata.get("width") else "")
    if codec: chips.append(_chip("Codec", codec))
    if res:   chips.append(_chip("Resolution", res))

    return CASE_TEMPLATE.format(
        case_name    = case_name,
        status       = status,
        status_upper = status.upper(),
        cmd          = cmd,
        chips        = "".join(chips),
        messages_html = _render_messages(messages),
        metadata_html = _render_metadata(metadata),
        frames_html   = _render_worst_frames(worst_frames),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _natural_sort_key(fname):
    """Sort key that orders hwaccel_2 before hwaccel_10 (numeric, not lexicographic)."""
    import re as _re
    parts = _re.split(r"(\d+)", fname)
    return [int(p) if p.isdigit() else p for p in parts]


def build_html(results_dir):
    report_files = sorted(
        (f for f in os.listdir(results_dir) if f.endswith("_RPR.json")),
        key=_natural_sort_key,
    )
    if not report_files:
        return "<p>No test case reports found.</p>"

    parts = []
    for fname in report_files:
        try:
            with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            report = data[0] if isinstance(data, list) else data
        except Exception as e:
            parts.append(f"<p style='color:red'>Failed to load {fname}: {e}</p>")
            continue
        parts.append(_build_case_html(report))

    return "\n".join(parts)


def generate(results_dir, output_html):
    """
    Callable entry point for run_local.py and any other Python caller.
    Generates the HTML file and returns the output path.
    """
    cases_html = build_html(results_dir)
    html = PAGE_TEMPLATE.format(
        generated  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cases_html = cases_html,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Frames report written: {output_html}")
    return output_html


def main():
    parser = argparse.ArgumentParser(description="Generate frame comparison HTML report")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing *_RPR.json files")
    parser.add_argument("--output_html", required=True,
                        help="Path to write the output HTML file")
    args = parser.parse_args()
    generate(args.results_dir, args.output_html)


if __name__ == "__main__":
    main()
