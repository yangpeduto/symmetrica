"""Render a match analysis into one self-contained, interactive HTML file.

The page carries its data as embedded JSON and draws everything with
vanilla JS (canvas for the animation, SVG for charts): no network requests,
no build step, so the file can be emailed, attached to a submission or
opened offline.
"""
import html
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def render_html(summary: dict) -> str:
    payload = json.dumps(summary, separators=(",", ":"), allow_nan=False)
    # Inside <script> only "</" can end the element early; "<!--" can confuse parsers.
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    title = html.escape(f"{summary['meta']['name']} · tactical report")
    return TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", title).replace("__DATA__", payload)


def write_report(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(summary), encoding="utf-8")
    return path
