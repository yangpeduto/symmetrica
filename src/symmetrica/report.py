"""Render a match analysis into one self-contained, interactive HTML file.

The page carries its data as embedded JSON and draws everything with
vanilla JS (canvas for the animation, SVG for charts): no network requests,
no build step, so the file can be emailed, attached to a submission or
opened offline.
"""
import datetime as dt
import html
import json
import shutil
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")
LANDING = Path(__file__).with_name("landing_template.html")
REPO_URL = "https://github.com/yangpeduto/symmetrica"


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


def _match_card(key: str, s: dict) -> str:
    e = html.escape
    sc = s["meta"]["score"]
    f = s.get("shape", {}).get("formations", {})
    form = {t: (f.get(t, {}).get("out") or {}).get("formation", "–") for t in ("home", "away")}
    rows = [
        ("Possession", f"{s['possession']['home_share']:.0%}", f"{1 - s['possession']['home_share']:.0%}"),
        ("PPDA", f"{s['pressing']['home']['ppda'] or 0:.1f}", f"{s['pressing']['away']['ppda'] or 0:.1f}"),
        ("xT created", f"{s['threat']['home']['xt_generated']:.2f}", f"{s['threat']['away']['xt_generated']:.2f}"),
        ("Defensive shape", form["home"], form["away"]),
    ]
    if "offball" in s:
        rows.append(("Off-ball runs", str(s["offball"]["home"]["runs"]), str(s["offball"]["away"]["runs"])))
    body = "".join(f"<tr><td>{e(h)}</td><td>{e(k)}</td><td>{e(a)}</td></tr>" for k, h, a in rows)
    return (f'<a class="match" href="{e(key)}.html"><div class="name">{e(s["meta"]["name"])}</div>'
            f'<div class="score"><i class="sw" style="background:var(--home)"></i>Home'
            f'<span class="n">{sc["home"]}</span><span class="n" style="color:var(--text-muted)">–</span>'
            f'<span class="n">{sc["away"]}</span>Away<i class="sw" style="background:var(--away)"></i></div>'
            f'<table class="stats">{body}</table><div class="cta">Open the interactive report →</div></a>')


def render_landing(summaries: dict[str, dict]) -> str:
    """GitHub Pages index linking to each match report (`<key>.html` next to it)."""
    cards = "".join(_match_card(k, s) for k, s in summaries.items())
    return (LANDING.read_text(encoding="utf-8").replace("__MATCHES__", cards)
            .replace("__REPO__", REPO_URL).replace("__GENERATED__", dt.date.today().isoformat()))


def write_landing(summaries: dict[str, dict], path: str | Path, images: list[Path] = ()) -> Path:
    """Write the landing page and copy screenshots into `assets/` next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_landing(summaries), encoding="utf-8")
    if images:
        (path.parent / "assets").mkdir(exist_ok=True)
        for img in images:
            if Path(img).exists():
                shutil.copy(img, path.parent / "assets" / Path(img).name)
    return path
