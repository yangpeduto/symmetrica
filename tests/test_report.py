"""End-to-end: analyse a synthetic match and render the self-contained HTML report."""
import base64
import json
import re
import shutil
import subprocess

import pytest

from symmetrica import report


def test_summary_is_complete_and_strict_json(synthetic_match, synthetic_summary):
    s = synthetic_summary
    json.dumps(s, allow_nan=False)  # no NaN / inf / numpy types anywhere
    for key in ("meta", "possession", "space", "pressing", "threat", "momentum", "networks", "insights", "clips"):
        assert key in s
    _, ev = synthetic_match
    goals = ev[(ev["type"] == "SHOT") & ev["subtype"].str.endswith("GOAL")]
    assert s["meta"]["score"] == {t: int((goals["team"] == t).sum()) for t in ("home", "away")}
    assert 0.0 <= s["space"]["home_share"] <= 1.0
    assert s["possession"]["tracking_model_accuracy"] > 0.7
    assert len(s["momentum"]["timeline"]) == 3
    assert s["insights"] and all(i["headline"] and i["detail"] for i in s["insights"])
    assert len(s["threat"]["xt_grid"]) == 8 and len(s["threat"]["xt_grid"][0]) == 12


def test_clips_carry_decodable_pitch_control(synthetic_summary):
    clip = synthetic_summary["clips"][0]
    nx, ny = clip["grid"]
    f = next(fr for fr in clip["frames"] if fr["pc"])
    surface = base64.b64decode(f["pc"])
    assert len(surface) == nx * ny
    assert len(f["home"]) == 11 and len(f["away"]) == 11
    assert f["pos"] in (-1, 0, 1)


def test_html_is_self_contained_and_escapes_its_payload(synthetic_summary):
    s = dict(synthetic_summary, meta={**synthetic_summary["meta"], "name": "A </script><b>tricky</b> name"})
    page = report.render_html(s)
    assert page.startswith("<!doctype html>")
    assert "__DATA__" not in page and "__TITLE__" not in page
    # No external resources: no src/href pointing at a URL, no @import, no fetch.
    assert not re.search(r'(src|href)\s*=\s*["\']?(https?:)?//', page)
    assert "@import" not in page and "fetch(" not in page
    # The embedded JSON round-trips and the hostile name cannot close the script tag early.
    blob = re.search(r'<script id="match-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(blob.replace("<\\/", "</"))["meta"]["name"] == "A </script><b>tricky</b> name"
    assert "&lt;/script&gt;" in page.split("</title>")[0]
    assert page.count("</script>") == 2


def test_write_report(tmp_path, synthetic_summary):
    path = report.write_report(synthetic_summary, tmp_path / "out" / "match.html")
    assert path.exists() and path.stat().st_size > 10_000


def _node_playwright() -> bool:
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _node_playwright(), reason="needs node + playwright for a headless browser")
def test_report_renders_without_errors_in_a_browser(tmp_path, synthetic_summary):
    path = report.write_report(synthetic_summary, tmp_path / "match.html")
    script = tmp_path / "check.js"
    script.write_text("""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  p.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await p.goto('file://' + process.argv[2]);
  await p.waitForTimeout(800);
  const out = await p.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > innerWidth,
    insights: document.querySelectorAll('#insights li').length,
    kpis: document.querySelectorAll('#kpi tr').length,
    svgs: document.querySelectorAll('svg').length,
    canvas: document.getElementById('pitch').width,
  }));
  await p.click('#theme');
  await p.click('[data-overlay="voronoi"]');
  await p.waitForTimeout(200);
  console.log(JSON.stringify({ ...out, errors }));
  await b.close();
})();
""")
    res = subprocess.run(["node", str(script), str(path)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["errors"] == []
    assert not out["overflow"]
    assert out["insights"] == len(synthetic_summary["insights"])
    assert out["kpis"] == 13 and out["svgs"] >= 6 and out["canvas"] > 0
