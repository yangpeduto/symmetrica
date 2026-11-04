// Regenerate the README / GitHub Pages screenshots from a built report.
//
//   node scripts/screenshots.js reports/game2.html figures
//
// Needs Node with Playwright (npm i playwright && npx playwright install chromium).
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const report = path.resolve(process.argv[2] || "reports/game2.html");
  const out = path.resolve(process.argv[3] || "figures");
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 2, colorScheme: "light" });
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto("file://" + report);
  await page.waitForTimeout(800);

  const box = sel => page.evaluate(s => {
    const r = document.querySelector(s).getBoundingClientRect();
    return { x: r.x, y: r.y + scrollY, width: r.width, height: r.height };
  }, sel);
  const shot = async (name, top, bottom, pad = 16) => {
    const a = await box(top), b = bottom ? await box(bottom) : a;
    await page.screenshot({ path: path.join(out, name), fullPage: true,
      clip: { x: a.x - pad, y: a.y - pad, width: a.width + 2 * pad, height: b.y + b.height - a.y + 2 * pad } });
  };

  // Overview: header, team comparison, insights.
  await shot("report_overview.png", "header.top", "section.card:nth-of-type(2)");

  // Replay: pause on the first clip whose pass the best option was not chosen for, layers on.
  await page.click("#play");
  const target = await page.evaluate(() => {
    const D = JSON.parse(document.getElementById("match-data").textContent);
    for (let ci = 0; ci < D.clips.length; ci++) {
      const c = D.clips[ci], p = (c.passes || []).find(q => q.best !== q.receiver) || (c.passes || [])[0];
      if (!p) continue;
      const t = (p.frame - c.frames[0].f) / 25 + 0.15, dur = (c.frames.length - 1) / c.fps;
      return { ci, v: Math.round(t / dur * 1000) };
    }
    return { ci: 0, v: 600 };
  });
  await page.selectOption("#clip", String(target.ci));
  await page.evaluate(v => { const s = document.getElementById("scrub"); s.value = v; s.dispatchEvent(new Event("input")); }, target.v);
  await page.waitForTimeout(200);
  await shot("report_pitch_control.png", "#replay");
  await page.check('[data-layer="shape"]');
  await page.click('[data-overlay="epv"]');
  await page.waitForTimeout(150);
  await shot("report_epv_shape.png", "#replay .canvas-wrap", "#passcap");
  await page.uncheck('[data-layer="shape"]');
  await page.click('[data-overlay="pc"]');

  await shot("report_momentum_networks.png", "#momentum", "section.grid.g2");
  await shot("report_shape_runs.png", "section.grid:has(#runs)", "section.card:has(#shapetable)");

  // Explorer: the top progressor against the busiest presser of the other team.
  await page.evaluate(() => {
    const D = JSON.parse(document.getElementById("match-data").textContent);
    const a = D.threat.leaders[0].player, team = D.players.find(p => p.id === a).team;
    const b = D.players.filter(p => p.team !== team && p.minutes >= 30 && p.role !== "GK").sort((x, y) => y.pressures - x.pressures)[0];
    const pa = document.getElementById("pa"), pb = document.getElementById("pb");
    pa.value = a; pa.dispatchEvent(new Event("change"));
    if (b) { pb.value = b.id; pb.dispatchEvent(new Event("change")); }
  });
  await page.waitForTimeout(200);
  await shot("report_explorer.png", "#explorer");

  await browser.close();
  if (errors.length) { console.error(errors); process.exit(1); }
  console.log("screenshots written to", out);
})();
