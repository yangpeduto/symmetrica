"""Build the interactive tactical match report (one self-contained HTML file per match).

    python scripts/report.py                  # Metrica sample game 2 -> reports/game2.html
    python scripts/report.py --game 1 --game 2
    python scripts/report.py --synthetic      # offline demo, no download needed
    python scripts/report.py --game 2 --json  # also write the metric summary as JSON
    python scripts/report.py --game 1 --game 2 --docs docs   # + GitHub Pages site in docs/

The xT surface is fitted on every Metrica game passed (pooled), so each
report values actions on the same scale, and player percentiles are ranked
against every player of those games.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symmetrica import analysis, data, pitch, players, report, synthetic, threat  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", type=int, action="append", choices=[1, 2], help="Metrica sample game (repeatable)")
    ap.add_argument("--synthetic", action="store_true", help="use a generated match instead of Metrica data")
    ap.add_argument("--out", type=Path, default=ROOT / "reports", help="output directory")
    ap.add_argument("--clips", type=int, default=5, help="number of animated moments")
    ap.add_argument("--json", action="store_true", help="also write the summary JSON")
    ap.add_argument("--docs", type=Path, help="also write a GitHub Pages site (reports + index.html) here")
    args = ap.parse_args()

    matches = {}
    if args.synthetic:
        matches["synthetic"] = ("Synthetic match", *synthetic.synthetic_match(minutes_per_half=8.0))
    for g in args.game or ([] if args.synthetic else [2]):
        data.download(g)
        matches[f"game{g}"] = (f"Metrica sample game {g}", data.tracking(g), data.events(g))

    xt = None
    real = [k for k in matches if k.startswith("game")]
    if real:
        acts = [threat.actions(e, pitch.attack_direction(t)) for k, (_, t, e) in matches.items() if k in real]
        xt = threat.ExpectedThreat.fit(pd.concat(acts, ignore_index=True))

    summaries, timings = {}, {}
    for key, (name, track, events) in matches.items():
        t0 = time.perf_counter()
        summaries[key] = analysis.analyse_match(track, events, name=name, xt=xt if key in real else None,
                                                max_clips=args.clips)
        timings[key] = time.perf_counter() - t0

    # Rank players against everyone in the real games analysed together.
    if len(real) > 1:
        pool = [p for k in real for p in summaries[k]["players"]]
        for k in real:
            players.add_percentiles(summaries[k]["players"], reference=pool)

    for key, summary in summaries.items():
        path = report.write_report(summary, args.out / f"{key}.html")
        if args.json:
            light = {k: v for k, v in summary.items() if k != "clips"}
            (args.out / f"{key}.json").write_text(json.dumps(light, indent=1))
        if args.docs:
            report.write_report(summary, args.docs / f"{key}.html")
        size = path.stat().st_size / 1e6
        print(f"{summary['meta']['name']}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
              f"({size:.1f} MB, {timings[key]:.1f} s)")
    if args.docs:
        shots = sorted((ROOT / "figures").glob("report_*.png"))
        index = report.write_landing({k: summaries[k] for k in real or summaries}, args.docs / "index.html", shots)
        print(f"GitHub Pages site: {index}")


if __name__ == "__main__":
    main()
