"""Time the models on a real match and write results/benchmarks.json.

    python scripts/benchmark.py        (downloads Metrica game 2 on first run)

Numbers depend on the machine; the JSON records Python, NumPy and the CPU
count next to them.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symmetrica import analysis, data, kinematics, pitch, pitch_control  # noqa: E402

GAME = 2


def timed(fn, repeats=1):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def main() -> None:
    data.download(GAME)
    (track, t_load) = timed(lambda: data.tracking(GAME))
    events = data.events(GAME)
    directions = pitch.attack_direction(track)
    vel, t_vel = timed(lambda: kinematics.velocities(track))

    # Pitch control per frame on two grid sizes, over 200 in-play frames spread across the match.
    rows = np.flatnonzero(np.isfinite(track["ball_x"].to_numpy()))
    rows = rows[np.linspace(0, len(rows) - 1, 200).astype(int)]
    states = [kinematics.frame_state(track, vel, int(r)) for r in rows]
    pc = {}
    for grid in (pitch_control.Grid(50, 32), pitch_control.Grid(32, 21)):
        times = []
        for s in states:
            t0 = time.perf_counter()
            surf = pitch_control.pitch_control(s, grid)
            times.append(time.perf_counter() - t0)
        total = surf.home + surf.away
        pc[f"{grid.nx}x{grid.ny}"] = {"median_ms": round(1000 * float(np.median(times)), 2),
                                      "p95_ms": round(1000 * float(np.percentile(times, 95)), 2),
                                      "min_probability_mass": round(float(total.min()), 4)}

    (vor, _), t_vor = timed(lambda: pitch_control.voronoi_series(track, directions, step=1))
    summary, t_all = timed(lambda: analysis.analyse_match(track, events, name=f"Metrica sample game {GAME}"))

    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    out = {
        "machine": {"python": platform.python_version(), "numpy": np.__version__,
                    "cpus": os.cpu_count(), "platform": platform.platform()},
        "match": {"game": GAME, "frames": int(len(track)), "minutes": round(len(track) / data.FPS / 60, 1)},
        "load_tracking_s": round(t_load, 2),
        "velocities_s": round(t_vel, 2),
        "pitch_control_per_frame": pc,
        "voronoi_all_frames": {"seconds": round(t_vor, 2),
                               "frames_per_second": int(len(vor) / t_vor), "grid": "21x14"},
        "full_report_analysis_s": round(t_all, 1),
        "report_stages_s": summary["timings"],
        "possession_accuracy_out_of_sample": {
            f"train {s['train']} -> test {s['test']}": round(s["control_with_memory"]["accuracy"], 4)
            for s in metrics["splits"]},
    }
    path = ROOT / "results" / "benchmarks.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
