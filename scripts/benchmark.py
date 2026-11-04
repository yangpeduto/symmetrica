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
import pandas as pd  # noqa: E402

from symmetrica import (analysis, data, kinematics, labels, offball, pass_options, pitch, pitch_control,  # noqa: E402
                        shape, threat)

GAME = 2


def timed(fn, repeats=1):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive scores above a random negative (Mann-Whitney)."""
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pass_model_validation(track, events, vel, directions, xt,
                          params=pitch_control.PitchControlParams()) -> dict:
    """Does modelled completion separate completed from intercepted passes?

    Completed passes are scored at their actual receiver. For intercepted
    passes the intended receiver is unknown; the teammate nearest to where
    the ball was cut out stands in for him.
    """
    index = {f: i for i, f in enumerate(track.index)}
    rows = []
    for e in events.itertuples(index=False):
        completed = e.type == "PASS" and e.receiver is not None
        failed = e.type == "BALL LOST" and "INTERCEPTION" in e.subtype and np.isfinite(e.end_x)
        if not (completed or failed) or int(e.start_frame) not in index:
            continue
        st = kinematics.frame_state(track, vel, index[int(e.start_frame)])
        d = directions[e.team].get(int(e.period), 0)
        opts = pass_options.evaluate_options(st, e.team, e.player, d, xt, params)
        if not opts:
            continue
        if completed:
            target = next((o for o in opts if o["id"] == e.receiver), None)
        else:
            target = min(opts, key=lambda o: np.hypot(o["x"] - e.end_x, o["y"] - e.end_y))
        if target is not None:
            rows.append((target["completion"], completed))
    c = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return {"passes": int(y.sum()), "intercepted": int((~y).sum()),
            "mean_completion_completed": round(float(c[y].mean()), 3),
            "mean_completion_intercepted": round(float(c[~y].mean()), 3),
            "auc": round(auc(c, y), 3)}


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
    for motion in ("physics", "linear"):
        params = pitch_control.PitchControlParams(motion=motion) if motion == "physics" else \
            pitch_control.PitchControlParams(motion="linear", tti_sigma=0.45)
        for grid in (pitch_control.Grid(50, 32), pitch_control.Grid(32, 21)):
            times = []
            for s in states:
                t0 = time.perf_counter()
                surf = pitch_control.pitch_control(s, grid, params)
                times.append(time.perf_counter() - t0)
            total = surf.home + surf.away
            pc[f"{motion} {grid.nx}x{grid.ny}"] = {"median_ms": round(1000 * float(np.median(times)), 2),
                                                   "p95_ms": round(1000 * float(np.percentile(times, 95)), 2),
                                                   "min_probability_mass": round(float(total.min()), 4)}
    cells = pitch_control.Grid(50, 32).cells
    t0 = time.perf_counter()
    for s in states[:50]:
        pitch_control.physics_arrival_time(np.r_[s.home_pos, s.away_pos], np.r_[s.home_vel, s.away_vel], cells)
    arrival_ms = 1000 * (time.perf_counter() - t0) / 50

    # Models fitted / run the way the report runs them.
    acts = pd.concat([threat.actions(data.events(g), pitch.attack_direction(data.tracking(g))) for g in (1, 2)])
    xt = threat.ExpectedThreat.fit(acts)
    poss = labels.possession_labels(events, track.index, track["period"])
    dec, t_dec = timed(lambda: pass_options.pass_decisions(events, track, vel, directions, xt))
    runs, t_runs = timed(lambda: offball.off_ball_runs(track, vel, poss, directions, xt))
    shapes, t_shape = timed(lambda: shape.shape_series(track, poss, directions))
    # Physics model with the report's defaults vs Spearman's original (linear, sigma 0.45).
    validation = {
        "physics": pass_model_validation(track, events, vel, directions, xt, pitch_control.PitchControlParams()),
        "linear_spearman": pass_model_validation(track, events, vel, directions, xt,
                                                 pitch_control.PitchControlParams(motion="linear", tti_sigma=0.45)),
    }

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
        "physics_arrival_times_50x32_x_22_players_ms": round(arrival_ms, 2),
        "pass_options": {"passes": int(len(dec)), "seconds": round(t_dec, 2),
                         "ms_per_pass": round(1000 * t_dec / max(len(dec), 1), 2)},
        "pass_completion_validation": validation,
        "offball_runs": {"runs": int(len(runs)), "seconds_incl_counterfactual_space": round(t_runs, 2)},
        "shape_series": {"samples": int(len(shapes)), "seconds": round(t_shape, 2)},
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
