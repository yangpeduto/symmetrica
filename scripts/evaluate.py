"""Tune on game 1, report on game 2 (and the reverse), write results/metrics.json.

    python scripts/evaluate.py
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symmetrica import data, labels, models  # noqa: E402

GRID = {
    "radius": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "max_speed": [4.0, 6.0, 8.0, 12.0, 99.0],
    "min_frames": [1, 3, 5, 8, 12],
}
TOLERANCE_FRAMES = data.FPS // 2  # ignore +-0.5 s around a labelled change of possession


def prepare(game: int) -> dict:
    data.download(game)
    track = data.tracking(game)
    y = labels.possession_labels(data.events(game), track.index, track["period"])
    d_home, d_away = models.nearest_distances(track)
    in_play = (y != 0) & np.isfinite(track["ball_x"].to_numpy())

    switch = np.flatnonzero(np.diff(y) != 0)
    near_switch = np.zeros(len(y), dtype=bool)
    for s in switch:
        near_switch[max(0, s - TOLERANCE_FRAMES): s + TOLERANCE_FRAMES + 1] = True

    return {"y": y, "d_home": d_home, "d_away": d_away, "speed": models.ball_speed(track),
            "mask": in_play, "mask_tolerant": in_play & ~near_switch}


def accuracy(est: np.ndarray, g: dict, tolerant: bool = False) -> float:
    m = g["mask_tolerant"] if tolerant else g["mask"]
    return float((est[m] == g["y"][m]).mean())


def tune(g: dict) -> tuple[dict, float]:
    best, best_acc = None, -1.0
    for radius, max_speed, min_frames in product(*GRID.values()):
        params = {"radius": radius, "max_speed": max_speed, "min_frames": min_frames}
        acc = accuracy(models.control_with_memory(g["d_home"], g["d_away"], g["speed"], **params), g)
        if acc > best_acc:
            best, best_acc = params, acc
    return best, best_acc


def main() -> None:
    games = {1: prepare(1), 2: prepare(2)}
    results = {"label_coverage": {g: float(v["mask"].mean()) for g, v in games.items()}, "splits": []}

    for train, test in ((1, 2), (2, 1)):
        params, train_acc = tune(games[train])
        g = games[test]
        baseline = models.nearest_player(g["d_home"], g["d_away"])
        tuned = models.control_with_memory(g["d_home"], g["d_away"], g["speed"], **params)
        split = {
            "train": train, "test": test, "params": params, "train_accuracy": train_acc,
            "nearest_player": {"accuracy": accuracy(baseline, g), "accuracy_tolerant": accuracy(baseline, g, True)},
            "control_with_memory": {"accuracy": accuracy(tuned, g), "accuracy_tolerant": accuracy(tuned, g, True)},
        }
        results["splits"].append(split)
        print(f"train game {train} -> test game {test}: {params}")
        print(f"  nearest player       {split['nearest_player']['accuracy']:.1%}"
              f"  (excl. +-0.5 s around changes: {split['nearest_player']['accuracy_tolerant']:.1%})")
        print(f"  control with memory  {split['control_with_memory']['accuracy']:.1%}"
              f"  (excl. +-0.5 s around changes: {split['control_with_memory']['accuracy_tolerant']:.1%})")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
