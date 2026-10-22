"""Animate a short passage of play with the model's possession call per frame.

    python scripts/animate.py      (run scripts/evaluate.py first)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symmetrica import data, labels, models  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HOME, AWAY = "#2a78d6", "#eb6834"
NAMES = {1: ("Home", HOME), -1: ("Away", AWAY), 0: ("none", INK_2)}
GAME, SECONDS, STEP = 2, 20, 2  # every 2nd frame -> 12.5 fps, real time


def main():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    params = next(s["params"] for s in metrics["splits"] if s["test"] == GAME)
    track = data.tracking(GAME)
    y = labels.possession_labels(data.events(GAME), track.index, track["period"])
    d_home, d_away = models.nearest_distances(track)
    est = models.control_with_memory(d_home, d_away, models.ball_speed(track), **params)

    # Busiest 20 seconds: most labelled changes of possession.
    length = SECONDS * data.FPS
    changes = np.r_[0, (np.diff(y) != 0).astype(int)]
    start = int(np.argmax(np.convolve(changes, np.ones(length, dtype=int), mode="valid")))
    frames = range(start, start + length, STEP)

    cols = {t: [c for c in track.columns if c.startswith(f"{t}_") and c.endswith("_x")] for t in ("home", "away")}
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    L, W = data.PITCH_LENGTH / 2, data.PITCH_WIDTH / 2
    line = dict(color=GRID, lw=1.5, fill=False)
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, **line))
    ax.plot([0, 0], [-W, W], color=GRID, lw=1.5)
    ax.add_patch(Circle((0, 0), 9.15, **line))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((sx * L - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, **line))
    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-W - 9, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")

    dots = {t: ax.scatter([], [], s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            for t, c in (("home", HOME), ("away", AWAY))}
    ball = ax.scatter([], [], s=40, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
    ring = Circle((0, 0), 2.5, fill=False, lw=2, zorder=2)
    ax.add_patch(ring)
    title = ax.text(-L, W + 1.5, "", fontsize=11, color=INK)
    model_txt = ax.text(-L, -W - 4, "", fontsize=11, fontweight="bold")
    label_txt = ax.text(-L, -W - 7.5, "", fontsize=10, color=INK_2)

    def draw(f):
        row = track.iloc[f]
        for t, xs in cols.items():
            dots[t].set_offsets(np.c_[row[xs].to_numpy(float), row[[c[:-2] + "_y" for c in xs]].to_numpy(float)])
        ball.set_offsets([[row["ball_x"], row["ball_y"]]])
        m_name, m_col = NAMES[int(est[f])]
        ring.center = (row["ball_x"], row["ball_y"])
        ring.set_edgecolor(m_col)
        t = int(row["time"])
        title.set_text(f"Game {GAME} · {t // 60}:{t % 60:02d} · frame {track.index[f]}")
        model_txt.set_text(f"Model (tracking only): {m_name}")
        model_txt.set_color(m_col)
        hit = "✓" if est[f] == y[f] else ("dead ball" if y[f] == 0 else "✗")
        label_txt.set_text(f"Event data says: {NAMES[int(y[f])][0]}   {hit}")
        return list(dots.values()) + [ball, ring, title, model_txt, label_txt]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(ROOT / "figures" / "clip.gif", writer=PillowWriter(fps=data.FPS / STEP), dpi=90,
              savefig_kwargs={"facecolor": SURFACE})


if __name__ == "__main__":
    main()
