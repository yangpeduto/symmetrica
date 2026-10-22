"""Figures for the README: a possession timeline and a single-frame pitch view.

    python scripts/figures.py      (run scripts/evaluate.py first)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symmetrica import data, labels, models  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
HOME = "#2a78d6"   # categorical slot 1
AWAY = "#eb6834"   # categorical slot 2
DEAD = "#d9d8d3"
GAME = 2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})


def strip(ax, values, y0, title):
    colours = np.where(values == 1, HOME, np.where(values == -1, AWAY, DEAD))
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for s, e in zip(starts, ends):
        ax.add_patch(Rectangle((s, y0), e - s, 0.8, color=colours[s], lw=0))
    ax.text(-30, y0 + 0.4, title, ha="right", va="center", color=INK_2)


def busiest_window(y, length):
    """Start of the window with the most labelled changes of possession."""
    changes = np.r_[0, (np.diff(y) != 0).astype(int)]
    counts = np.convolve(changes, np.ones(length, dtype=int), mode="valid")
    return int(np.argmax(counts))


def timeline(track, y, est, params):
    length = 120 * data.FPS  # two minutes of play
    start = busiest_window(y, length)
    sl = slice(start, start + length)
    t0 = int(track["time"].iloc[start])
    period = int(track["period"].iloc[start])
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    strip(ax, y[sl], 1.0, "Event data (label)")
    strip(ax, est[sl], 0.0, "Tracking only (model)")
    n = sl.stop - sl.start
    ax.set_xlim(0, n)
    ax.set_ylim(-0.3, 2.2)
    ax.set_yticks([])
    ticks = np.arange(0, n + 1, 20 * data.FPS)
    ax.set_xticks(ticks, [f"{(t0 + t // data.FPS) // 60}:{(t0 + t // data.FPS) % 60:02d}" for t in ticks])
    ax.set_xlabel(f"Match clock (period {period}), the two busiest minutes of game {GAME}")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (HOME, AWAY, DEAD)]
    ax.legend(handles, ["Home", "Away", "Dead ball / none"], ncol=3, frameon=False,
              loc="upper left", bbox_to_anchor=(0, 1.25))
    fig.savefig(ROOT / "figures" / "timeline.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def pitch_frame(track, est, frame_pos):
    row = track.iloc[frame_pos]
    fig, ax = plt.subplots(figsize=(8, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    L, W = data.PITCH_LENGTH / 2, data.PITCH_WIDTH / 2
    line = dict(color=GRID, lw=1.5, fill=False)
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, **line))
    ax.plot([0, 0], [-W, W], color=GRID, lw=1.5)
    ax.add_patch(Circle((0, 0), 9.15, **line))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((sx * L - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, **line))

    for team, colour in (("home", HOME), ("away", AWAY)):
        xs = [c for c in track.columns if c.startswith(f"{team}_") and c.endswith("_x")]
        px, py = row[xs].to_numpy(float), row[[c[:-2] + "_y" for c in xs]].to_numpy(float)
        ax.scatter(px, py, s=90, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=team.capitalize())
    bx, by = row["ball_x"], row["ball_y"]
    ax.scatter([bx], [by], s=40, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4, label="Ball")
    owner = {1: ("Home", HOME), -1: ("Away", AWAY)}.get(int(est[frame_pos]), ("none", INK_2))
    ax.set_title(f"Frame {track.index[frame_pos]}  ·  model says: {owner[0]} in possession",
                 loc="left", color=INK, fontsize=11)
    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-W - 3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.savefig(ROOT / "figures" / "pitch_frame.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    params = next(s["params"] for s in metrics["splits"] if s["test"] == GAME)
    track = data.tracking(GAME)
    y = labels.possession_labels(data.events(GAME), track.index, track["period"])
    d_home, d_away = models.nearest_distances(track)
    est = models.control_with_memory(d_home, d_away, models.ball_speed(track), **params)
    (ROOT / "figures").mkdir(exist_ok=True)
    timeline(track, y, est, params)
    start = int(np.flatnonzero(track["period"].to_numpy() == 1)[0]) + 15 * 60 * data.FPS
    pitch_frame(track, est, start + 40 * data.FPS)


if __name__ == "__main__":
    main()
