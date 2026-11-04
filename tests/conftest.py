"""Shared builders for synthetic tracking frames and event feeds.

Unit tests build tiny, hand-placed scenes whose answer is known analytically;
integration tests use one generated match (`symmetrica.synthetic`) so the
suite never needs the Metrica download.
"""
import numpy as np
import pandas as pd
import pytest

from symmetrica import analysis, synthetic
from symmetrica.data import EVENT_COLUMNS, FPS, PITCH_LENGTH, PITCH_WIDTH, prepare_events


def make_track(frames: list[dict], start_frame: int = 1) -> pd.DataFrame:
    """Tracking table from a list of {'home': [(x, y), ...], 'away': [...], 'ball': (x, y), 'period': p}.

    Home players are numbered 1.., away players 101.., matching how data.tracking names columns.
    Use None for a player who is off the pitch in that frame.
    """
    n_home = max(len(f.get("home", [])) for f in frames)
    n_away = max(len(f.get("away", [])) for f in frames)
    rows = []
    for i, f in enumerate(frames):
        r = {"frame": start_frame + i, "period": f.get("period", 1), "time": (start_frame + i) / FPS}
        for team, n, base in (("home", n_home, 1), ("away", n_away, 101)):
            pts = list(f.get(team, [])) + [None] * (n - len(f.get(team, [])))
            for k, p in enumerate(pts):
                r[f"{team}_{base + k}_x"], r[f"{team}_{base + k}_y"] = (np.nan, np.nan) if p is None else p
        r["ball_x"], r["ball_y"] = f.get("ball", (np.nan, np.nan))
        rows.append(r)
    return pd.DataFrame(rows).set_index("frame")


def repeat(frame: dict, n: int, **overrides) -> list[dict]:
    return [{**frame, **overrides} for _ in range(n)]


def make_events(rows: list[tuple]) -> pd.DataFrame:
    """Prepared event table from compact tuples in *metres*.

    Each tuple: (team, type, subtype, period, start_frame, end_frame, from, to, start_xy, end_xy)
    with `from` / `to` as player numbers (or None) and xy as (x, y) metres or None.
    """
    raw = []
    for team, typ, sub, period, f0, f1, frm, to, p0, p1 in rows:
        x0, y0 = ((p0[0] / PITCH_LENGTH + 0.5), (0.5 - p0[1] / PITCH_WIDTH)) if p0 else (np.nan, np.nan)
        x1, y1 = ((p1[0] / PITCH_LENGTH + 0.5), (0.5 - p1[1] / PITCH_WIDTH)) if p1 else (np.nan, np.nan)
        raw.append([team.capitalize(), typ, sub, period, f0, f0 / FPS, f1, f1 / FPS,
                    f"Player{frm}" if frm else np.nan, f"Player{to}" if to else np.nan, x0, y0, x1, y1])
    return prepare_events(pd.DataFrame(raw, columns=EVENT_COLUMNS))


# Home attacks +x in period 1 and -x in period 2 (the usual layout of these tests).
DIRECTIONS = {"home": {1: 1, 2: -1}, "away": {1: -1, 2: 1}}


@pytest.fixture(scope="session")
def synthetic_match():
    return synthetic.synthetic_match(minutes_per_half=1.5, seed=3)


@pytest.fixture(scope="session")
def synthetic_summary(synthetic_match):
    track, events = synthetic_match
    return analysis.analyse_match(track, events, name="Synthetic test match",
                                  pc_interval=30.0, max_clips=2, clip_step=25)
