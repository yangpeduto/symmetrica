"""Player and ball velocities, and per-frame state snapshots.

Raw tracking positions jitter by a few centimetres per frame, which at 25 Hz
turns into metres per second of fake speed. Velocities are therefore taken
from a centred moving average of the positions and impossible values
(faster than any footballer) are discarded.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FPS, player_ids

MAX_PLAYER_SPEED = 12.0  # m/s; Usain Bolt peaks at ~12.4


def velocities(track: pd.DataFrame, window: int = 7, max_speed: float = MAX_PLAYER_SPEED) -> pd.DataFrame:
    """Per-frame vx, vy (m/s) for every player, same index as `track`.

    Positions are smoothed with a centred `window`-frame moving average
    before differencing; differences never straddle the half-time break.
    """
    stems = player_ids(track, "home") + player_ids(track, "away")
    x = track[[f"{s}_x" for s in stems]].to_numpy(float)
    y = track[[f"{s}_y" for s in stems]].to_numpy(float)
    period = track["period"].to_numpy()

    def smooth(a: np.ndarray) -> np.ndarray:
        return pd.DataFrame(a).rolling(window, center=True, min_periods=1).mean().to_numpy()

    vx, vy = np.full_like(x, np.nan), np.full_like(y, np.nan)
    for p in np.unique(period[np.isfinite(period)]):
        rows = np.flatnonzero(period == p)
        sx, sy = smooth(x[rows]), smooth(y[rows])
        vx[rows] = np.gradient(sx, axis=0) * FPS if len(rows) > 1 else 0.0
        vy[rows] = np.gradient(sy, axis=0) * FPS if len(rows) > 1 else 0.0
    too_fast = np.hypot(vx, vy) > max_speed
    vx[too_fast], vy[too_fast] = np.nan, np.nan
    cols = {}
    for i, s in enumerate(stems):
        cols[f"{s}_vx"], cols[f"{s}_vy"] = vx[:, i], vy[:, i]
    return pd.DataFrame(cols, index=track.index)


@dataclass
class FrameState:
    """Everything a spatial model needs about one frame. Players off the pitch are dropped."""

    home_ids: list[str]
    home_pos: np.ndarray  # (n, 2)
    home_vel: np.ndarray  # (n, 2)
    away_ids: list[str]
    away_pos: np.ndarray
    away_vel: np.ndarray
    ball: np.ndarray  # (2,)

    def team(self, team: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        if team == "home":
            return self.home_ids, self.home_pos, self.home_vel
        return self.away_ids, self.away_pos, self.away_vel


def frame_state(track: pd.DataFrame, vel: pd.DataFrame | None, pos: int) -> FrameState:
    """Snapshot of row `pos` (positional, not frame number). Missing velocity counts as standing still."""
    row = track.iloc[pos]
    vrow = vel.iloc[pos] if vel is not None else None
    parts = {}
    for team in ("home", "away"):
        ids, p, v = [], [], []
        for s in player_ids(track, team):
            x, y = row[f"{s}_x"], row[f"{s}_y"]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ids.append(s)
            p.append((x, y))
            if vrow is None:
                v.append((0.0, 0.0))
            else:
                vx, vy = vrow.get(f"{s}_vx", 0.0), vrow.get(f"{s}_vy", 0.0)
                v.append((vx if np.isfinite(vx) else 0.0, vy if np.isfinite(vy) else 0.0))
        parts[team] = (ids, np.array(p, float).reshape(-1, 2), np.array(v, float).reshape(-1, 2))
    return FrameState(*parts["home"], *parts["away"],
                      ball=np.array([row["ball_x"], row["ball_y"]], dtype=float))
