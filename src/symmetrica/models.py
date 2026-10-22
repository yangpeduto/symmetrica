"""Possession estimators that only look at tracking data.

All of them return one value per frame: 1 = home, -1 = away, 0 = no estimate.
"""
import numpy as np
import pandas as pd

from .data import FPS, player_arrays


def nearest_distances(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from the ball to the closest home and closest away player per frame."""
    bx = track["ball_x"].to_numpy()[:, None]
    by = track["ball_y"].to_numpy()[:, None]
    out = []
    for team in ("home", "away"):
        px, py = player_arrays(track, team)
        with np.errstate(invalid="ignore"):
            d = np.hypot(px - bx, py - by)
        out.append(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
    return out[0], out[1]


def ball_speed(track: pd.DataFrame) -> np.ndarray:
    """Ball speed in m/s, smoothed over 5 frames (0.2 s)."""
    x = track["ball_x"].interpolate(limit=5)
    y = track["ball_y"].interpolate(limit=5)
    speed = np.hypot(x.diff(), y.diff()) * FPS
    return speed.rolling(5, center=True, min_periods=1).mean().to_numpy()


def nearest_player(d_home: np.ndarray, d_away: np.ndarray) -> np.ndarray:
    """Baseline: whichever team has the player closest to the ball."""
    est = np.where(d_home < d_away, 1, -1).astype(np.int8)
    est[~np.isfinite(np.minimum(d_home, d_away))] = 0
    return est


def control_with_memory(d_home: np.ndarray, d_away: np.ndarray, speed: np.ndarray,
                        radius: float, max_speed: float, min_frames: int) -> np.ndarray:
    """Possession only changes when a player actually controls the ball.

    A team gains control when its closest player is within `radius` metres of a
    ball moving slower than `max_speed` m/s and is clearly closer than the
    nearest opponent. While the ball travels (passes, clearances) the last team
    in control keeps possession. A change only counts after `min_frames`
    consecutive frames of control, which filters out deflections.
    """
    n = len(d_home)
    closest = np.minimum(d_home, d_away)
    controller = np.where(d_home < d_away, 1, -1)
    in_control = (closest < radius) & (np.nan_to_num(speed, nan=99.0) < max_speed)
    candidate = np.where(in_control, controller, 0)

    est = np.zeros(n, dtype=np.int8)
    current, run_team, run_len = 0, 0, 0
    for i in range(n):
        c = candidate[i]
        if c != 0 and c == run_team:
            run_len += 1
        elif c != 0:
            run_team, run_len = c, 1
        else:
            run_len = 0
        if run_team != current and run_len >= min_frames:
            current = run_team
        if not np.isfinite(closest[i]):
            current, run_team, run_len = 0, 0, 0  # ball not tracked: dead ball
        est[i] = current
    return est
