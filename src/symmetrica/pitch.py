"""Pitch geometry shared by every metric.

Coordinates are metres with (0, 0) at the centre spot, x along the length
(-52.5 .. 52.5) and y across the width (-34 .. 34). Teams swap ends at half
time, so most metrics work in an *attacking frame*: x multiplied by the
team's direction, after which that team always attacks towards +x.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FPS, PITCH_LENGTH, PITCH_WIDTH, player_arrays

HALF_L = PITCH_LENGTH / 2
HALF_W = PITCH_WIDTH / 2
GOAL_WIDTH = 7.32
TEAMS = ("home", "away")


def other(team: str) -> str:
    return "away" if team == "home" else "home"


@dataclass(frozen=True)
class Grid:
    """Regular grid of cell centres covering the pitch."""

    nx: int = 50
    ny: int = 32

    @property
    def xs(self) -> np.ndarray:
        step = PITCH_LENGTH / self.nx
        return -HALF_L + step * (np.arange(self.nx) + 0.5)

    @property
    def ys(self) -> np.ndarray:
        step = PITCH_WIDTH / self.ny
        return -HALF_W + step * (np.arange(self.ny) + 0.5)

    @property
    def cells(self) -> np.ndarray:
        """(ny * nx, 2) array of cell centres, row-major with y as the slow axis."""
        gx, gy = np.meshgrid(self.xs, self.ys)
        return np.c_[gx.ravel(), gy.ravel()]

    @property
    def cell_area(self) -> float:
        return (PITCH_LENGTH / self.nx) * (PITCH_WIDTH / self.ny)


def attack_direction(track: pd.DataFrame, seconds: float = 5.0) -> dict[str, dict[int, int]]:
    """Which way each team attacks in each period: +1 towards +x, -1 towards -x.

    At kick-off every player stands in his own half, so the team whose
    centroid is further left attacks to the right. Uses the first few seconds
    of each period, which is robust to substitutions and red cards.
    """
    hx, _ = player_arrays(track, "home")
    ax, _ = player_arrays(track, "away")
    period = track["period"].to_numpy()
    out: dict[str, dict[int, int]] = {"home": {}, "away": {}}
    for p in (int(v) for v in np.unique(period[np.isfinite(period)])):
        idx = np.flatnonzero(period == p)[: int(seconds * FPS)]
        home_x, away_x = np.nanmean(hx[idx]), np.nanmean(ax[idx])
        home_dir = 1 if home_x < away_x else -1
        out["home"][p], out["away"][p] = home_dir, -home_dir
    return out


def direction_per_frame(track: pd.DataFrame, team: str, directions: dict) -> np.ndarray:
    """Attacking direction (+1 / -1) of `team` for every frame, 0 where the period is unknown."""
    period = track["period"].to_numpy()
    out = np.zeros(len(track), dtype=np.int8)
    for p, d in directions[team].items():
        out[period == p] = d
    return out


def distance_to_goal(x_att: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance (m) from a point in the attacking frame to the centre of the goal being attacked."""
    return np.hypot(HALF_L - np.asarray(x_att), np.asarray(y))


def goal_angle(x_att: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle (rad) subtended by the goal mouth, the classic geometric shot-quality feature."""
    dx = np.maximum(HALF_L - np.asarray(x_att, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    a = np.arctan2(GOAL_WIDTH / 2 - y, dx) - np.arctan2(-GOAL_WIDTH / 2 - y, dx)
    return np.abs(a)


def zone_of(x_att: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thirds (0 defensive, 1 middle, 2 final) and channels (0 left, 1 centre, 2 right)."""
    third = np.clip(((np.asarray(x_att) + HALF_L) // (PITCH_LENGTH / 3)).astype(int), 0, 2)
    # Channels split the width at the edges of the penalty area (40.32 m wide).
    y = np.asarray(y)
    channel = np.where(y > 20.16, 0, np.where(y < -20.16, 2, 1))
    return third, channel
