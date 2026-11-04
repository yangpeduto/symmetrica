"""Space dominance: who controls which part of the pitch, frame by frame.

Two models, from cheap to physical:

* **Voronoi dominance.** Every location belongs to the team of the nearest
  player. Purely geometric, fast enough to run on every frame of a match
  (vectorised over frames x cells x players).

* **Probabilistic pitch control** (Spearman, 2018, "Beyond Expected Goals").
  For a target location r the ball needs t_ball(r) = |r - ball| / v_ball to
  arrive. Player j needs

      tau_j(r) = t_react + |r - (p_j + v_j * t_react)| / v_max

  i.e. he keeps running on his current velocity for a reaction time, then
  sprints straight to r. Because nobody hits that time exactly, the chance he
  has arrived by time t is a logistic in (t - tau_j) with spread sigma, and
  once there he controls the ball at rate lambda. The control probability of
  each player then follows

      dP_j/dt = (1 - sum_k P_k(t)) * f_j(t) * lambda,
      f_j(t)  = 1 / (1 + exp(-pi / (sqrt(3) * sigma) * (t - tau_j)))

  integrated from t_ball until the probabilities sum to ~1. Summing P_j over
  each team gives the team's control of r. The integration is vectorised
  over all cells and players at once, so a full-pitch 50 x 32 surface costs
  a few tens of milliseconds.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import player_arrays
from .kinematics import FrameState, frame_state
from .pitch import HALF_L, PITCH_LENGTH, Grid, direction_per_frame


@dataclass(frozen=True)
class PitchControlParams:
    """Defaults follow Spearman (2018) as popularised by Shaw's open implementation."""

    max_player_speed: float = 5.0   # m/s, sustained sprint towards the ball
    reaction_time: float = 0.7      # s before a player can change direction
    tti_sigma: float = 0.45         # s, uncertainty of the arrival time
    lambda_att: float = 4.3         # 1/s, ball-control rate once arrived
    kappa_def: float = 1.0          # defender advantage (lambda_def = kappa * lambda_att)
    ball_speed: float = 15.0        # m/s, average ball speed of a pass
    dt: float = 0.04                # s, integration step (one tracking frame)
    max_time: float = 10.0          # s, integration horizon after the ball arrives
    tolerance: float = 0.01         # stop once 1 - sum(P) < tolerance everywhere


@dataclass
class PitchControlSurface:
    home: np.ndarray       # (ny, nx) probability that home controls the ball there
    away: np.ndarray       # (ny, nx)
    grid: Grid

    @property
    def home_share(self) -> float:
        total = self.home + self.away
        return float(np.sum(self.home) / np.sum(total)) if np.sum(total) > 0 else 0.5


def time_to_intercept(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray,
                      params: PitchControlParams) -> np.ndarray:
    """(n_targets, n_players) arrival times tau_j(r) in seconds."""
    if len(pos) == 0:
        return np.empty((len(targets), 0))
    reaction = pos + vel * params.reaction_time
    d = np.linalg.norm(targets[:, None, :] - reaction[None, :, :], axis=2)
    return params.reaction_time + d / params.max_player_speed


def pitch_control_at(targets: np.ndarray, ball: np.ndarray,
                     att_pos: np.ndarray, att_vel: np.ndarray,
                     def_pos: np.ndarray, def_vel: np.ndarray,
                     params: PitchControlParams = PitchControlParams()) -> tuple[np.ndarray, np.ndarray]:
    """Control probability of the attacking and defending team at each target.

    Returns two (n_targets,) arrays. They sum to ~1 (to within `tolerance`)
    wherever at least one player is on the pitch.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    n = len(targets)
    tau = np.concatenate([time_to_intercept(att_pos, att_vel, targets, params),
                          time_to_intercept(def_pos, def_vel, targets, params)], axis=1)
    n_att = len(att_pos)
    if tau.shape[1] == 0:
        return np.zeros(n), np.zeros(n)
    lam = np.r_[np.full(n_att, params.lambda_att),
                np.full(len(def_pos), params.lambda_att * params.kappa_def)]

    ball = np.asarray(ball, dtype=float)
    if np.all(np.isfinite(ball)):
        t_ball = np.linalg.norm(targets - ball, axis=1) / params.ball_speed
    else:
        t_ball = np.zeros(n)
    # Skip straight to the moment the first player could plausibly arrive:
    # before that every f_j is ~0 and the integration does no work.
    t0 = np.maximum(t_ball, tau.min(axis=1) - 4 * params.tti_sigma)

    scale = np.pi / (np.sqrt(3.0) * params.tti_sigma)
    P = np.zeros_like(tau)
    remaining = np.ones(n)
    for k in range(1, int(params.max_time / params.dt) + 1):
        t = (t0 + k * params.dt)[:, None]
        f = 1.0 / (1.0 + np.exp(-scale * (t - tau)))
        dP = remaining[:, None] * f * lam * params.dt
        # A step can't hand out more probability than is left.
        total = dP.sum(axis=1)
        over = total > remaining
        if np.any(over):
            dP[over] *= (remaining[over] / total[over])[:, None]
        P += dP
        remaining = 1.0 - P.sum(axis=1)
        if np.all(remaining < params.tolerance):
            break
    return P[:, :n_att].sum(axis=1), P[:, n_att:].sum(axis=1)


def pitch_control(state: FrameState, grid: Grid = Grid(),
                  params: PitchControlParams = PitchControlParams()) -> PitchControlSurface:
    """Full-pitch control surface for one frame (home as the reference team)."""
    home, away = pitch_control_at(grid.cells, state.ball, state.home_pos, state.home_vel,
                                  state.away_pos, state.away_vel, params)
    return PitchControlSurface(home.reshape(grid.ny, grid.nx), away.reshape(grid.ny, grid.nx), grid)


def voronoi(home_pos: np.ndarray, away_pos: np.ndarray, grid: Grid = Grid()) -> np.ndarray:
    """(ny, nx) map: +1 where the nearest player is home, -1 where away, 0 if nobody is on."""
    cells = grid.cells
    inf = np.full(len(cells), np.inf)
    dh = np.linalg.norm(cells[:, None] - home_pos[None], axis=2).min(axis=1) if len(home_pos) else inf
    da = np.linalg.norm(cells[:, None] - away_pos[None], axis=2).min(axis=1) if len(away_pos) else inf
    out = np.where(dh < da, 1, -1).astype(np.int8)
    out[~np.isfinite(np.minimum(dh, da))] = 0
    return out.reshape(grid.ny, grid.nx)


def _nearest_sq(cells: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """(frames, cells) squared distance from each cell to the nearest player."""
    dx = cells[None, :, 0, None] - px[:, None, :]
    dy = cells[None, :, 1, None] - py[:, None, :]
    d2 = dx * dx + dy * dy
    return np.min(np.where(np.isnan(d2), np.inf, d2), axis=2)


def voronoi_series(track: pd.DataFrame, directions: dict, grid: Grid = Grid(21, 14),
                   step: int = 5, chunk: int = 512) -> tuple[pd.DataFrame, np.ndarray]:
    """Voronoi dominance for every `step`-th frame of a match.

    Returns
      * a DataFrame indexed by frame with `home_share` (share of the pitch
        nearer to a home player), and `home_final_third` / `away_final_third`
        (share of the final third each team attacks that it dominates);
      * the average home-dominance map (ny, nx) in the home *attacking frame*,
        i.e. with home always attacking left to right.
    """
    rows = np.arange(0, len(track), step)
    hx, hy = (a[rows].astype(np.float32) for a in player_arrays(track, "home"))
    ax, ay = (a[rows].astype(np.float32) for a in player_arrays(track, "away"))
    home_dir = direction_per_frame(track, "home", directions)[rows]
    cells = grid.cells.astype(np.float32)
    att_third_home = cells[:, 0] > HALF_L - PITCH_LENGTH / 3
    att_third_away = cells[:, 0] < -HALF_L + PITCH_LENGTH / 3

    share = np.full(len(rows), np.nan)
    home_ft = np.full(len(rows), np.nan)
    away_ft = np.full(len(rows), np.nan)
    acc_map = np.zeros(len(cells))
    acc_n = 0
    for s in range(0, len(rows), chunk):
        sl = slice(s, s + chunk)
        dh = _nearest_sq(cells, hx[sl], hy[sl])
        da = _nearest_sq(cells, ax[sl], ay[sl])
        valid = np.isfinite(dh).any(axis=1) & np.isfinite(da).any(axis=1)
        home_cell = (dh < da).astype(float)
        # home_dir +1: home attacks +x, so its final third is the right one.
        d = home_dir[sl]
        right = home_cell[:, att_third_home].mean(axis=1)
        left = home_cell[:, att_third_away].mean(axis=1)
        share[sl] = np.where(valid, home_cell.mean(axis=1), np.nan)
        home_ft[sl] = np.where(valid & (d != 0), np.where(d > 0, right, left), np.nan)
        away_ft[sl] = np.where(valid & (d != 0), np.where(d > 0, 1 - left, 1 - right), np.nan)
        # Accumulate in the home attacking frame: flip the x axis when home attacks -x.
        m = home_cell.reshape(-1, grid.ny, grid.nx)
        m = np.where((d < 0)[:, None, None], m[:, :, ::-1], m)
        keep = valid & (d != 0)
        acc_map += m[keep].reshape(-1, len(cells)).sum(axis=0)
        acc_n += int(keep.sum())

    df = pd.DataFrame({"home_share": share, "home_final_third": home_ft, "away_final_third": away_ft},
                      index=track.index[rows])
    mean_map = (acc_map / max(acc_n, 1)).reshape(grid.ny, grid.nx)
    return df, mean_map


def pitch_control_series(track: pd.DataFrame, vel: pd.DataFrame, positions: np.ndarray,
                         grid: Grid = Grid(32, 21),
                         params: PitchControlParams = PitchControlParams()) -> pd.DataFrame:
    """Home share of probabilistic pitch control at selected row positions."""
    shares = []
    for pos in positions:
        state = frame_state(track, vel, int(pos))
        if not np.all(np.isfinite(state.ball)) or len(state.home_pos) == 0 or len(state.away_pos) == 0:
            shares.append(np.nan)
            continue
        shares.append(pitch_control(state, grid, params).home_share)
    return pd.DataFrame({"home_share": shares}, index=track.index[np.asarray(positions, dtype=int)])
