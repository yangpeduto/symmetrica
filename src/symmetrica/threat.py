"""Progression and threat: expected threat (xT), progressive and line-breaking passes.

**Expected threat** (Singh, 2019). Split the pitch (in the attacking frame)
into cells s. From the event data estimate, for each cell, the probability
of shooting s_s, of moving the ball m_s, the transition matrix T[s, z] of
*successful* moves from s to z (failed moves lead to value 0), and the
probability that a shot from s scores, xG_s. xT is the fixed point of

    xT_s = s_s * xG_s + m_s * sum_z T[s, z] * xT_z

found by value iteration. Two matches are a thin sample for a 12 x 8 grid,
so the counts are smoothed with a Gaussian kernel over neighbouring cells,
and xG_s comes from a geometric (distance + goal-angle) logistic prior
instead of the ~50 shots in the data. The value of an action is
xT(end) - xT(start).

**Progressive pass** (Wyscout's rule). A completed pass that brings the ball
at least 30 m closer to goal inside the own half, 15 m when it crosses the
halfway line, or 10 m inside the opponent half.

**Line-breaking pass.** At the moment of the pass, the opponent's outfield
players are grouped into defensive lines by splitting their x positions at
the largest gaps (natural breaks). A completed pass breaks a line when it
starts at least 2 m behind it and is received at least 2 m beyond it.

**Packing** (outplayed opponents). Opponents goal-side of the ball when the
pass is played minus opponents goal-side of the ball when it is received,
both from tracking. It counts players taken out of the game by the pass.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PITCH_LENGTH, PITCH_WIDTH, player_ids
from .pitch import HALF_L, HALF_W, TEAMS, distance_to_goal, goal_angle, other
from .pressing import possession_chains

# logit(xG) = b0 + b_angle * goal_angle + b_dist * distance, calibrated to
# public rule-of-thumb values: ~0.6 six-yard box, ~0.25 penalty spot,
# ~0.11 edge of the box, ~0.04 at 25 m, ~0.06 from a tight angle at 5 m.
XG_COEF = (-1.75, 2.22, -0.082)
NON_ACTIONS = {"END HALF", "REFEREE HIT", "OFFSIDE"}
MIN_CARRY = 3.0  # m; shorter displacements between touches are just a touch


def geometric_xg(x_att: np.ndarray, y: np.ndarray) -> np.ndarray:
    b0, b_angle, b_dist = XG_COEF
    z = b0 + b_angle * goal_angle(x_att, y) + b_dist * distance_to_goal(x_att, y)
    return 1.0 / (1.0 + np.exp(-z))


def _att(events: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Copy of the events with every coordinate in the acting team's attacking frame."""
    d = np.array([directions[t].get(int(p), 0) for t, p in zip(events["team"], events["period"])])
    out = events.copy()
    for c in ("start_x", "start_y", "end_x", "end_y"):
        out[c] = out[c] * d  # a 180 degree rotation flips both axes
    return out


def actions(events: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Passes, carries, failed moves and shots in each team's attacking frame.

    Carries are not in Metrica's event feed; they are inferred as the ball
    displacement between consecutive touches of the same player within a
    possession chain (e.g. receive a pass, run 20 m, pass again).
    """
    ev = _att(events, directions)
    chains, _ = possession_chains(events)
    rows = []
    for c in chains:
        prev_owner, prev_pos, prev_frame = None, None, None
        for i in c.events:
            e = ev.iloc[i]
            start = (e.start_x, e.start_y)
            if prev_owner is not None and e.player == prev_owner and np.all(np.isfinite(start + prev_pos)):
                if np.hypot(start[0] - prev_pos[0], start[1] - prev_pos[1]) >= MIN_CARRY:
                    rows.append(dict(team=e.team, period=e.period, frame=prev_frame, end_frame=e.start_frame,
                                     kind="carry", player=e.player, receiver=e.player,
                                     x=prev_pos[0], y=prev_pos[1], end_x=start[0], end_y=start[1],
                                     success=True, goal=False))
            if e.type == "PASS":
                rows.append(dict(team=e.team, period=e.period, frame=e.start_frame, end_frame=e.end_frame,
                                 kind="pass", player=e.player, receiver=e.receiver,
                                 x=e.start_x, y=e.start_y, end_x=e.end_x, end_y=e.end_y,
                                 success=True, goal=False))
                prev_owner, prev_pos, prev_frame = e.receiver, (e.end_x, e.end_y), e.end_frame
                continue
            if e.type == "BALL LOST" and e.subtype not in NON_ACTIONS:
                rows.append(dict(team=e.team, period=e.period, frame=e.start_frame, end_frame=e.end_frame,
                                 kind="lost", player=e.player, receiver=None,
                                 x=e.start_x, y=e.start_y, end_x=e.end_x, end_y=e.end_y,
                                 success=False, goal=False))
            elif e.type == "SHOT":
                rows.append(dict(team=e.team, period=e.period, frame=e.start_frame, end_frame=e.end_frame,
                                 kind="shot", player=e.player, receiver=None,
                                 x=e.start_x, y=e.start_y, end_x=np.nan, end_y=np.nan, success=True,
                                 goal=e.subtype.endswith("GOAL") and "WOODWORK" not in e.subtype))
            prev_owner, prev_pos, prev_frame = e.player, start, e.start_frame
    cols = ["team", "period", "frame", "end_frame", "kind", "player", "receiver",
            "x", "y", "end_x", "end_y", "success", "goal"]
    return pd.DataFrame(rows, columns=cols)


def _cell(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Flat cell index (row-major, y slow) of attacking-frame points; -1 if outside / NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    ix = np.clip(((np.where(ok, x, 0) + HALF_L) / PITCH_LENGTH * nx).astype(int), 0, nx - 1)
    iy = np.clip(((np.where(ok, y, 0) + HALF_W) / PITCH_WIDTH * ny).astype(int), 0, ny - 1)
    return np.where(ok, iy * nx + ix, -1)


@dataclass
class ExpectedThreat:
    values: np.ndarray       # (ny, nx), attacking towards +x
    shot_prob: np.ndarray
    move_prob: np.ndarray
    xg: np.ndarray
    iterations: int

    @property
    def nx(self) -> int:
        return self.values.shape[1]

    @property
    def ny(self) -> int:
        return self.values.shape[0]

    def value(self, x_att, y) -> np.ndarray:
        """xT at attacking-frame locations (NaN where the location is unknown)."""
        idx = _cell(x_att, y, self.nx, self.ny)
        out = self.values.ravel()[np.maximum(idx, 0)]
        return np.where(idx >= 0, out, np.nan)

    @classmethod
    def fit(cls, acts: pd.DataFrame, nx: int = 12, ny: int = 8, bandwidth: float = 1.0,
            tol: float = 1e-7, max_iter: int = 500) -> "ExpectedThreat":
        """Estimate the xT surface from an `actions()` table (any number of matches)."""
        n = nx * ny
        src = _cell(acts["x"], acts["y"], nx, ny)
        dst = _cell(acts["end_x"], acts["end_y"], nx, ny)
        is_shot = (acts["kind"] == "shot").to_numpy()
        is_move = acts["kind"].isin(["pass", "carry", "lost"]).to_numpy()
        ok_move = acts["success"].to_numpy(bool) & is_move & (dst >= 0)

        shots = np.bincount(src[is_shot & (src >= 0)], minlength=n).astype(float)
        moves = np.bincount(src[is_move & (src >= 0)], minlength=n).astype(float)
        trans = np.zeros((n, n))
        np.add.at(trans, (src[ok_move & (src >= 0)], dst[ok_move & (src >= 0)]), 1.0)

        # Smooth over neighbouring *source* cells: a sparse cell borrows the
        # behaviour of its neighbours, weighted by a Gaussian in cell units.
        if bandwidth > 0:
            gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
            c = np.c_[gx.ravel(), gy.ravel()]
            w = np.exp(-np.sum((c[:, None] - c[None]) ** 2, axis=2) / (2 * bandwidth ** 2))
            shots, moves, trans = w @ shots, w @ moves, w @ trans

        total = shots + moves
        with np.errstate(invalid="ignore", divide="ignore"):
            shot_prob = np.where(total > 0, shots / total, 0.0)
            move_prob = np.where(total > 0, moves / total, 0.0)
            T = np.where(moves[:, None] > 0, trans / moves[:, None], 0.0)

        xs = -HALF_L + PITCH_LENGTH / nx * (np.arange(nx) + 0.5)
        ys = -HALF_W + PITCH_WIDTH / ny * (np.arange(ny) + 0.5)
        gx, gy = np.meshgrid(xs, ys)
        xg = geometric_xg(gx.ravel(), gy.ravel())

        xt = np.zeros(n)
        for it in range(1, max_iter + 1):
            new = shot_prob * xg + move_prob * (T @ xt)
            done = np.max(np.abs(new - xt)) < tol
            xt = new
            if done:
                break
        shape = (ny, nx)
        return cls(xt.reshape(shape), shot_prob.reshape(shape), move_prob.reshape(shape),
                   xg.reshape(shape), it)


def is_progressive(x: np.ndarray, y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> np.ndarray:
    """Wyscout's progressive-pass rule on attacking-frame coordinates."""
    gain = distance_to_goal(x, y) - distance_to_goal(end_x, end_y)
    x, end_x = np.asarray(x), np.asarray(end_x)
    need = np.where((x < 0) & (end_x < 0), 30.0, np.where((x < 0) & (end_x >= 0), 15.0, 10.0))
    return np.nan_to_num(gain, nan=-1.0) >= need


def defensive_lines(x_att: np.ndarray, n_lines: int = 3) -> np.ndarray:
    """Positions (attacking-frame x, ascending) of the opponent's lines.

    `x_att` holds the opponent outfield players' x in the *passer's*
    attacking frame. Players are sorted and cut at the `n_lines - 1` largest
    gaps; each line sits at the mean x of its players.
    """
    x = np.sort(np.asarray(x_att, float)[np.isfinite(x_att)])
    if len(x) == 0:
        return np.array([])
    k = min(n_lines, len(x))
    cuts = np.sort(np.argsort(np.diff(x))[::-1][: k - 1]) + 1 if k > 1 else np.array([], dtype=int)
    return np.array([g.mean() for g in np.split(x, cuts)])


LINE_MARGIN = 2.0  # m the ball must start behind / arrive beyond a line


def lines_broken(start_x: float, end_x: float, lines: np.ndarray, margin: float = LINE_MARGIN) -> int:
    """Number of lines the ball starts at least `margin` behind and arrives at least `margin` beyond."""
    return int(np.sum((lines > start_x + margin) & (lines < end_x - margin)))


def packing(opp_x_start: np.ndarray, ball_x_start: float, opp_x_end: np.ndarray, ball_x_end: float) -> int:
    """Opponents taken out by the pass (never negative)."""
    before = np.sum(np.asarray(opp_x_start) > ball_x_start)
    after = np.sum(np.asarray(opp_x_end) > ball_x_end)
    return int(max(before - after, 0))


def _opponents_x(track: pd.DataFrame, frame: int, team: str, direction: int) -> np.ndarray | None:
    if frame not in track.index:
        return None
    row = track.loc[frame]
    stems = player_ids(track, other(team))
    x = row[[f"{s}_x" for s in stems]].to_numpy(float) * direction
    return x[np.isfinite(x)]


def pass_table(events: pd.DataFrame, directions: dict, xt: ExpectedThreat,
               track: pd.DataFrame | None = None, acts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Completed passes and carries with progression, xT added, lines broken and packing."""
    acts = actions(events, directions) if acts is None else acts
    df = acts[acts["kind"].isin(["pass", "carry"])].copy().reset_index(drop=True)
    df["metres_gained"] = df["end_x"] - df["x"]
    df["distance"] = np.hypot(df["end_x"] - df["x"], df["end_y"] - df["y"])
    df["progressive"] = is_progressive(df["x"], df["y"], df["end_x"], df["end_y"])
    df["xt_start"] = xt.value(df["x"], df["y"])
    df["xt_end"] = xt.value(df["end_x"], df["end_y"])
    df["xt_added"] = df["xt_end"] - df["xt_start"]
    df["lines_broken"] = 0
    df["packing"] = 0
    if track is not None:
        lb, pk = [], []
        for r in df.itertuples():
            d = directions[r.team].get(int(r.period), 0)
            opp = _opponents_x(track, int(r.frame), r.team, d)
            opp_end = _opponents_x(track, int(r.end_frame), r.team, d)
            if r.kind != "pass" or opp is None or opp_end is None or len(opp) < 2:
                lb.append(0)
                pk.append(0)
                continue
            outfield = np.delete(opp, np.argmax(opp))  # deepest opponent = goalkeeper
            lb.append(lines_broken(r.x, r.end_x, defensive_lines(outfield)))
            pk.append(packing(opp, r.x, opp_end, r.end_x))
        df["lines_broken"], df["packing"] = lb, pk
    df["line_breaking"] = df["lines_broken"] > 0
    return df


def player_threat(passes: pd.DataFrame) -> pd.DataFrame:
    """Per-player progression leaderboard."""
    p = passes.assign(pos_xt=passes["xt_added"].clip(lower=0))
    g = p.groupby(["team", "player"])
    out = pd.DataFrame({
        "xt": g["pos_xt"].sum(),
        "xt_passes": p[p["kind"] == "pass"].groupby(["team", "player"])["pos_xt"].sum(),
        "xt_carries": p[p["kind"] == "carry"].groupby(["team", "player"])["pos_xt"].sum(),
        "progressive": p[p["kind"] == "pass"].groupby(["team", "player"])["progressive"].sum(),
        "line_breaking": p[p["kind"] == "pass"].groupby(["team", "player"])["line_breaking"].sum(),
        "packing": g["packing"].sum(),
    }).fillna(0)
    return out.sort_values("xt", ascending=False).reset_index()


def threat_summary(passes: pd.DataFrame, acts: pd.DataFrame) -> dict:
    out = {}
    for team in TEAMS:
        p = passes[passes["team"] == team]
        pp = p[p["kind"] == "pass"]
        shots = acts[(acts["team"] == team) & (acts["kind"] == "shot")]
        out[team] = {
            "xt_generated": float(p["xt_added"].clip(lower=0).sum()),
            "xt_passes": float(pp["xt_added"].clip(lower=0).sum()),
            "xt_carries": float(p[p["kind"] == "carry"]["xt_added"].clip(lower=0).sum()),
            "passes": int(len(pp)),
            "progressive_passes": int(pp["progressive"].sum()),
            "line_breaking_passes": int(pp["line_breaking"].sum()),
            "lines_broken": int(pp["lines_broken"].sum()),
            "packing": int(pp["packing"].sum()),
            "metres_gained": float(p["metres_gained"].clip(lower=0).sum()),
            "shots": int(len(shots)),
            "goals": int(shots["goal"].sum()),
            "xg": float(geometric_xg(shots["x"].to_numpy(), shots["y"].to_numpy()).sum()) if len(shots) else 0.0,
        }
    return out
