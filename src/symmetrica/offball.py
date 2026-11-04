"""High-intensity off-ball runs and the space they create.

**Detection.** A run is a spell of at least `min_duration` (1.5 s) in which a
player moves faster than `min_speed` (5 m/s) while a *teammate* has the
ball (event-derived possession, and the runner is never within 2 m of the
ball, so the carrier's own dribbles are excluded). Dips shorter than 0.2 s
are bridged so a run is not split by tracking noise.

**Classification**, in the runner's attacking frame, first rule that fits:

1. *Behind defensive line*: the run starts goal-side of the opponent's
   deepest outfield player and ends beyond him, moving at least 5 m forward.
2. *Half-space underlap / overlap*: the runner starts behind the ball
   carrier and finishes ahead of him on the same flank; outside of the
   carrier (further from the centre line) is an overlap, inside an underlap.
3. *Dragging defender / space creation*: the opponent nearest the runner
   at the start (within 8 m) follows him (moves >= 5 m and is still within
   8 m at the end), or the run creates measurable space (below).
4. *Support run*: everything else.

**Space creation** is measured counterfactually at the end of the run:
pitch control is recomputed as if the runner (and the defender he dragged)
had stayed where they were when the run started. Outside an 8 m radius
around the runner (space he occupies himself does not count):

* `space_m2`: the attacking team's extra pitch control, in m^2;
* `space_epv`: how much the most valuable space available to teammates
  improved, max EPV(actual) - max EPV(counterfactual), with EPV = PC * xT.
  It reads as a change in goal probability of the best spot to play into.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FPS, player_ids
from .kinematics import frame_state
from .pitch import TEAMS, Grid, other
from .pitch_control import PitchControlParams, pitch_control_at
from .threat import ExpectedThreat

CATEGORIES = ("Behind defensive line", "Half-space underlap/overlap",
              "Dragging defender / space creation", "Support run")
SPACE_GRID = Grid(21, 14)


@dataclass(frozen=True)
class RunParams:
    min_speed: float = 5.0        # m/s
    min_duration: float = 1.5     # s
    carrier_radius: float = 2.0   # m; closer than this to the ball = on the ball
    bridge: int = 5               # frames of slower running that do not end a run
    marker_radius: float = 8.0    # m; nearest opponent this close is the runner's marker
    space_radius: float = 8.0     # m; control around the runner himself is not "created" for others
    min_space_m2: float = 100.0   # m^2 of control (a 10 x 10 m pocket) that counts as space creation


def _segments(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """(start, end) index pairs (end exclusive) of True runs, bridging gaps of <= bridge frames."""
    m = mask.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    segs = list(zip(edges[::2], edges[1::2]))
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= bridge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_runs(track: pd.DataFrame, vel: pd.DataFrame, possession: np.ndarray,
                params: RunParams = RunParams()) -> pd.DataFrame:
    """Raw high-intensity off-ball runs (row positions), not yet classified."""
    bx, by = track["ball_x"].to_numpy(), track["ball_y"].to_numpy()
    period = track["period"].to_numpy()
    rows = []
    for team, code in (("home", 1), ("away", -1)):
        with_ball = possession == code
        for pid in player_ids(track, team):
            x, y = track[f"{pid}_x"].to_numpy(), track[f"{pid}_y"].to_numpy()
            speed = np.hypot(vel[f"{pid}_vx"].to_numpy(), vel[f"{pid}_vy"].to_numpy())
            with np.errstate(invalid="ignore"):
                far = np.hypot(x - bx, y - by) > params.carrier_radius
                fast = speed > params.min_speed
            ok = np.nan_to_num(fast) & with_ball & np.nan_to_num(far)
            for s, e in _segments(ok, params.bridge):
                # A bridged gap must not hide a moment on the ball or a change of possession.
                if (e - s) < params.min_duration * FPS or not with_ball[s:e].all() or not far[s:e].all():
                    continue
                if period[s] != period[e - 1]:
                    continue
                seg_speed = speed[s:e]
                rows.append({
                    "team": team, "player": pid, "start": int(s), "end": int(e - 1),
                    "frame": int(track.index[s]), "end_frame": int(track.index[e - 1]),
                    "period": int(period[s]), "duration": (e - s) / FPS,
                    "distance": float(np.nansum(seg_speed) / FPS), "max_speed": float(np.nanmax(seg_speed)),
                    "x0": float(x[s]), "y0": float(y[s]), "x1": float(x[e - 1]), "y1": float(y[e - 1]),
                })
    return pd.DataFrame(rows, columns=["team", "player", "start", "end", "frame", "end_frame", "period",
                                       "duration", "distance", "max_speed", "x0", "y0", "x1", "y1"])


def _positions(track: pd.DataFrame, pos: int, team: str, direction: int) -> dict[str, np.ndarray]:
    row = track.iloc[pos]
    out = {}
    for pid in player_ids(track, team):
        x, y = row[f"{pid}_x"], row[f"{pid}_y"]
        if np.isfinite(x) and np.isfinite(y):
            out[pid] = np.array([x, y]) * direction
    return out


def classify_run(track: pd.DataFrame, run: dict, direction: int, params: RunParams = RunParams()) -> dict:
    """Category, subtype, ball carrier and dragged marker of one run (attacking-frame geometry)."""
    team, pid = run["team"], run["player"]
    s, e = run["start"], run["end"]
    r0 = np.array([run["x0"], run["y0"]]) * direction
    r1 = np.array([run["x1"], run["y1"]]) * direction
    opp0, opp1 = _positions(track, s, other(team), direction), _positions(track, e, other(team), direction)
    mates1 = {k: v for k, v in _positions(track, e, team, direction).items() if k != pid}
    mates0 = {k: v for k, v in _positions(track, s, team, direction).items() if k != pid}
    ball1 = np.array([track["ball_x"].iloc[e], track["ball_y"].iloc[e]]) * direction

    out = {"category": CATEGORIES[3], "subtype": "", "carrier": None, "marker": None}
    # Deepest opponent outfield player (the keeper is the deepest of all).
    def last_line(opp):
        xs = sorted(v[0] for v in opp.values())
        return xs[-2] if len(xs) >= 2 else (xs[-1] if xs else np.inf)

    line0, line1 = last_line(opp0), last_line(opp1)
    carrier = min(mates1, key=lambda k: np.linalg.norm(mates1[k] - ball1)) if mates1 else None
    out["carrier"] = carrier

    marker = None
    if opp0:
        near = min(opp0, key=lambda k: np.linalg.norm(opp0[k] - r0))
        if near in opp1 and np.linalg.norm(opp0[near] - r0) <= params.marker_radius:
            moved = np.linalg.norm(opp1[near] - opp0[near])
            if moved >= 5.0 and np.linalg.norm(opp1[near] - r1) <= params.marker_radius:
                marker = near
    out["marker"] = marker

    if r0[0] < line0 and r1[0] > line1 and r1[0] - r0[0] >= 5.0:
        out["category"] = CATEGORIES[0]
    elif carrier is not None and carrier in mates0:
        c0, c1 = mates0[carrier], mates1[carrier]
        same_flank = np.sign(r1[1]) == np.sign(c1[1]) or abs(r1[1] - c1[1]) < 10
        if r0[0] < c0[0] and r1[0] > c1[0] and same_flank:
            out["category"] = CATEGORIES[1]
            out["subtype"] = "overlap" if abs(r1[1]) > abs(c1[1]) else "underlap"
    if out["category"] == CATEGORIES[3] and marker is not None:
        out["category"] = CATEGORIES[2]
        out["subtype"] = "dragged marker"
    return out


def space_created(track: pd.DataFrame, vel: pd.DataFrame, run: dict, marker: str | None, direction: int,
                  xt: ExpectedThreat | None, params: RunParams = RunParams(),
                  pc_params: PitchControlParams = PitchControlParams(), grid: Grid = SPACE_GRID) -> tuple[float, float]:
    """(m^2 of pitch control, EPV) the run opened for teammates, by counterfactual at the run's end."""
    team, pid = run["team"], run["player"]
    st_end = frame_state(track, vel, run["end"])
    st_start = frame_state(track, vel, run["start"])
    if not np.all(np.isfinite(st_end.ball)):
        return 0.0, 0.0
    ids, pos, v = st_end.team(team)
    oids, opos, ov = st_end.team(other(team))
    cf_pos, cf_v, cf_opos, cf_ov = pos.copy(), v.copy(), opos.copy(), ov.copy()
    sids, spos, _ = st_start.team(team)
    soids, sopos, _ = st_start.team(other(team))
    if pid not in ids or pid not in sids:
        return 0.0, 0.0
    i = ids.index(pid)
    cf_pos[i], cf_v[i] = spos[sids.index(pid)], 0.0
    if marker is not None and marker in oids and marker in soids:
        j = oids.index(marker)
        cf_opos[j], cf_ov[j] = sopos[soids.index(marker)], 0.0

    cells = grid.cells
    keep = np.linalg.norm(cells - pos[i], axis=1) > params.space_radius
    actual, _ = pitch_control_at(cells, st_end.ball, pos, v, opos, ov, pc_params)
    counter, _ = pitch_control_at(cells, st_end.ball, cf_pos, cf_v, cf_opos, cf_ov, pc_params)
    area = float(((actual - counter) * keep).sum() * grid.cell_area)
    if xt is None or not keep.any():
        return area, 0.0
    w = xt.interpolate(cells[:, 0] * direction, cells[:, 1] * direction)
    best = lambda pc: float(np.nanmax(np.where(keep, pc * w, -np.inf)))  # noqa: E731
    return area, best(actual) - best(counter)


def off_ball_runs(track: pd.DataFrame, vel: pd.DataFrame, possession: np.ndarray, directions: dict,
                  xt: ExpectedThreat | None = None, params: RunParams = RunParams(),
                  pc_params: PitchControlParams = PitchControlParams()) -> pd.DataFrame:
    """Detected, classified runs with the space each one created."""
    runs = detect_runs(track, vel, possession, params)
    extra = []
    for r in runs.to_dict(orient="records"):
        d = directions[r["team"]].get(r["period"], 0)
        c = classify_run(track, r, d, params)
        area, epv = space_created(track, vel, r, c["marker"], d, xt, params, pc_params)
        if c["category"] == CATEGORIES[3] and area >= params.min_space_m2:
            c["category"], c["subtype"] = CATEGORIES[2], "space creation"
        extra.append({**c, "space_m2": area, "space_epv": epv,
                      "forward_m": (r["x1"] - r["x0"]) * d})
    cols = ["category", "subtype", "carrier", "marker", "space_m2", "space_epv", "forward_m"]
    return pd.concat([runs.reset_index(drop=True), pd.DataFrame(extra, columns=cols)], axis=1)


def runs_summary(runs: pd.DataFrame) -> dict:
    out = {}
    for team in TEAMS:
        r = runs[runs["team"] == team]
        out[team] = {
            "runs": int(len(r)),
            "by_category": {c: int((r["category"] == c).sum()) for c in CATEGORIES},
            "overlaps": int((r["subtype"] == "overlap").sum()),
            "underlaps": int((r["subtype"] == "underlap").sum()),
            "space_m2": float(r["space_m2"].clip(lower=0).sum()),
            "space_epv": float(r["space_epv"].clip(lower=0).sum()),
            "mean_space_m2": float(r["space_m2"].mean()) if len(r) else 0.0,
            "distance_m": float(r["distance"].sum()),
        }
    return out
