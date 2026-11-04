"""Team shape: compactness, block dimensions and formation, in and out of possession.

**Per frame** (outfield players only, the goalkeeper is excluded):

* convex-hull area (m^2) of the ten outfield players (Andrew's monotone
  chain + shoelace formula);
* width (spread across the pitch) and depth (spread along it), in m;
* centroid height: distance of the outfield centroid from the team's own
  goal line (high = a team pushed up).

Each frame is tagged *in possession* / *out of possession* from the event
labels, so the summary separates the attacking shape from the defensive block.

**Formation detection.** For each phase, every outfield player's position
relative to the team centroid is averaged over the phase (in the attacking
frame, so both halves fold together) and the ten most present players are
kept. Both this average shape and every template formation (4-4-2, 4-3-3,
4-2-3-1, ...) are standardised (zero mean, unit spread along each axis), and
each template is matched to the shape with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) on squared distances. The template
with the lowest mean cost wins, and the assignment gives every player a role
(e.g. D-L = left back, M-C = central midfielder, F-C = striker).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .data import FPS, player_ids
from .pitch import HALF_L, TEAMS, direction_per_frame

FORMATIONS = ("4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1", "4-4-1-1", "4-3-1-2", "4-5-1",
              "3-5-2", "3-4-3", "3-4-2-1", "5-3-2", "5-4-1")


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Vertices of the convex hull, counter-clockwise (Andrew's monotone chain)."""
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def goalkeepers(track: pd.DataFrame, directions: dict) -> dict[str, set[str]]:
    """Per team, the player who stands deepest (towards his own goal) in each period."""
    out = {}
    for team in TEAMS:
        d = direction_per_frame(track, team, directions)
        keepers = set()
        for p in np.unique(track["period"].dropna()):
            rows = track["period"].to_numpy() == p
            best, best_x = None, np.inf
            for pid in player_ids(track, team):
                x = track[f"{pid}_x"].to_numpy()[rows] * d[rows]
                if np.isfinite(x).sum() < 0.25 * rows.sum():
                    continue
                m = np.nanmean(x)
                if m < best_x:
                    best, best_x = pid, m
            if best:
                keepers.add(best)
        out[team] = keepers
    return out


def shape_metrics(pts: np.ndarray) -> dict:
    """Hull area, width, depth and centroid of outfield points in the attacking frame."""
    if len(pts) < 3:
        return {"area": np.nan, "width": np.nan, "depth": np.nan, "centroid_x": np.nan}
    return {"area": polygon_area(convex_hull(pts)),
            "width": float(np.ptp(pts[:, 1])), "depth": float(np.ptp(pts[:, 0])),
            "centroid_x": float(pts[:, 0].mean())}


def shape_series(track: pd.DataFrame, possession: np.ndarray, directions: dict, step: int = 10,
                 keepers: dict | None = None) -> pd.DataFrame:
    """Shape of both teams every `step` frames (attacking frame, keepers excluded)."""
    keepers = keepers or goalkeepers(track, directions)
    sample = np.arange(0, len(track), step)
    parts = []
    for team, code in (("home", 1), ("away", -1)):
        ids = [p for p in player_ids(track, team) if p not in keepers[team]]
        d = direction_per_frame(track, team, directions)[sample].astype(float)
        X = track[[f"{p}_x" for p in ids]].to_numpy()[sample] * d[:, None]
        Y = track[[f"{p}_y" for p in ids]].to_numpy()[sample] * d[:, None]
        ok = np.isfinite(X) & np.isfinite(Y)
        valid = (ok.sum(axis=1) >= 3) & (d != 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are dropped below
            width = np.nanmax(Y, axis=1) - np.nanmin(Y, axis=1)
            depth = np.nanmax(X, axis=1) - np.nanmin(X, axis=1)
            cx = np.nanmean(X, axis=1)
        area = np.full(len(sample), np.nan)
        for k in np.flatnonzero(valid):
            area[k] = polygon_area(convex_hull(np.c_[X[k, ok[k]], Y[k, ok[k]]]))
        phase = np.where(possession[sample] == code, "in", np.where(possession[sample] == -code, "out", "dead"))
        parts.append(pd.DataFrame({"frame": track.index[sample], "team": team, "phase": phase, "area": area,
                                   "width": width, "depth": depth, "centroid_x": cx,
                                   "height": cx + HALF_L})[valid])
    return pd.concat(parts, ignore_index=True)


def shape_summary(series: pd.DataFrame) -> dict:
    out = {}
    for team in TEAMS:
        out[team] = {}
        for phase in ("in", "out"):
            s = series[(series["team"] == team) & (series["phase"] == phase)]
            out[team][phase] = {k: float(s[k].mean()) if len(s) else None
                                for k in ("area", "width", "depth", "height")}
    return out


def shape_timeline(series: pd.DataFrame) -> list[dict]:
    """Per-minute mean hull area of each team, split by phase."""
    s = series.assign(minute=((series["frame"] - 1) / FPS // 60).astype(int))
    out = []
    for m, g in s.groupby("minute"):
        row = {"minute": int(m)}
        for team in TEAMS:
            for phase in ("in", "out"):
                v = g[(g["team"] == team) & (g["phase"] == phase)]["area"]
                row[f"{team}_{phase}"] = float(v.mean()) if len(v) else None
        out.append(row)
    return out


# ---------- formations ----------

def _line_labels(n_lines: int) -> list[str]:
    return {3: ["D", "M", "F"], 4: ["D", "DM", "AM", "F"], 5: ["D", "DM", "M", "AM", "F"]}[n_lines]


def _side_labels(n: int) -> list[str]:
    # y > 0 is the left touchline when attacking towards +x.
    return {1: ["C"], 2: ["LC", "RC"], 3: ["L", "C", "R"], 4: ["L", "LC", "RC", "R"],
            5: ["L", "LC", "C", "RC", "R"]}[n]


def template(formation: str) -> tuple[np.ndarray, list[str]]:
    """Canonical outfield positions (attacking frame, x = depth, y = width) and role names."""
    lines = [int(c) for c in formation.split("-")]
    xs = np.linspace(-1.0, 1.0, len(lines))
    pts, roles = [], []
    for line, (n, x) in enumerate(zip(lines, xs)):
        half = min(0.9, 0.3 * (n - 1))
        ys = np.linspace(half, -half, n) if n > 1 else np.array([0.0])
        for y, side in zip(ys, _side_labels(n)):
            pts.append((x, y))
            roles.append(f"{_line_labels(len(lines))[line]}-{side}")
    return np.array(pts), roles


def _standardise(p: np.ndarray) -> np.ndarray:
    c = p - p.mean(axis=0)
    sd = c.std(axis=0)
    return c / np.where(sd > 0, sd, 1.0)


def match_formation(shape: np.ndarray, candidates=FORMATIONS) -> dict:
    """Best template for an average outfield shape (n x 2, attacking frame)."""
    obs = _standardise(np.asarray(shape, float))
    results = []
    for f in candidates:
        tpl, roles = template(f)
        t = _standardise(tpl)
        cost = ((obs[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
        r, c = linear_sum_assignment(cost)
        results.append((float(cost[r, c].mean()), f, [roles[j] for j in c], r))
    results.sort(key=lambda z: z[0])
    best_cost, best, roles, rows = results[0]
    assigned = [None] * len(obs)
    for i, role in zip(rows, roles):
        assigned[i] = role
    return {"formation": best, "cost": best_cost, "roles": assigned,
            "ranking": [{"formation": f, "cost": round(c, 4)} for c, f, _, _ in results[:4]]}


def average_shape(track: pd.DataFrame, possession: np.ndarray, directions: dict, team: str, phase: str,
                  keepers: dict | None = None, step: int = 5, frames: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    """Average centroid-relative outfield positions of the ten most present players in a phase."""
    keepers = keepers or goalkeepers(track, directions)
    code = 1 if team == "home" else -1
    want = code if phase == "in" else -code
    sample = np.arange(0, len(track), step)
    sel = sample[possession[sample] == want]
    if frames is not None:
        sel = np.intersect1d(sel, frames)
    ids = [p for p in player_ids(track, team) if p not in keepers[team]]
    d = direction_per_frame(track, team, directions)[sel].astype(float)
    X = track[[f"{p}_x" for p in ids]].to_numpy()[sel] * d[:, None]
    Y = track[[f"{p}_y" for p in ids]].to_numpy()[sel] * d[:, None]
    cx, cy = np.nanmean(X, axis=1, keepdims=True), np.nanmean(Y, axis=1, keepdims=True)
    RX, RY = X - cx, Y - cy
    present = np.isfinite(RX).sum(axis=0)
    top = np.argsort(-present)[:10]
    top = [i for i in top if present[i] > 0]
    pts = np.c_[np.nanmean(RX[:, top], axis=0), np.nanmean(RY[:, top], axis=0)]
    return [ids[i] for i in top], pts


def formations(track: pd.DataFrame, possession: np.ndarray, directions: dict, keepers: dict | None = None) -> dict:
    """Detected formation and roles per team and phase, plus per half."""
    keepers = keepers or goalkeepers(track, directions)
    out = {}
    period = track["period"].to_numpy()
    for team in TEAMS:
        out[team] = {}
        for phase in ("in", "out"):
            ids, pts = average_shape(track, possession, directions, team, phase, keepers)
            if len(ids) < 7:
                out[team][phase] = None
                continue
            m = match_formation(pts)
            halves = {}
            for p in (1, 2):
                hid, hpts = average_shape(track, possession, directions, team, phase, keepers,
                                          frames=np.flatnonzero(period == p))
                if len(hid) >= 7:
                    halves[str(p)] = match_formation(hpts)["formation"]
            out[team][phase] = {
                "formation": m["formation"], "cost": m["cost"], "ranking": m["ranking"],
                "players": [{"id": pid, "role": role, "x": float(x), "y": float(y)}
                            for pid, role, (x, y) in zip(ids, m["roles"], pts)],
                "by_half": halves,
            }
        out[team]["keepers"] = sorted(keepers[team])
    return out


def roles(formation_result: dict, team: str) -> dict[str, str]:
    """Player id -> role (out-of-possession shape preferred; keepers are 'GK')."""
    out = {k: "GK" for k in formation_result[team]["keepers"]}
    for phase in ("in", "out"):
        res = formation_result[team].get(phase)
        if res:
            for p in res["players"]:
                out.setdefault(p["id"], p["role"])
    return out


