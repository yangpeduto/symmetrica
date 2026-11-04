"""Player scouting profiles: physical load from tracking, tactical output from events.

Physical (from smoothed tracking velocities; implausible speeds are already
discarded by `kinematics.velocities`):

* total distance (km), high-speed running distance (> 5.5 m/s), sprint
  distance (> 7.0 m/s), number of sprints (>= 1 s above 7 m/s), all on a
  0.4 s rolling mean of speed so single-frame tracking spikes do not count;
* top speed: the highest speed held for a full second (km/h).

Tactical:

* xT added from passes and carries, progressive and line-breaking passes,
  packing (opponents bypassed);
* pressing: *pressures* from tracking (spells of >= 0.4 s within 4 m of the
  ball while the opponent has it, at least 2 s apart) plus defensive actions
  from events (challenges, interceptions, tackles);
* off-ball runs by category, and pass decision quality where available.

Counting stats are also given per 90 minutes. Percentiles rank a player
against a reference pool (by default every player in the match with at
least 30 minutes; `scripts/report.py` pools both sample matches).
"""
import numpy as np
import pandas as pd

from .data import FPS, player_ids
from .pitch import HALF_L, HALF_W, TEAMS, direction_per_frame
from .pressing import DEFENSIVE_RECOVERY, possession_chains

HSR_SPEED = 5.5
SPRINT_SPEED = 7.0
HEATMAP = (12, 8)

# Metrics shown as percentiles, with direction (higher is better unless noted) and label.
PERCENTILE_METRICS = [
    ("distance_p90", "Distance / 90"), ("hsr_p90", "High-speed running / 90"),
    ("sprint_p90", "Sprint distance / 90"), ("top_speed_kmh", "Top speed"),
    ("xt_p90", "xT added / 90"), ("progressive_p90", "Progressive passes / 90"),
    ("line_breaking_p90", "Line-breaking passes / 90"), ("packing_p90", "Packing / 90"),
    ("pressures_p90", "Pressures / 90"), ("defensive_actions_p90", "Defensive actions / 90"),
    ("runs_p90", "Off-ball runs / 90"),
]


def _spells(mask: np.ndarray, min_len: int, min_gap: int) -> int:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    count, last_end = 0, -10 ** 9
    for s, e in zip(starts, ends):
        if e - s >= min_len and s - last_end >= min_gap:
            count += 1
            last_end = e
    return count


def physical(track: pd.DataFrame, vel: pd.DataFrame) -> pd.DataFrame:
    """Distance, high-speed running, sprints and top speed per player."""
    rows = []
    for team in TEAMS:
        for pid in player_ids(track, team):
            on = np.isfinite(track[f"{pid}_x"].to_numpy())
            if not on.any():
                continue
            raw = np.hypot(vel[f"{pid}_vx"].to_numpy(), vel[f"{pid}_vy"].to_numpy())
            raw = pd.Series(np.where(on, raw, np.nan))
            # Speed bands on a 0.4 s mean so single-frame tracking spikes cannot count as sprints;
            # top speed is the best speed held for a full second.
            v = np.nan_to_num(raw.rolling(10, center=True, min_periods=1).mean().to_numpy())
            peak = raw.rolling(FPS, min_periods=FPS).mean().max()
            rows.append({
                "id": pid, "team": team, "minutes": on.sum() / FPS / 60,
                "distance_km": float(v.sum() / FPS / 1000),
                "hsr_m": float(v[v > HSR_SPEED].sum() / FPS),
                "sprint_m": float(v[v > SPRINT_SPEED].sum() / FPS),
                "sprints": _spells(v > SPRINT_SPEED, FPS, FPS),
                "top_speed_kmh": float(peak * 3.6) if np.isfinite(peak) else 0.0,
            })
    return pd.DataFrame(rows)


def pressures(track: pd.DataFrame, possession: np.ndarray, radius: float = 4.0,
              min_seconds: float = 0.4, gap_seconds: float = 2.0) -> dict[str, int]:
    """Tracking-based pressing spells per player (close to the ball while the opponent has it)."""
    bx, by = track["ball_x"].to_numpy(), track["ball_y"].to_numpy()
    out = {}
    for team, defending in (("home", -1), ("away", 1)):
        mask_team = possession == defending
        for pid in player_ids(track, team):
            with np.errstate(invalid="ignore"):
                close = np.hypot(track[f"{pid}_x"].to_numpy() - bx, track[f"{pid}_y"].to_numpy() - by) < radius
            out[pid] = _spells(np.nan_to_num(close) & mask_team, int(min_seconds * FPS), int(gap_seconds * FPS))
    return out


def defensive_actions(events: pd.DataFrame) -> dict[str, int]:
    """Ground challenges, tackles and interceptions made while the opponent had the ball."""
    _, before = possession_chains(events)
    ev = events.assign(before=before)
    opp = ev["team"].map({"home": "away", "away": "home"})
    mask = (ev["before"] == opp) & (
        ((ev["type"] == "CHALLENGE") & ~ev["subtype"].str.startswith("AERIAL"))
        | ((ev["type"] == "RECOVERY") & ev["subtype"].isin(DEFENSIVE_RECOVERY)))
    return ev[mask]["player"].value_counts().to_dict()


def heatmap(track: pd.DataFrame, pid: str, team: str, directions: dict, step: int = 5) -> list[list[float]]:
    """Share of time in each zone (attacking frame, 12 x 8), rows = y from bottom."""
    nx, ny = HEATMAP
    d = direction_per_frame(track, team, directions)[::step].astype(float)
    x = track[f"{pid}_x"].to_numpy()[::step] * d
    y = track[f"{pid}_y"].to_numpy()[::step] * d
    ok = np.isfinite(x) & np.isfinite(y) & (d != 0)
    h, _, _ = np.histogram2d(y[ok], x[ok], bins=[ny, nx], range=[[-HALF_W, HALF_W], [-HALF_L, HALF_L]])
    total = h.sum()
    return np.round(h / total, 4).tolist() if total else h.tolist()


def player_profiles(track: pd.DataFrame, vel: pd.DataFrame, events: pd.DataFrame, possession: np.ndarray,
                    directions: dict, passes: pd.DataFrame, runs: pd.DataFrame | None = None,
                    decisions: pd.DataFrame | None = None, roles: dict | None = None,
                    min_minutes: float = 1.0) -> list[dict]:
    """One profile per player who played at least `min_minutes`."""
    phys = physical(track, vel)
    press = pressures(track, possession)
    dact = defensive_actions(events)
    roles = roles or {}
    out = []
    for r in phys.itertuples(index=False):
        if r.minutes < min_minutes:
            continue
        p = passes[passes["player"] == r.id]
        pp = p[p["kind"] == "pass"]
        prog = pp[pp["progressive"] | pp["line_breaking"]]
        my_runs = runs[runs["player"] == r.id] if runs is not None and len(runs) else pd.DataFrame()
        dec = decisions[decisions["player"] == r.id] if decisions is not None and len(decisions) else pd.DataFrame()
        prof = {
            "id": r.id, "team": r.team, "number": int(r.id.split("_")[1]), "role": roles.get(r.id, ""),
            "minutes": round(float(r.minutes), 1),
            "distance_km": round(r.distance_km, 2), "hsr_m": round(r.hsr_m), "sprint_m": round(r.sprint_m),
            "sprints": int(r.sprints), "top_speed_kmh": round(r.top_speed_kmh, 1),
            "xt": float(p["xt_added"].clip(lower=0).sum()),
            "xt_passes": float(pp["xt_added"].clip(lower=0).sum()),
            "xt_carries": float(p[p["kind"] == "carry"]["xt_added"].clip(lower=0).sum()),
            "passes": int(len(pp)),
            "progressive": int(pp["progressive"].sum()), "line_breaking": int(pp["line_breaking"].sum()),
            "packing": int(pp["packing"].sum()),
            "pressures": int(press.get(r.id, 0)), "defensive_actions": int(dact.get(r.id, 0)),
            "runs": int(len(my_runs)),
            "runs_behind": int((my_runs["category"] == "Behind defensive line").sum()) if len(my_runs) else 0,
            "runs_by_category": my_runs["category"].value_counts().to_dict() if len(my_runs) else {},
            "best_option_share": float(dec["chose_best"].mean()) if len(dec) else None,
            "heatmap": heatmap(track, r.id, r.team, directions),
            "pass_map": prog[["x", "y", "end_x", "end_y", "progressive", "line_breaking", "xt_added"]]
            .round(2).to_dict(orient="records"),
        }
        per90 = 90.0 / max(prof["minutes"], 1e-9)
        prof["distance_p90"] = prof["distance_km"] * per90
        prof["hsr_p90"] = prof["hsr_m"] * per90
        prof["sprint_p90"] = prof["sprint_m"] * per90
        for k in ("xt", "progressive", "line_breaking", "packing", "pressures", "defensive_actions", "runs"):
            prof[f"{k}_p90"] = prof[k] * per90
        out.append(prof)
    return out


def add_percentiles(players: list[dict], reference: list[dict] | None = None, min_minutes: float = 30.0) -> None:
    """Attach `pct` (0-100) for each PERCENTILE_METRICS key, ranked against outfield players of `reference`.

    Players below `min_minutes` get percentiles too (against the same pool)
    but are flagged with `small_sample`. Goalkeepers are not in the pool.
    """
    ref = [p for p in (reference or players) if p["minutes"] >= min_minutes and p.get("role") != "GK"]
    for key, _ in PERCENTILE_METRICS:
        pool = np.array([p[key] for p in ref], dtype=float)
        for p in players:
            v = float(p[key])
            if len(pool) == 0:
                p.setdefault("pct", {})[key] = None
                continue
            pct = 100.0 * ((pool < v).sum() + 0.5 * (pool == v).sum()) / len(pool)
            p.setdefault("pct", {})[key] = round(pct, 1)
    for p in players:
        p["small_sample"] = p["minutes"] < min_minutes
        p["pool_size"] = len(ref)
