"""Pressing intensity, counter-pressing and where turnovers are won.

Event-based metrics work on *possession chains*: maximal runs of on-ball
events by one team with no dead ball in between. A chain that follows an
opponent chain without a stoppage is a **turnover in open play**.

* **PPDA** (passes allowed per defensive action). For pressing team T,

      PPDA_T = opponent passes in zone / defensive actions of T in zone

  where the zone is the 60 % of the pitch furthest from T's own goal (the
  standard definition). Defensive actions are ground challenges / tackles and
  interceptions made while the opponent has the ball; aerial duels are left
  out, as in Opta's definition. Lower = more aggressive press.

* **Counter-pressing.** After T loses the ball in open play, did T win it
  back (start a new open-play chain) within `window` seconds? The share of
  turnovers regained that way is the counter-press success rate.

* **High turnovers.** Open-play regains whose first touch is within 40 m of
  the opponent's goal (StatsBomb's definition), plus whether the chain they
  start ends in a shot.

Tracking-based **pressure** complements the events: in every frame where the
opponent has the ball, how many of T's players are within a radius of it,
and how close is the nearest one.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FPS, PITCH_LENGTH, player_arrays
from .labels import ON_BALL
from .pitch import HALF_L, TEAMS, distance_to_goal, other, zone_of

DEFENSIVE_RECOVERY = ("INTERCEPTION", "THEFT", "BLOCKED")


@dataclass
class Chain:
    team: str
    period: int
    start_frame: int
    end_frame: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    open_play: bool          # started without a stoppage (i.e. by winning the ball)
    events: list[int] = field(default_factory=list)
    passes: int = 0
    shot: bool = False
    goal: bool = False


def _is_goal(subtype: str) -> bool:
    return subtype.endswith("GOAL") and "WOODWORK" not in subtype


def possession_chains(events: pd.DataFrame) -> tuple[list[Chain], pd.Series]:
    """Split the event feed into possession chains.

    Also returns, for every event, the team in possession just *before* it
    (None during a stoppage), which is what decides whether a challenge was
    a defensive action.
    """
    ev = events  # Metrica's feed is chronological; keep its order (ties are meaningful)
    chains: list[Chain] = []
    cur: Chain | None = None
    dead = True
    period = None
    before = []
    for i, e in enumerate(ev.itertuples(index=False)):
        if e.period != period:
            period, dead = e.period, True
        before.append(None if (dead or cur is None) else cur.team)
        if e.type in ON_BALL:
            if cur is None or dead or e.team != cur.team:
                cur = Chain(e.team, int(e.period), int(e.start_frame), int(e.start_frame),
                            e.start_x, e.start_y, e.start_x, e.start_y,
                            open_play=not dead and e.type != "SET PIECE")
                chains.append(cur)
            dead = False
            cur.events.append(i)
            cur.end_frame = int(max(e.start_frame, e.end_frame))
            if np.isfinite(e.start_x):
                cur.end_x, cur.end_y = e.start_x, e.start_y
            cur.passes += e.type == "PASS"
            if e.type == "SHOT":
                cur.shot = True
                cur.goal = cur.goal or _is_goal(e.subtype)
            if "OUT" in e.subtype or (e.type == "SHOT" and _is_goal(e.subtype)):
                dead = True
        elif e.type in ("BALL OUT", "FAULT RECEIVED"):
            dead = True
    return chains, pd.Series(before, index=events.index, dtype=object)


def to_attacking(x: np.ndarray, periods: np.ndarray, team: str, directions: dict) -> np.ndarray:
    d = np.array([directions[team].get(int(p), 0) for p in periods])
    return np.asarray(x, dtype=float) * d


def ppda(events: pd.DataFrame, directions: dict, zone_fraction: float = 0.6,
         start_frame: int | None = None, end_frame: int | None = None) -> dict[str, float]:
    """PPDA per pressing team (optionally restricted to a frame window)."""
    _, before = possession_chains(events)
    ev = events.assign(before=before)
    if start_frame is not None:
        ev = ev[(ev["start_frame"] >= start_frame) & (ev["start_frame"] < end_frame)]
    zone_start = HALF_L - zone_fraction * PITCH_LENGTH  # in the pressing team's attacking frame
    out = {}
    for team in TEAMS:
        opp = other(team)
        x_att = to_attacking(ev["start_x"].to_numpy(), ev["period"].to_numpy(), team, directions)
        in_zone = x_att >= zone_start
        passes = (ev["team"] == opp) & (
            (ev["type"] == "PASS") | ((ev["type"] == "BALL LOST") & ev["subtype"].str.contains("INTERCEPTION")))
        defensive = (ev["team"] == team) & (ev["before"] == opp) & (
            ((ev["type"] == "CHALLENGE") & ~ev["subtype"].str.startswith("AERIAL"))
            | ((ev["type"] == "RECOVERY") & ev["subtype"].isin(DEFENSIVE_RECOVERY)))
        n_pass, n_def = int((passes & in_zone).sum()), int((defensive & in_zone).sum())
        out[team] = n_pass / n_def if n_def else float("nan")
    return out


def ppda_timeline(events: pd.DataFrame, directions: dict, window_minutes: float = 15.0) -> pd.DataFrame:
    """PPDA per team in consecutive windows of match time."""
    span = int(window_minutes * 60 * FPS)
    last = int(events["start_frame"].max()) + 1
    rows = []
    for s in range(0, last, span):
        rows.append({"start_minute": s / FPS / 60, **ppda(events, directions, start_frame=s, end_frame=s + span)})
    return pd.DataFrame(rows)


def turnovers(events: pd.DataFrame, directions: dict, window: float = 5.0,
              track: pd.DataFrame | None = None, radius: float = 10.0) -> pd.DataFrame:
    """One row per open-play turnover, seen from the team that lost the ball.

    Columns: team (that lost it), frame, location of the loss in that team's
    attacking frame, whether it was regained within `window` seconds, and the
    regain location. With tracking, `pressers` is the average number of the
    losing team's players within `radius` m of the ball during the window,
    i.e. how many joined the counter-press.
    """
    chains, _ = possession_chains(events)
    rows = []
    for lost, won, nxt in zip(chains, chains[1:], chains[2:] + [None]):
        if not won.open_play or won.team == lost.team or won.period != lost.period:
            continue
        frame = won.start_frame
        regained = (nxt is not None and nxt.team == lost.team and nxt.open_play
                    and nxt.period == won.period and nxt.start_frame - frame <= window * FPS)
        d = directions[lost.team].get(lost.period, 0)
        rows.append({
            "team": lost.team, "period": lost.period, "frame": frame,
            "x": lost.end_x * d, "y": lost.end_y * d,
            "regained": bool(regained),
            "seconds_to_regain": (nxt.start_frame - frame) / FPS if regained else np.nan,
            "regain_x": nxt.start_x * d if regained else np.nan,
        })
    df = pd.DataFrame(rows, columns=["team", "period", "frame", "x", "y", "regained",
                                     "seconds_to_regain", "regain_x"])
    if track is not None and len(df):
        df["pressers"] = [_mean_pressers(track, r.team, r.frame, int(window * FPS), radius)
                          for r in df.itertuples()]
    return df


def _mean_pressers(track: pd.DataFrame, team: str, frame: int, length: int, radius: float) -> float:
    sl = track.loc[frame: frame + length]
    if sl.empty:
        return np.nan
    px, py = player_arrays(sl, team)
    with np.errstate(invalid="ignore"):
        d = np.hypot(px - sl["ball_x"].to_numpy()[:, None], py - sl["ball_y"].to_numpy()[:, None])
    return float(np.nanmean(np.sum(d < radius, axis=1)))


def recoveries(events: pd.DataFrame, directions: dict, high_distance: float = 40.0) -> pd.DataFrame:
    """Open-play ball wins per team with zone, 'high turnover' flag and chain outcome."""
    chains, _ = possession_chains(events)
    rows = []
    for c in chains:
        if not c.open_play or not np.isfinite(c.start_x):
            continue
        d = directions[c.team].get(c.period, 0)
        x_att = c.start_x * d
        y_att = c.start_y * d  # mirror y too, so "left" stays the team's left
        third, channel = zone_of(np.array([x_att]), np.array([y_att]))
        rows.append({
            "team": c.team, "period": c.period, "frame": c.start_frame,
            "x": x_att, "y": y_att, "third": int(third[0]), "channel": int(channel[0]),
            "high": bool(distance_to_goal(x_att, y_att) <= high_distance),
            "shot": c.shot, "goal": c.goal,
        })
    return pd.DataFrame(rows, columns=["team", "period", "frame", "x", "y", "third", "channel",
                                       "high", "shot", "goal"])


def recovery_zones(rec: pd.DataFrame, team: str) -> np.ndarray:
    """(3 thirds x 3 channels) count of open-play regains; rows = defensive, middle, final third."""
    z = np.zeros((3, 3), dtype=int)
    for r in rec[rec["team"] == team].itertuples():
        z[r.third, r.channel] += 1
    return z


def pressure_series(track: pd.DataFrame, possession: np.ndarray, radius: float = 5.0) -> pd.DataFrame:
    """Per frame, pressure applied by the team *without* the ball.

    `possession` is +1 (home), -1 (away) or 0 per frame. Columns
    `home_pressers` / `away_pressers` count that team's players within
    `radius` m of the ball while the opponent has it (NaN otherwise);
    `home_nearest` / `away_nearest` give the distance of its closest player.
    """
    bx = track["ball_x"].to_numpy()[:, None]
    by = track["ball_y"].to_numpy()[:, None]
    out = {}
    for team, defending_when in (("home", -1), ("away", 1)):
        px, py = player_arrays(track, team)
        with np.errstate(invalid="ignore"):
            d = np.hypot(px - bx, py - by)
        mask = (possession == defending_when) & np.isfinite(bx[:, 0])
        count = np.sum(d < radius, axis=1).astype(float)
        nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        out[f"{team}_pressers"] = np.where(mask, count, np.nan)
        out[f"{team}_nearest"] = np.where(mask & np.isfinite(nearest), nearest, np.nan)
    return pd.DataFrame(out, index=track.index)


def pressing_summary(events: pd.DataFrame, directions: dict, track: pd.DataFrame | None = None,
                     possession: np.ndarray | None = None) -> dict:
    """All pressing KPIs per team in one dictionary (used by the report and insights)."""
    p = ppda(events, directions)
    tov = turnovers(events, directions, track=track)
    rec = recoveries(events, directions)
    pres = pressure_series(track, possession) if track is not None and possession is not None else None
    out = {}
    for team in TEAMS:
        t = tov[tov["team"] == team]
        r = rec[rec["team"] == team]
        high = r[r["high"]]
        row = {
            "ppda": p[team],
            "turnovers": int(len(t)),
            "counterpress_regains": int(t["regained"].sum()),
            "counterpress_rate": float(t["regained"].mean()) if len(t) else 0.0,
            "recoveries": int(len(r)),
            "high_turnovers": int(len(high)),
            "high_turnover_shots": int(high["shot"].sum()),
            "recovery_zones": recovery_zones(rec, team).tolist(),
        }
        if "pressers" in t:
            row["counterpress_pressers"] = float(t["pressers"].mean()) if len(t) else 0.0
        if pres is not None:
            row["pressers_on_ball"] = float(np.nanmean(pres[f"{team}_pressers"]))
            row["nearest_presser_m"] = float(np.nanmedian(pres[f"{team}_nearest"]))
        out[team] = row
    return out
