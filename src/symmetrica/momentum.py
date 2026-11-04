"""Match momentum: who is on top, minute by minute.

Momentum is the exponentially weighted difference of threat created
(positive xT added by passes and carries) per minute,

    m_t = EWM_span( xT_home(t) - xT_away(t) ),

scaled to [-1, 1] by its largest absolute value in the match. Alongside it
the timeline carries possession share (from the event labels) and space
share (Voronoi), so a spell of pressure can be read against who had the
ball and who owned the pitch.
"""
import numpy as np
import pandas as pd

from .data import FPS


def minute_of_frame(frames: np.ndarray) -> np.ndarray:
    return (np.asarray(frames, dtype=float) - 1) / FPS / 60.0


def momentum_timeline(passes: pd.DataFrame, track: pd.DataFrame, possession: np.ndarray,
                      space: pd.DataFrame | None = None, span: float = 5.0) -> pd.DataFrame:
    """One row per match minute with xT per team, possession, space and smoothed momentum."""
    n_min = int(np.ceil(len(track) / FPS / 60))
    minutes = np.arange(n_min)
    pos_xt = passes.assign(minute=minute_of_frame(passes["frame"]).astype(int),
                           v=passes["xt_added"].clip(lower=0))
    out = pd.DataFrame(index=minutes)
    for team in ("home", "away"):
        s = pos_xt[pos_xt["team"] == team].groupby("minute")["v"].sum()
        out[f"{team}_xt"] = s.reindex(minutes, fill_value=0.0)

    frame_min = minute_of_frame(track.index.to_numpy()).astype(int)
    pos = pd.DataFrame({"m": frame_min, "home": possession == 1, "live": possession != 0})
    agg = pos.groupby("m")[["home", "live"]].sum()
    out["home_possession"] = (agg["home"] / agg["live"].replace(0, np.nan)).reindex(minutes)
    if space is not None:
        sm = minute_of_frame(space.index.to_numpy()).astype(int)
        out["home_space"] = space["home_share"].groupby(sm).mean().reindex(minutes)

    diff = (out["home_xt"] - out["away_xt"]).ewm(span=span, adjust=False).mean()
    scale = float(np.max(np.abs(diff))) or 1.0
    out["momentum"] = diff / scale
    out.index.name = "minute"
    return out


def best_spell(timeline: pd.DataFrame, team: str, length: int = 10) -> tuple[int, float]:
    """Start minute and xT difference of the team's strongest `length`-minute spell."""
    sign = 1 if team == "home" else -1
    d = sign * (timeline["home_xt"] - timeline["away_xt"])
    roll = d.rolling(length, min_periods=1).sum().shift(-(length - 1)).dropna()
    if roll.empty:
        return 0, 0.0
    start = int(roll.idxmax())
    return start, float(roll.max())
