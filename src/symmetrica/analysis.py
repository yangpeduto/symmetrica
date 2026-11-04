"""Run every model on one match and collect a JSON-serialisable summary.

`analyse_match` only needs a tracking table and an event table in the
formats produced by `symmetrica.data`, so it runs the same on the Metrica
sample games and on synthetic matches (see `symmetrica.synthetic`).
"""
import base64
import math
import time

import numpy as np
import pandas as pd

from . import insights, labels, models, momentum, networks, pitch_control, pressing, threat
from .data import FPS, player_ids
from .kinematics import frame_state, velocities
from .pitch import TEAMS, Grid, attack_direction

# Parameters tuned out-of-sample in scripts/evaluate.py (identical for both splits).
POSSESSION_PARAMS = {"radius": 0.5, "max_speed": 99.0, "min_frames": 1}
CLIP_GRID = Grid(32, 21)


def _clean(obj):
    """Recursively convert numpy scalars / NaN / inf into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _goal_frames(acts: pd.DataFrame) -> list[tuple[int, str]]:
    g = acts[(acts["kind"] == "shot") & acts["goal"]]
    return [(int(f), t) for f, t in zip(g["frame"], g["team"])]


def _encode_surface(home: np.ndarray) -> str:
    """Home control probability quantised to one byte per cell, base64."""
    q = np.clip(np.round(np.nan_to_num(home, nan=0.5) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def build_clip(track: pd.DataFrame, vel: pd.DataFrame, possession: np.ndarray, start: int, end: int,
               title: str, step: int = 3, grid: Grid = CLIP_GRID,
               params: pitch_control.PitchControlParams = pitch_control.PitchControlParams()) -> dict:
    """Positions + pitch-control surfaces for an animated passage of play (row positions)."""
    start, end = max(start, 0), min(end, len(track))
    stems = {t: player_ids(track, t) for t in TEAMS}
    frames = []
    for pos in range(start, end, step):
        st = frame_state(track, vel, pos)
        if not np.all(np.isfinite(st.ball)):
            surface = None
        else:
            surface = _encode_surface(pitch_control.pitch_control(st, grid, params).home)
        row = track.iloc[pos]
        players = {}
        for t in TEAMS:
            players[t] = [[int(s.split("_")[1]), round(float(row[f"{s}_x"]), 2), round(float(row[f"{s}_y"]), 2)]
                          for s in stems[t] if np.isfinite(row[f"{s}_x"])]
        frames.append({
            "f": int(track.index[pos]), "t": round(float(row["time"]), 2),
            "ball": [round(float(row["ball_x"]), 2), round(float(row["ball_y"]), 2)]
            if np.isfinite(row["ball_x"]) else None,
            "home": players["home"], "away": players["away"],
            "pos": int(possession[pos]), "pc": surface,
        })
    return {"title": title, "fps": FPS / step, "grid": [grid.nx, grid.ny], "frames": frames}


def choose_clips(track: pd.DataFrame, acts: pd.DataFrame, passes: pd.DataFrame, max_clips: int = 5,
                 before: float = 10.0, after: float = 3.0) -> list[tuple[int, int, str]]:
    """Goals first, then the passes that took the most opponents out of the game."""
    index = {f: i for i, f in enumerate(track.index)}
    clips = []
    for frame, team in _goal_frames(acts)[:max_clips]:
        if frame in index:
            minute = int((frame - 1) / FPS // 60)
            clips.append((index[frame] - int(before * FPS), index[frame] + int(after * FPS),
                          f"{team.capitalize()} goal · {minute}'"))
    best = passes[passes["kind"] == "pass"].sort_values(["packing", "xt_added"], ascending=False)
    for r in best.itertuples():
        if len(clips) >= max_clips:
            break
        if int(r.frame) not in index:
            continue
        i = index[int(r.frame)]
        if any(abs(i - (s + int(before * FPS))) < 15 * FPS for s, _, _ in clips):
            continue
        minute = int((r.frame - 1) / FPS // 60)
        clips.append((i - int(6 * FPS), i + int(6 * FPS),
                      f"{r.team.capitalize()} line-break · {_short(r.player)} → {_short(r.receiver)} · "
                      f"{int(r.packing)} opponents out · {minute}'"))
    return clips


def _short(pid) -> str:
    return f"#{str(pid).split('_')[1]}" if isinstance(pid, str) and "_" in pid else "?"


def analyse_match(track: pd.DataFrame, events: pd.DataFrame, name: str = "Match",
                  xt: threat.ExpectedThreat | None = None, pc_interval: float = 10.0,
                  max_clips: int = 5, clip_step: int = 3) -> dict:
    """Every metric for one match. `xt` defaults to a surface fitted on this match alone."""
    timings = {}
    t0 = time.perf_counter()

    def lap(key):
        nonlocal t0
        now = time.perf_counter()
        timings[key] = round(now - t0, 3)
        t0 = now

    directions = attack_direction(track)
    poss = labels.possession_labels(events, track.index, track["period"])
    d_home, d_away = models.nearest_distances(track)
    est = models.control_with_memory(d_home, d_away, models.ball_speed(track), **POSSESSION_PARAMS)
    live = (poss != 0) & np.isfinite(track["ball_x"].to_numpy())
    vel = velocities(track)
    lap("possession_and_kinematics")

    space, space_map = pitch_control.voronoi_series(track, directions)
    lap("voronoi")
    pc_rows = np.arange(0, len(track), max(int(pc_interval * FPS), 1))
    pc = pitch_control.pitch_control_series(track, vel, pc_rows, CLIP_GRID)
    lap("pitch_control_series")

    acts = threat.actions(events, directions)
    xt = xt or threat.ExpectedThreat.fit(acts)
    passes = threat.pass_table(events, directions, xt, track, acts)
    leaders = threat.player_threat(passes)
    lap("threat")

    press = pressing.pressing_summary(events, directions, track, poss)
    ppda_t = pressing.ppda_timeline(events, directions)
    rec = pressing.recoveries(events, directions)
    lap("pressing")

    tl = momentum.momentum_timeline(passes, track, poss, space)
    nets = {t: networks.passing_network(passes, track, poss, t, directions) for t in TEAMS}
    lap("momentum_and_networks")

    score = {t: int(((acts["team"] == t) & acts["goal"]).sum()) for t in TEAMS}
    shots = acts[acts["kind"] == "shot"]
    top_passes = passes[passes["kind"] == "pass"].sort_values("xt_added", ascending=False).head(12)

    summary = {
        "meta": {"name": name, "frames": int(len(track)), "minutes": round(len(track) / FPS / 60, 1),
                 "score": score, "directions": directions,
                 "period_starts": [round(float(momentum.minute_of_frame(track.index[track["period"] == p][0])), 2)
                                   for p in sorted(track["period"].dropna().unique())]},
        "possession": {
            "home_share": float((poss[live] == 1).mean()) if live.any() else 0.5,
            "tracking_model_accuracy": float((est[live] == poss[live]).mean()) if live.any() else None,
            "in_play_share": float(live.mean()),
        },
        "space": {
            "home_share": float(space["home_share"].mean()),
            "home_final_third": float(space["home_final_third"].mean()),
            "away_final_third": float(space["away_final_third"].mean()),
            "pitch_control_home_share": float(pc["home_share"].mean()),
            "voronoi_map": np.round(space_map, 3),
            "voronoi_grid": [space_map.shape[1], space_map.shape[0]],
        },
        "pressing": {**press, "ppda_timeline": ppda_t.to_dict(orient="records"),
                     "recoveries": rec.assign(minute=momentum.minute_of_frame(rec["frame"]).round(1))
                     [["team", "minute", "x", "y", "high", "shot", "goal"]].round(2).to_dict(orient="records")},
        "threat": {
            **threat.threat_summary(passes, acts),
            "xt_grid": np.round(xt.values, 4),
            "leaders": leaders.head(10).to_dict(orient="records"),
            "top_passes": top_passes[["team", "player", "receiver", "frame", "x", "y", "end_x", "end_y",
                                      "xt_added", "lines_broken", "packing", "progressive"]]
            .round(3).to_dict(orient="records"),
        },
        "momentum": {
            "timeline": tl.reset_index().round(4).to_dict(orient="records"),
            "shots": [{"minute": round(float(momentum.minute_of_frame(f)), 2), "team": t, "goal": bool(g)}
                      for f, t, g in zip(shots["frame"], shots["team"], shots["goal"])],
            "best_spell": {t: momentum.best_spell(tl, t) for t in TEAMS},
        },
        "networks": nets,
    }
    summary["insights"] = [i.to_dict() for i in insights.generate(_clean(summary))]
    lap("insights")

    summary["clips"] = [build_clip(track, vel, poss, s, e, title, step=clip_step)
                        for s, e, title in choose_clips(track, acts, passes, max_clips)]
    lap("clips")
    summary["timings"] = timings
    return _clean(summary)
