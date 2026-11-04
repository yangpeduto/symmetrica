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

from . import (insights, labels, models, momentum, networks, offball, pass_options, pitch_control, players,
               pressing, shape, threat)
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


def _encode(values: np.ndarray, scale: float = 1.0, fill: float = 0.5) -> str:
    """Values in [0, scale] quantised to one byte per cell, base64."""
    q = np.clip(np.round(np.nan_to_num(values / scale, nan=fill) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _num(pid) -> int | None:
    return int(str(pid).split("_")[1]) if isinstance(pid, str) and "_" in pid else None


def build_clip(track: pd.DataFrame, vel: pd.DataFrame, possession: np.ndarray, start: int, end: int,
               title: str, directions: dict, xt: threat.ExpectedThreat | None = None,
               runs: pd.DataFrame | None = None, events: pd.DataFrame | None = None,
               keepers: dict | None = None, step: int = 3, grid: Grid = CLIP_GRID,
               params: pitch_control.PitchControlParams = pitch_control.PitchControlParams()) -> dict:
    """Everything the replay canvas draws for one passage of play (row positions start..end).

    Per sampled frame: player positions and velocities (with a flag for
    players in a detected off-ball run), the ball, possession, the home
    pitch-control surface, the EPV surface of the team in possession and its
    most valuable cell. Passes played during the clip carry the evaluation
    of every pass option (best option vs the one chosen).
    """
    start, end = max(start, 0), min(end, len(track))
    stems = {t: player_ids(track, t) for t in TEAMS}
    running = set()
    if runs is not None and len(runs):
        for r in runs[(runs["end"] >= start) & (runs["start"] < end)].itertuples():
            running.update((r.player, p) for p in range(max(r.start, start), min(r.end, end) + 1))
    period = track["period"].to_numpy()
    frames, epvs = [], []
    last_att = None
    for pos in range(start, end, step):
        st = frame_state(track, vel, pos)
        row = track.iloc[pos]
        vrow = vel.iloc[pos]
        p = int(period[pos])
        att = {1: "home", -1: "away"}.get(int(possession[pos]), last_att)
        last_att = att or last_att
        pc_surface, epv = None, None
        if np.all(np.isfinite(st.ball)) and len(st.home_pos) and len(st.away_pos):
            surf = pitch_control.pitch_control(st, grid, params)
            pc_surface = _encode(surf.home)
            if xt is not None and att is not None:
                ctrl = surf.home if att == "home" else surf.away
                epv = threat.epv_surface(ctrl, grid.cells, xt, directions[att].get(p, 0))
        players_out = {}
        for t in TEAMS:
            players_out[t] = []
            for s in stems[t]:
                x, y = row[f"{s}_x"], row[f"{s}_y"]
                if not np.isfinite(x):
                    continue
                vx, vy = vrow.get(f"{s}_vx", 0.0), vrow.get(f"{s}_vy", 0.0)
                players_out[t].append([int(s.split("_")[1]), round(float(x), 2), round(float(y), 2),
                                       round(float(vx) if np.isfinite(vx) else 0.0, 2),
                                       round(float(vy) if np.isfinite(vy) else 0.0, 2),
                                       int((s, pos) in running)])
        frame = {
            "f": int(track.index[pos]), "t": round(float(row["time"]), 2),
            "ball": [round(float(row["ball_x"]), 2), round(float(row["ball_y"]), 2)]
            if np.isfinite(row["ball_x"]) else None,
            "home": players_out["home"], "away": players_out["away"],
            "pos": int(possession[pos]), "att": att, "hd": directions["home"].get(p, 0), "pc": pc_surface,
        }
        if epv is not None:
            k = int(np.nanargmax(epv))
            frame["best_space"] = [round(float(grid.cells[k, 0]), 1), round(float(grid.cells[k, 1]), 1),
                                   round(float(epv[k]), 4)]
        frames.append(frame)
        epvs.append(epv)
    scale = max([float(np.nanmax(e)) for e in epvs if e is not None] + [1e-6])
    for f, e in zip(frames, epvs):
        f["epv"] = _encode(e, scale, fill=0.0) if e is not None else None

    # Pass options for each pass played during the clip, shown from its sampled frame onwards.
    passes_out = []
    if events is not None and xt is not None:
        index = {fr: i for i, fr in enumerate(track.index)}
        f0, f1 = int(track.index[start]), int(track.index[end - 1])
        ev = events[(events["type"] == "PASS") & (events["start_frame"] >= f0) & (events["start_frame"] <= f1)]
        for e in ev.itertuples(index=False):
            i = index.get(int(e.start_frame))
            if i is None:
                continue
            st = frame_state(track, vel, i)
            opts = pass_options.evaluate_options(st, e.team, e.player, directions[e.team].get(int(e.period), 0),
                                                 xt, params)
            if not opts:
                continue
            passes_out.append({
                "team": e.team, "frame": int(e.start_frame), "end_frame": int(max(e.end_frame, e.start_frame)),
                "passer": _num(e.player), "receiver": _num(e.receiver),
                "ball": [round(float(st.ball[0]), 2), round(float(st.ball[1]), 2)],
                "best": opts[0]["number"],
                "options": [{"n": o["number"], "x": round(o["x"], 2), "y": round(o["y"], 2),
                             "c": round(o["completion"], 3), "g": round(o["gain"], 4)} for o in opts],
            })
    return {"title": title, "fps": FPS / step, "grid": [grid.nx, grid.ny], "epv_scale": scale,
            "keepers": {t: sorted(_num(k) for k in (keepers or {}).get(t, [])) for t in TEAMS},
            "frames": frames, "passes": passes_out}


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


def _run_records(runs: pd.DataFrame, directions: dict) -> list[dict]:
    out = []
    for r in runs.itertuples():
        d = directions[r.team].get(int(r.period), 0)
        out.append({"team": r.team, "player": r.player, "minute": round(float(momentum.minute_of_frame(r.frame)), 1),
                    "x0": round(r.x0 * d, 1), "y0": round(r.y0 * d, 1), "x1": round(r.x1 * d, 1),
                    "y1": round(r.y1 * d, 1), "category": r.category, "subtype": r.subtype,
                    "duration": round(r.duration, 2), "max_speed": round(r.max_speed, 2),
                    "space_m2": round(r.space_m2, 1), "space_epv": round(r.space_epv, 4),
                    "marker": r.marker})
    return out


def analyse_match(track: pd.DataFrame, events: pd.DataFrame, name: str = "Match",
                  xt: threat.ExpectedThreat | None = None, pc_interval: float = 10.0,
                  max_clips: int = 5, clip_step: int = 3,
                  pc_params: pitch_control.PitchControlParams = pitch_control.PitchControlParams()) -> dict:
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
    pc = pitch_control.pitch_control_series(track, vel, pc_rows, CLIP_GRID, pc_params)
    lap("pitch_control_series")

    acts = threat.actions(events, directions)
    xt = xt or threat.ExpectedThreat.fit(acts)
    passes = threat.pass_table(events, directions, xt, track, acts)
    leaders = threat.player_threat(passes)
    lap("threat")

    decisions = pass_options.pass_decisions(events, track, vel, directions, xt, pc_params)
    lap("pass_options")

    press = pressing.pressing_summary(events, directions, track, poss)
    ppda_t = pressing.ppda_timeline(events, directions)
    rec = pressing.recoveries(events, directions)
    lap("pressing")

    runs = offball.off_ball_runs(track, vel, poss, directions, xt, pc_params=pc_params)
    lap("offball_runs")

    keepers = shape.goalkeepers(track, directions)
    shapes = shape.shape_series(track, poss, directions, keepers=keepers)
    forms = shape.formations(track, poss, directions, keepers)
    lap("shape")

    roles = {**shape.roles(forms, "home"), **shape.roles(forms, "away")}
    profiles = players.player_profiles(track, vel, events, poss, directions, passes, runs, decisions, roles)
    players.add_percentiles(profiles)
    lap("players")

    tl = momentum.momentum_timeline(passes, track, poss, space)
    nets = {t: networks.passing_network(passes, track, poss, t, directions) for t in TEAMS}
    lap("momentum_and_networks")

    score = {t: int(((acts["team"] == t) & acts["goal"]).sum()) for t in TEAMS}
    shots = acts[acts["kind"] == "shot"]
    top_passes = passes[passes["kind"] == "pass"].sort_values("xt_added", ascending=False).head(12)
    missed = decisions.assign(left=decisions["best_gain"] - decisions["actual_gain"]).sort_values(
        "left", ascending=False).head(10)

    summary = {
        "meta": {"name": name, "frames": int(len(track)), "minutes": round(len(track) / FPS / 60, 1),
                 "score": score, "directions": directions,
                 "period_starts": [round(float(momentum.minute_of_frame(track.index[track["period"] == p][0])), 2)
                                   for p in sorted(track["period"].dropna().unique())],
                 "pitch_control_model": {"motion": pc_params.motion, "a_max": pc_params.max_acceleration,
                                         "v_max": pc_params.top_speed}},
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
        "decisions": {**pass_options.decision_summary(decisions),
                      "missed": missed[["team", "player", "receiver", "best", "frame", "actual_gain", "best_gain",
                                        "actual_completion", "best_completion"]].round(4).to_dict(orient="records")},
        "offball": {**offball.runs_summary(runs), "categories": list(offball.CATEGORIES),
                    "runs": _run_records(runs, directions)},
        "shape": {**shape.shape_summary(shapes), "formations": forms, "timeline": shape.shape_timeline(shapes)},
        "players": profiles,
        "player_metrics": players.PERCENTILE_METRICS,
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

    summary["clips"] = [build_clip(track, vel, poss, s, e, title, directions, xt, runs, events, keepers,
                                   step=clip_step, params=pc_params)
                        for s, e, title in choose_clips(track, acts, passes, max_clips)]
    lap("clips")
    summary["timings"] = timings
    return _clean(summary)

