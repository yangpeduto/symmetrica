"""Pass options: when a pass is played, how good was every alternative?

At the frame a pass starts, every teammate of the passer is a candidate
receiver j. For each one

    completion_j = PC_att(r_j) * (1 - max_k I(l_jk))

i.e. the passer's team must win the ball at the receiver's position
(Spearman's pass-success probability: pitch control with the ball's travel
time to r_j; the two teams' shares normalised to sum to 1), and no
defender may cut out the ball on the way. For a lane point l the ball gets
there after t_ball(l) = |l - ball| / v_ball, and the chance that at least
one defender is there first is

    I(l) = 1 - prod_d (1 - F_d(t_ball(l))),   F_d(t) = logistic((t - tau_d(l)) / s)

with the same arrival-time model and spread as pitch control, except that a
defender only has to get within leg reach (1 m) of the ball's path. Lane
points are sampled at 20 / 40 / 60 / 80 % of the straight line.

The expected value of the option is completion_j * xT(r_j), and its gain is
measured against the ball's current location,

    gain_j = completion_j * xT(r_j) - xT(ball),

so a safe square pass scores ~0, a risky forward pass is worth its threat
times its chance of arriving, and a lost ball is worth nothing (the
opponent's counter-threat is not modelled). The best option is the
teammate with the largest gain; comparing it with the receiver actually
chosen gives a decision-quality measure for every pass.
"""
import numpy as np
import pandas as pd

from .kinematics import FrameState, frame_state
from .pitch import TEAMS, other
from .pitch_control import PitchControlParams, pitch_control_at, time_to_intercept
from .threat import ExpectedThreat

LANE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def interception_risk(points: np.ndarray, ball: np.ndarray, def_pos: np.ndarray, def_vel: np.ndarray,
                      params: PitchControlParams = PitchControlParams()) -> np.ndarray:
    """Probability that at least one defender reaches each point before a ball played through it."""
    if len(def_pos) == 0:
        return np.zeros(len(points))
    tau = time_to_intercept(def_pos, def_vel, points, params, reach=params.interception_reach)
    t_ball = np.linalg.norm(points - ball, axis=1) / params.ball_speed
    scale = np.pi / (np.sqrt(3.0) * params.tti_sigma)
    f = 1.0 / (1.0 + np.exp(-scale * (t_ball[:, None] - tau)))
    return 1.0 - np.prod(1.0 - f, axis=1)


def evaluate_options(state: FrameState, team: str, passer: str, direction: int, xt: ExpectedThreat,
                     params: PitchControlParams = PitchControlParams()) -> list[dict]:
    """Every teammate of `passer` as a pass target, best (highest gain) first."""
    ids, pos, vel = state.team(team)
    _, opp_pos, opp_vel = state.team(other(team))
    ball = state.ball
    keep = [i for i, pid in enumerate(ids) if pid != passer]
    if not keep or not np.all(np.isfinite(ball)):
        return []
    targets = pos[keep]
    n = len(targets)
    lanes = np.concatenate([ball + f * (targets - ball) for f in LANE_FRACTIONS])
    att, dfn = pitch_control_at(targets, ball, pos, vel, opp_pos, opp_vel, params)
    att = att / np.maximum(att + dfn, 1e-9)
    risk = interception_risk(lanes, ball, opp_pos, opp_vel, params)
    lane_risk = risk.reshape(len(LANE_FRACTIONS), n).max(axis=0)
    completion = att * (1 - lane_risk)
    xt_target = xt.interpolate(targets[:, 0] * direction, targets[:, 1] * direction)
    xt_ball = float(xt.interpolate(ball[0] * direction, ball[1] * direction))
    value = completion * xt_target
    out = [{"id": ids[k], "number": int(ids[k].split("_")[1]),
            "x": float(targets[j, 0]), "y": float(targets[j, 1]),
            "completion": float(completion[j]), "lane_risk": float(lane_risk[j]),
            "xt": float(xt_target[j]), "value": float(value[j]), "gain": float(value[j] - xt_ball)}
           for j, k in enumerate(keep)]
    return sorted(out, key=lambda o: -o["gain"])


def pass_decisions(events: pd.DataFrame, track: pd.DataFrame, vel: pd.DataFrame, directions: dict,
                   xt: ExpectedThreat, params: PitchControlParams = PitchControlParams()) -> pd.DataFrame:
    """One row per completed pass: the chosen option against the best one available."""
    index = {f: i for i, f in enumerate(track.index)}
    rows = []
    passes = events[(events["type"] == "PASS") & events["receiver"].notna()]
    for e in passes.itertuples(index=False):
        pos = index.get(int(e.start_frame))
        if pos is None:
            continue
        state = frame_state(track, vel, pos)
        d = directions[e.team].get(int(e.period), 0)
        opts = evaluate_options(state, e.team, e.player, d, xt, params)
        actual = next((o for o in opts if o["id"] == e.receiver), None)
        if not opts or actual is None or e.player not in state.team(e.team)[0]:
            continue
        best = opts[0]
        rows.append({
            "team": e.team, "player": e.player, "receiver": e.receiver, "frame": int(e.start_frame),
            "best": best["id"], "chose_best": best["id"] == e.receiver,
            "rank": next(i for i, o in enumerate(opts) if o["id"] == e.receiver) + 1,
            "actual_gain": actual["gain"], "best_gain": best["gain"],
            "actual_completion": actual["completion"], "best_completion": best["completion"],
            "n_options": len(opts),
        })
    cols = ["team", "player", "receiver", "frame", "best", "chose_best", "rank", "actual_gain",
            "best_gain", "actual_completion", "best_completion", "n_options"]
    return pd.DataFrame(rows, columns=cols)


def decision_summary(dec: pd.DataFrame) -> dict:
    out = {}
    for team in TEAMS:
        d = dec[dec["team"] == team]
        out[team] = {
            "passes_evaluated": int(len(d)),
            "best_option_share": float(d["chose_best"].mean()) if len(d) else 0.0,
            "top3_share": float((d["rank"] <= 3).mean()) if len(d) else 0.0,
            "mean_gain_chosen": float(d["actual_gain"].mean()) if len(d) else 0.0,
            "mean_gain_best": float(d["best_gain"].mean()) if len(d) else 0.0,
            "epv_left_per_pass": float((d["best_gain"] - d["actual_gain"]).mean()) if len(d) else 0.0,
            "mean_completion_chosen": float(d["actual_completion"].mean()) if len(d) else 0.0,
        }
    return out
