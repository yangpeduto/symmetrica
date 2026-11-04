"""A synthetic match in Metrica's formats, for tests and offline demos.

Two 4-4-2 teams shift with the ball around their formation (with smooth
per-player noise); the ball is held, passed, intercepted, tackled away,
kicked out and shot. Tracking and events are generated together, so the
event feed is consistent with the positions, and the event table comes out
in Metrica's raw layout (normalised coordinates, 'PlayerN' names) so it
goes through the same `prepare_events` as the real data.

Nothing here is meant to be realistic football, only plausible enough to
exercise every code path of the analysis.
"""
import numpy as np
import pandas as pd

from .data import EVENT_COLUMNS, FPS, PITCH_LENGTH, PITCH_WIDTH, prepare_events

# Formation in the attacking frame (team attacks +x): GK, back four, midfield four, two strikers.
FORMATION = np.array([[-48, 0], [-30, -20], [-32, -7], [-32, 7], [-30, 20],
                      [-12, -22], [-14, -7], [-14, 7], [-12, 22], [4, -7], [4, 7]], dtype=float)
NUMBERS = {"home": list(range(1, 12)), "away": list(range(12, 23))}


def _norm(x, y):
    return x / PITCH_LENGTH + 0.5, 0.5 - y / PITCH_WIDTH


def synthetic_match(minutes_per_half: float = 3.0, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_half = int(minutes_per_half * 60 * FPS)
    rows, events = [], []
    frame = 1

    def ev(team, typ, sub, period, f0, f1, frm, to, p0, p1):
        x0, y0 = _norm(*p0) if p0 is not None else (np.nan, np.nan)
        x1, y1 = _norm(*p1) if p1 is not None else (np.nan, np.nan)
        events.append([team.capitalize(), typ, sub, period, f0, f0 / FPS, f1, f1 / FPS,
                       f"Player{frm}" if frm else np.nan, f"Player{to}" if to else np.nan, x0, y0, x1, y1])

    noise = {t: np.zeros((11, 2)) for t in ("home", "away")}
    for period in (1, 2):
        direction = {"home": 1 if period == 1 else -1, "away": -1 if period == 1 else 1}
        team = "home" if period == 1 else "away"
        holder = 9  # a striker kicks off
        ball = np.zeros(2)
        pass_plan = None      # (t0, t1, start, end, receiver_team, receiver_idx)
        hold_left = int(FPS * 0.5)
        kickoff = True

        def positions(t, ball_pos):
            d = direction[t]
            shift = np.array([0.45 * ball_pos[0], 0.3 * ball_pos[1]])
            base = FORMATION.copy()
            base[1:, 0] += 12  # outfield lines push up once play starts
            pos = base * [d, d] + shift
            pos[0] = base[0] * [d, d] + shift * [0.15, 0.3]  # keeper stays home
            return pos + noise[t]

        for k in range(n_half):
            for t in noise:
                noise[t] = 0.98 * noise[t] + rng.normal(0, 0.12, (11, 2))
            pos = {t: positions(t, ball) for t in ("home", "away")}
            if kickoff:
                for t in pos:  # at kick-off everyone is in his own half
                    pos[t][:, 0] = np.where(np.sign(pos[t][:, 0]) == direction[t],
                                            -direction[t] * 1.0, pos[t][:, 0])
            if pass_plan is None:
                ball = pos[team][holder] + [0.4 * direction[team], 0.0]
                hold_left -= 1
                if hold_left <= 0:
                    kickoff = False
                    x_att = ball[0] * direction[team]
                    num = NUMBERS[team][holder]
                    opp = "away" if team == "home" else "home"
                    if k == 0:
                        ev(team, "SET PIECE", "KICK OFF", period, frame, frame, num, None, ball, None)
                    u = rng.random()
                    if x_att > 25 and u < 0.35:  # shot
                        goal_pt = np.array([52.5 * direction[team], rng.uniform(-3, 3)])
                        scored = rng.random() < 0.3
                        t1 = frame + int(np.linalg.norm(goal_pt - ball) / 25 * FPS) + 1
                        ev(team, "SHOT", "ON TARGET-GOAL" if scored else "ON TARGET-SAVED", period,
                           frame, t1, num, None, ball, goal_pt)
                        if scored:
                            pass_plan = (frame, t1, ball.copy(), goal_pt, "restart", opp)
                        else:
                            pass_plan = (frame, t1, ball.copy(), goal_pt, "save", opp)
                    elif u < 0.06:  # tackled
                        taker = int(np.argmin(np.linalg.norm(pos[opp] - ball, axis=1)))
                        tn = NUMBERS[opp][taker]
                        ev(opp, "CHALLENGE", "TACKLE-WON", period, frame, frame, tn, None, ball, None)
                        ev(team, "CHALLENGE", "TACKLE-LOST", period, frame, frame, num, None, ball, None)
                        ev(team, "BALL LOST", "THEFT", period, frame, frame, num, None, ball, None)
                        ev(opp, "RECOVERY", "THEFT", period, frame, frame, tn, None, ball, None)
                        team, holder, hold_left = opp, taker, int(FPS * rng.uniform(0.6, 2.0))
                    elif u < 0.09:  # kicked out of play, throw-in to the opponent
                        out_pt = np.array([ball[0], 34.5 * np.sign(ball[1] or 1)])
                        t1 = frame + int(FPS * 1.0)
                        ev(team, "BALL LOST", "", period, frame, t1, num, None, ball, out_pt)
                        ev(team, "BALL OUT", "", period, t1, t1, num, None, out_pt, None)
                        pass_plan = (frame, t1, ball.copy(), out_pt, "throw", opp)
                    else:  # pass, sometimes intercepted
                        mates = [i for i in range(11) if i != holder]
                        fwd = np.array([(pos[team][i, 0] - ball[0]) * direction[team] for i in mates])
                        w = np.exp(np.clip(fwd, -30, 30) / 15)
                        rec = mates[int(rng.choice(len(mates), p=w / w.sum()))]
                        end = pos[team][rec].copy()
                        t1 = frame + max(int(np.linalg.norm(end - ball) / 14 * FPS), 3)
                        if rng.random() < 0.14:
                            mid = ball + (end - ball) * rng.uniform(0.5, 0.9)
                            icp = int(np.argmin(np.linalg.norm(pos[opp] - mid, axis=1)))
                            t1 = frame + max(int(np.linalg.norm(mid - ball) / 14 * FPS), 3)
                            ev(team, "BALL LOST", "INTERCEPTION", period, frame, t1, num, None, ball, mid)
                            ev(opp, "RECOVERY", "INTERCEPTION", period, t1, t1, NUMBERS[opp][icp], None, mid, None)
                            pass_plan = (frame, t1, ball.copy(), mid, opp, icp)
                        else:
                            ev(team, "PASS", "", period, frame, t1, num, NUMBERS[team][rec], ball, end)
                            pass_plan = (frame, t1, ball.copy(), end, team, rec)
            else:
                t0, t1, start, end, who, idx = pass_plan
                a = min((frame - t0) / max(t1 - t0, 1), 1.0)
                ball = start + (end - start) * a
                if frame >= t1:
                    if who in ("restart", "save", "throw"):
                        new = idx
                        if who == "restart":
                            holder, ball = 9, np.zeros(2)
                            ev(new, "SET PIECE", "KICK OFF", period, frame + 1, frame + 1,
                               NUMBERS[new][holder], None, ball, None)
                        elif who == "save":
                            holder = 0
                            ev(new, "RECOVERY", "SAVED", period, frame + 1, frame + 1,
                               NUMBERS[new][holder], None, pos[new][0], None)
                        else:
                            holder = int(np.argmin(np.linalg.norm(pos[new] - end, axis=1)))
                            ev(new, "SET PIECE", "THROW IN", period, frame + 1, frame + 1,
                               NUMBERS[new][holder], None, end, None)
                        team = new
                    else:
                        team, holder = who, idx
                    pass_plan = None
                    hold_left = int(FPS * rng.uniform(0.6, 2.0))

            row = {"frame": frame, "period": period, "time": frame / FPS}
            for t in ("home", "away"):
                for i, n in enumerate(NUMBERS[t]):
                    row[f"{t}_{n}_x"], row[f"{t}_{n}_y"] = pos[t][i]
            row["ball_x"], row["ball_y"] = ball
            rows.append(row)
            frame += 1

    track = pd.DataFrame(rows).set_index("frame")
    raw = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return track, prepare_events(raw)
