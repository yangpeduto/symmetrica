"""Off-ball runs, team shape / formations and player profiles."""
import numpy as np
import pytest

from conftest import make_events, make_track
from symmetrica import kinematics, offball, players, shape
from symmetrica.data import FPS


def run_scene(n=75, speed=6.0, runner_start=(-10.0, -15.0), direction=(1.0, 0.0), ball=(0.0, 20.0),
              defenders=((40.0, 0.0), (40.0, 10.0), (50.0, 0.0)), marker=None, carrier=None):
    """Home #1 runs at `speed` m/s, home #2 holds the ball; home attacks +x (period 1)."""
    frames = []
    ux, uy = direction
    carrier = carrier or ball
    for k in range(n):
        dt = k / FPS
        runner = (runner_start[0] + speed * ux * dt, runner_start[1] + speed * uy * dt)
        away = list(defenders)
        if marker is not None:
            away.append((marker[0] + speed * ux * dt, marker[1] + speed * uy * dt))
        frames.append({"home": [runner, carrier, (-48.0, 0.0)], "away": away, "ball": ball})
    t = make_track(frames)
    return t, kinematics.velocities(t)


def test_detects_a_sustained_run_while_a_teammate_has_the_ball():
    t, v = run_scene(n=75)                                  # 3 s at 6 m/s
    runs = offball.detect_runs(t, v, np.ones(len(t), dtype=int))
    assert len(runs) == 1
    r = runs.iloc[0]
    assert r["player"] == "home_1" and r["team"] == "home"
    assert r["duration"] >= 2.5 and r["max_speed"] == pytest.approx(6.0, abs=0.05)
    assert r["distance"] == pytest.approx(6.0 * r["duration"], rel=0.05)


@pytest.mark.parametrize("case", ["too_short", "too_slow", "opponent_ball", "on_the_ball"])
def test_ignores_what_is_not_an_off_ball_run(case):
    kw = {"too_short": dict(n=30), "too_slow": dict(speed=4.5), "opponent_ball": {},
          "on_the_ball": dict(runner_start=(0.0, 20.0), direction=(0.0, 0.0))}[case]
    t, v = run_scene(**kw)   # "on_the_ball": the runner stands on the ball, i.e. he is the carrier
    poss = np.full(len(t), -1 if case == "opponent_ball" else 1)
    assert offball.detect_runs(t, v, poss).empty


def test_segments_bridge_short_dips():
    mask = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    assert offball._segments(mask, bridge=2) == [(0, 7), (13, 14)]


def classify(**kw):
    t, v = run_scene(**kw)
    runs = offball.detect_runs(t, v, np.ones(len(t), dtype=int))
    r = runs.iloc[0].to_dict()
    return offball.classify_run(t, r, 1), t, v, r


def test_run_in_behind_the_defensive_line():
    c, *_ = classify(runner_start=(12.0, -5.0), defenders=((20.0, 0.0), (20.0, -10.0), (50.0, 0.0)))
    assert c["category"] == "Behind defensive line"


def test_overlap_and_underlap_around_the_carrier():
    over, *_ = classify(runner_start=(0.0, 25.0), ball=(10.0, 20.0))
    under, *_ = classify(runner_start=(0.0, 12.0), ball=(10.0, 20.0))
    assert over["category"] == under["category"] == "Half-space underlap/overlap"
    assert (over["subtype"], under["subtype"]) == ("overlap", "underlap")
    assert over["carrier"] == "home_2"


def test_dragging_a_marker():
    c, t, v, r = classify(runner_start=(0.0, -25.0), direction=(0.0, 1.0), ball=(-20.0, 25.0),
                          marker=(2.0, -25.0))
    assert c["category"] == "Dragging defender / space creation"
    assert c["marker"] is not None and c["subtype"] == "dragged marker"
    area, epv = offball.space_created(t, v, r, c["marker"], 1, None)
    assert np.isfinite(area) and epv == 0.0


def test_space_created_is_zero_without_movement():
    t, v = run_scene(n=60, speed=0.0)
    r = {"team": "home", "player": "home_1", "start": 0, "end": 59}
    area, _ = offball.space_created(t, v, r, None, 1, None)
    assert area == pytest.approx(0.0, abs=1e-9)


def test_off_ball_runs_on_a_match(synthetic_match):
    from symmetrica import labels
    from symmetrica.pitch import attack_direction
    track, ev = synthetic_match
    poss = labels.possession_labels(ev, track.index, track["period"])
    runs = offball.off_ball_runs(track, kinematics.velocities(track), poss, attack_direction(track))
    assert len(runs) > 0 and set(runs["category"]) <= set(offball.CATEGORIES)
    assert (runs["duration"] >= 1.5).all() and (runs["max_speed"] > 5.0).all()
    s = offball.runs_summary(runs)
    assert s["home"]["runs"] + s["away"]["runs"] == len(runs)
    assert sum(s["home"]["by_category"].values()) == s["home"]["runs"]


# ---------- shape ----------

def test_convex_hull_and_area():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [2, 8]], float)
    h = shape.convex_hull(square)
    assert len(h) == 4 and shape.polygon_area(h) == pytest.approx(100.0)
    assert shape.polygon_area(np.array([[0, 0], [4, 0], [0, 3]], float)) == pytest.approx(6.0)
    assert shape.polygon_area(shape.convex_hull(np.array([[0, 0], [1, 1]], float))) == 0.0
    m = shape.shape_metrics(square)
    assert (m["width"], m["depth"]) == (10.0, 10.0) and m["area"] == pytest.approx(100.0)


@pytest.mark.parametrize("formation", shape.FORMATIONS)
def test_every_template_recognises_itself(formation):
    pts, roles = shape.template(formation)
    assert len(pts) == len(roles) == 10
    assert shape.match_formation(pts)["formation"] == formation


@pytest.mark.parametrize("formation", ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2"])
def test_formation_survives_scaling_and_noise(formation):
    rng = np.random.default_rng(7)
    pts, roles = shape.template(formation)
    noisy = pts * [18.0, 22.0] + rng.normal(0, 1.0, pts.shape)
    m = shape.match_formation(noisy)
    assert m["formation"] == formation
    assert sorted(m["roles"]) == sorted(roles)


def test_roles_follow_position():
    pts, _ = shape.template("4-3-3")
    m = shape.match_formation(pts * [20, 20])
    by_pos = dict(zip(map(tuple, pts), m["roles"]))
    assert by_pos[tuple(pts[0])] == "D-L"          # back line, y > 0 = left when attacking +x
    assert by_pos[tuple(pts[-1])].startswith("F-")


def test_shape_and_formations_on_a_match(synthetic_match):
    from symmetrica import labels
    from symmetrica.pitch import attack_direction
    track, ev = synthetic_match
    d = attack_direction(track)
    poss = labels.possession_labels(ev, track.index, track["period"])
    keepers = shape.goalkeepers(track, d)
    assert keepers == {"home": {"home_1"}, "away": {"away_12"}}
    s = shape.shape_series(track, poss, d, keepers=keepers)
    assert set(s["phase"]) <= {"in", "out", "dead"} and (s["area"] > 0).all()
    assert (s["width"] <= 68).all() and (s["height"].between(0, 105)).all()
    summ = shape.shape_summary(s)
    assert summ["home"]["out"]["area"] > 0
    forms = shape.formations(track, poss, d, keepers)
    # The generator plays a 4-4-2.
    assert forms["home"]["out"]["formation"] == forms["away"]["out"]["formation"] == "4-4-2"
    roles = shape.roles(forms, "home")
    assert roles["home_1"] == "GK" and len(roles) == 11
    assert len(shape.shape_timeline(s)) == 3


# ---------- players ----------

def test_physical_metrics_from_constant_speed():
    n = 20 * FPS
    frames = [{"home": [(-50 + 5 * k / FPS, 0), (-50 + 6 * k / FPS, 10), (-50 + 8 * k / FPS, -10)],
               "away": [(30, 0)], "ball": (0, 0)} for k in range(n)]
    t = make_track(frames)
    ph = players.physical(t, kinematics.velocities(t)).set_index("id")
    assert ph.loc["home_1", "distance_km"] == pytest.approx(0.1, rel=0.02)
    assert ph.loc["home_1", "hsr_m"] == 0 and ph.loc["home_2", "hsr_m"] == pytest.approx(120, rel=0.03)
    assert ph.loc["home_3", "sprint_m"] == pytest.approx(160, rel=0.03) and ph.loc["home_3", "sprints"] == 1
    assert ph.loc["home_3", "top_speed_kmh"] == pytest.approx(8 * 3.6, rel=0.01)
    assert ph.loc["away_101", "distance_km"] == 0
    assert ph.loc["home_1", "minutes"] == pytest.approx(20 / 60)


def test_pressures_count_spells_near_the_ball_without_it():
    frames = [{"home": [(0, 1 if k < 30 else 20)], "away": [(1, 0)], "ball": (0, 0)} for k in range(100)]
    t = make_track(frames)
    out_of_poss = players.pressures(t, np.full(100, -1))
    in_poss = players.pressures(t, np.full(100, 1))
    assert out_of_poss["home_1"] == 1 and in_poss["home_1"] == 0
    assert in_poss["away_101"] == 1


def test_percentiles_rank_and_exclude_keepers():
    base = {k: 0.0 for k, _ in players.PERCENTILE_METRICS}
    ps = [{**base, "id": f"p{i}", "minutes": 90, "role": "M-C", "xt_p90": float(i)} for i in range(5)]
    ps.append({**base, "id": "gk", "minutes": 90, "role": "GK", "xt_p90": 100.0})
    ps.append({**base, "id": "sub", "minutes": 10, "role": "F-C", "xt_p90": 2.0})
    players.add_percentiles(ps)
    pct = {p["id"]: p["pct"]["xt_p90"] for p in ps}
    assert pct["p0"] < pct["p2"] < pct["p4"] and pct["p2"] == 50.0
    assert pct["gk"] == 100.0 and ps[0]["pool_size"] == 5
    assert ps[-1]["small_sample"] and not ps[0]["small_sample"]


def test_profiles_on_a_match(synthetic_match):
    from symmetrica import labels, threat
    from symmetrica.pitch import attack_direction
    track, ev = synthetic_match
    d = attack_direction(track)
    poss = labels.possession_labels(ev, track.index, track["period"])
    acts = threat.actions(ev, d)
    passes = threat.pass_table(ev, d, threat.ExpectedThreat.fit(acts), track, acts)
    vel = kinematics.velocities(track)
    prof = players.player_profiles(track, vel, ev, poss, d, passes)
    players.add_percentiles(prof, min_minutes=1)
    assert len(prof) == 22
    p = prof[0]
    for key in ("distance_km", "hsr_m", "sprint_m", "top_speed_kmh", "xt", "line_breaking", "packing",
                "pressures", "runs", "heatmap", "pass_map", "pct"):
        assert key in p
    assert np.array(p["heatmap"]).shape == (8, 12) and sum(map(sum, p["heatmap"])) == pytest.approx(1.0, abs=1e-3)
    assert all(0 <= v <= 100 for q in prof for v in q["pct"].values())
    assert players.defensive_actions(make_events([
        ("away", "PASS", "", 1, 1, 5, 15, 16, (0, 0), (5, 0)),
        ("home", "CHALLENGE", "TACKLE-WON", 1, 6, 6, 4, None, (5, 0), None)])) == {"home_4": 1}
