"""Pitch control (Spearman) and Voronoi space dominance."""
import numpy as np
import pytest

from conftest import DIRECTIONS, make_track, repeat
from symmetrica import pitch_control as pc
from symmetrica.kinematics import FrameState
from symmetrica.pitch import Grid

P = pc.PitchControlParams()
Z = np.zeros((1, 2))


def one_v_one(home=(-10.0, 0.0), away=(10.0, 0.0), ball=(0.0, 0.0), home_vel=(0, 0), away_vel=(0, 0)):
    return FrameState(["home_1"], np.array([home], float), np.array([home_vel], float),
                      ["away_101"], np.array([away], float), np.array([away_vel], float), np.array(ball, float))


def test_time_to_intercept_linear_formula():
    linear = pc.PitchControlParams(motion="linear")
    tau = pc.time_to_intercept(np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]]), np.array([[10.0, 0.0]]), linear)
    # Drifts 1.4 m during the 0.7 s reaction, then covers the remaining 8.6 m at 5 m/s.
    assert tau[0, 0] == pytest.approx(0.7 + 8.6 / 5.0)


def test_probabilities_sum_to_one_everywhere():
    s = pc.pitch_control(one_v_one(), Grid(30, 20))
    total = s.home + s.away
    assert total.min() > 1 - P.tolerance - 1e-9
    assert total.max() < 1 + 1e-9
    assert np.all(s.home >= 0) and np.all(s.away >= 0)


def test_mirror_symmetry_and_midline_is_contested():
    s = pc.pitch_control(one_v_one(), Grid(30, 20))
    assert s.home == pytest.approx(s.away[:, ::-1], abs=1e-9)
    assert s.home_share == pytest.approx(0.5, abs=1e-9)
    mid = pc.pitch_control_at(np.array([[0.0, 0.0], [0.0, 15.0]]), np.zeros(2),
                              np.array([[-10.0, 0.0]]), Z, np.array([[10.0, 0.0]]), Z)
    # Equal arrival times: an exact split (raw values sum to 1 - tolerance, the integration's stop rule).
    assert mid[0] == pytest.approx(mid[1], abs=1e-12)
    assert mid[0] / (mid[0] + mid[1]) == pytest.approx([0.5, 0.5])


def test_a_player_controls_the_ground_at_his_feet():
    att, dfn = pc.pitch_control_at(np.array([[-10.0, 0.0], [10.0, 0.0]]), np.zeros(2),
                                   np.array([[-10.0, 0.0]]), Z, np.array([[10.0, 0.0]]), Z)
    assert att[0] > 0.95 and dfn[1] > 0.95


def test_running_towards_a_space_wins_it():
    target = np.array([[0.0, 20.0]])
    standing = [v[0] for v in pc.pitch_control_at(target, np.zeros(2), np.array([[-10.0, 0.0]]), Z,
                                                  np.array([[10.0, 0.0]]), Z)]
    running = pc.pitch_control_at(target, np.zeros(2), np.array([[-10.0, 0.0]]), np.array([[3.0, 5.0]]),
                                  np.array([[10.0, 0.0]]), Z)[0]
    assert standing[0] / (standing[0] + standing[1]) == pytest.approx(0.5)
    assert running[0] > 0.8


def test_defender_advantage_kappa():
    params = pc.PitchControlParams(kappa_def=1.72)
    att, _ = pc.pitch_control_at(np.array([[0.0, 0.0]]), np.zeros(2), np.array([[-10.0, 0.0]]), Z,
                                 np.array([[10.0, 0.0]]), Z, params)
    assert att[0] < 0.45


def test_empty_team_cedes_the_pitch():
    att, dfn = pc.pitch_control_at(Grid(5, 4).cells, np.zeros(2), np.array([[0.0, 0.0]]), Z,
                                   np.empty((0, 2)), np.empty((0, 2)))
    assert att == pytest.approx(1.0, abs=P.tolerance)
    assert np.all(dfn == 0)


def test_voronoi_matches_brute_force():
    rng = np.random.default_rng(0)
    home, away = rng.uniform(-40, 40, (5, 2)), rng.uniform(-40, 40, (6, 2))
    g = Grid(21, 14)
    v = pc.voronoi(home, away, g)
    for (x, y), val in zip(g.cells, v.ravel()):
        dh = min(np.hypot(*(home - (x, y)).T))
        da = min(np.hypot(*(away - (x, y)).T))
        assert val == (1 if dh < da else -1)


def test_voronoi_series_shares_and_attacking_frame():
    # Period 1: home owns the left half (its defensive half, home attacks +x).
    frame = {"home": [(-30, 0), (-30, 20), (-30, -20)], "away": [(30, 0), (30, 20), (30, -20)], "ball": (0, 0)}
    t = make_track(repeat(frame, 10, period=1) + repeat(frame, 10, period=2))
    series, mean_map = pc.voronoi_series(t, DIRECTIONS, Grid(20, 10), step=1)
    assert series["home_share"].to_numpy() == pytest.approx(0.5)
    # Period 1 home attacks right, where away is: it controls none of that final third.
    p1, p2 = series.iloc[:10], series.iloc[10:]
    assert p1["home_final_third"].to_numpy() == pytest.approx(0.0)
    assert p1["away_final_third"].to_numpy() == pytest.approx(0.0)
    # Period 2 the ends swap but the players do not: both teams now own their final third.
    assert p2["home_final_third"].to_numpy() == pytest.approx(1.0)
    # In the home attacking frame the left half is home's in P1 and the right half in P2: average 0.5.
    assert mean_map == pytest.approx(0.5)


def test_pitch_control_series_skips_frames_without_ball():
    frame = {"home": [(-10, 0)], "away": [(10, 0)], "ball": (0, 0)}
    t = make_track(repeat(frame, 3) + [{**frame, "ball": (np.nan, np.nan)}])
    s = pc.pitch_control_series(t, None, np.array([0, 3]), Grid(10, 6))
    assert s["home_share"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert np.isnan(s["home_share"].iloc[1])
