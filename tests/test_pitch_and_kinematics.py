"""Geometry helpers, attacking direction, velocities and frame snapshots."""
import numpy as np
import pytest

from conftest import make_track, repeat
from symmetrica import kinematics, pitch
from symmetrica.data import FPS, PITCH_LENGTH, PITCH_WIDTH


def test_grid_covers_the_pitch():
    g = pitch.Grid(10, 4)
    assert g.cells.shape == (40, 2)
    assert g.cell_area * 40 == pytest.approx(PITCH_LENGTH * PITCH_WIDTH)
    assert g.xs[0] == pytest.approx(-52.5 + 5.25) and g.ys[-1] == pytest.approx(34 - 8.5)
    # Row-major with y slow: the first nx cells share the lowest y.
    assert np.all(g.cells[:10, 1] == g.ys[0])


def test_attack_direction_from_kickoff_shape():
    home_left = {"home": [(-20, 0), (-5, 10)], "away": [(20, 0), (5, -10)], "ball": (0, 0)}
    frames = repeat(home_left, 10, period=1) + repeat(
        {"home": [(20, 0), (5, 10)], "away": [(-20, 0), (-5, -10)], "ball": (0, 0)}, 10, period=2)
    t = make_track(frames)
    d = pitch.attack_direction(t)
    assert d == {"home": {1: 1, 2: -1}, "away": {1: -1, 2: 1}}
    assert all(isinstance(k, int) for k in d["home"])
    per_frame = pitch.direction_per_frame(t, "away", d)
    assert per_frame[0] == -1 and per_frame[-1] == 1


def test_goal_geometry():
    assert pitch.distance_to_goal(41.5, 0.0) == pytest.approx(11.0)  # penalty spot
    near, far = pitch.goal_angle(np.array([47.0, 30.0]), np.array([0.0, 0.0]))
    assert near > far > 0
    assert pitch.goal_angle(40.0, 5.0) == pytest.approx(pitch.goal_angle(40.0, -5.0))
    # From the goal line itself, wide of the post, the goal subtends nothing.
    assert pitch.goal_angle(52.5, 10.0) == pytest.approx(0.0, abs=1e-3)


def test_zones():
    third, channel = pitch.zone_of(np.array([-50, 0, 50]), np.array([30, 0, -30]))
    assert third.tolist() == [0, 1, 2]
    assert channel.tolist() == [0, 1, 2]


def test_velocity_recovers_constant_motion_and_respects_halves():
    # Home player runs at 4 m/s along x in period 1, then the second half starts elsewhere.
    frames = [{"home": [(-10 + 4 * k / FPS, 0)], "away": [(20, 0)], "ball": (0, 0), "period": 1} for k in range(50)]
    frames += [{"home": [(30, 0)], "away": [(20, 0)], "ball": (0, 0), "period": 2} for _ in range(20)]
    t = make_track(frames)
    v = kinematics.velocities(t)
    assert v["home_1_vx"].iloc[10:40].to_numpy() == pytest.approx(4.0, abs=1e-6)
    assert v["home_1_vy"].iloc[10:40].to_numpy() == pytest.approx(0.0, abs=1e-6)
    # No 40 m "jump" across half time: second-half velocity is zero, not a huge spike.
    assert np.nanmax(np.abs(v["home_1_vx"].iloc[50:])) == pytest.approx(0.0, abs=1e-6)
    assert v["away_101_vx"].abs().max() == pytest.approx(0.0)


def test_velocity_discards_impossible_speeds():
    frames = [{"home": [(0, 0) if k < 10 else (30, 0)], "away": [(5, 5)], "ball": (0, 0)} for k in range(20)]
    v = kinematics.velocities(make_track(frames), window=1)
    assert np.isnan(v["home_1_vx"].iloc[10])  # a 30 m teleport is tracking noise, not a sprint


def test_frame_state_drops_players_off_the_pitch():
    t = make_track([{"home": [(1, 2), None, (3, 4)], "away": [(5, 6)], "ball": (0, 1)}])
    s = kinematics.frame_state(t, None, 0)
    assert s.home_ids == ["home_1", "home_3"]
    assert s.home_pos.tolist() == [[1, 2], [3, 4]]
    assert s.home_vel.tolist() == [[0, 0], [0, 0]]
    assert s.ball.tolist() == [0, 1]
    ids, pos, vel = s.team("away")
    assert ids == ["away_101"] and pos.shape == (1, 2)
