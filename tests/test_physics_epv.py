"""Momentum-aware arrival times, the EPV surface and pass-option evaluation."""
import numpy as np
import pytest

from conftest import DIRECTIONS, make_events, make_track, repeat
from symmetrica import pass_options as po
from symmetrica import pitch_control as pc
from symmetrica import threat
from symmetrica.kinematics import FrameState
from symmetrica.pitch import Grid

A_MAX, V_MAX = 7.0, 7.8
ALPHA = A_MAX / V_MAX


def arrive(target, vel=(0.0, 0.0), pos=(0.0, 0.0), reaction=0.0):
    return pc.physics_arrival_time(np.array([pos], float), np.array([vel], float), np.array([target], float),
                                   A_MAX, V_MAX, reaction)[0, 0]


def distance_from_rest(t):
    """Closed form of the drag-limited model: distance covered from standing after t seconds."""
    return V_MAX * (t - (1 - np.exp(-ALPHA * t)) / ALPHA)


@pytest.mark.parametrize("d", [2.0, 10.0, 20.0, 40.0])
def test_from_rest_matches_the_closed_form(d):
    t = arrive((d, 0.0))
    assert distance_from_rest(t) == pytest.approx(d, abs=0.01)


def test_from_rest_is_isotropic_and_starts_with_a_max():
    times = [arrive((10 * np.cos(a), 10 * np.sin(a))) for a in np.linspace(0, 2 * np.pi, 7)]
    assert np.ptp(times) < 1e-3
    # Over a short distance the player is still accelerating at ~a_max: d = a t^2 / 2.
    assert arrive((0.1, 0.0)) == pytest.approx(np.sqrt(2 * 0.1 / A_MAX), rel=0.03)
    # Far away the player runs at top speed after a fixed acceleration lag of 1 / alpha.
    assert arrive((100.0, 0.0)) == pytest.approx(100 / V_MAX + 1 / ALPHA, rel=0.01)


def test_momentum_makes_turning_round_expensive():
    sprint = (7.0, 0.0)
    ahead, behind, rest = arrive((10, 0), sprint), arrive((-10, 0), sprint), arrive((10, 0))
    side = arrive((0, 10), sprint)
    assert ahead < rest < side < behind
    assert behind - rest > 0.8          # braking + turning costs close to a second
    # The linear (Spearman) model only shifts the start point during the reaction time:
    lin = pc.PitchControlParams(motion="linear")
    t_lin = pc.time_to_intercept(np.zeros((1, 2)), np.array([sprint]), np.array([[10.0, 0], [-10.0, 0]]), lin)[:, 0]
    assert t_lin[1] - t_lin[0] == pytest.approx(2 * 7.0 * 0.7 / 5.0)


def test_first_arrival_is_found_for_a_runner_passing_through():
    # A sprinter 3 m short of a point he is running at reaches it in ~0.4 s even though,
    # without stopping, he is past it shortly afterwards (later roots of the reach equation).
    t = arrive((3.0, 0.0), vel=(7.0, 0.0))
    assert t == pytest.approx(3.0 / 7.0, abs=0.06)


def test_reach_shrinks_the_distance_to_cover():
    with_reach = pc.physics_arrival_time(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[10.0, 0.0]]),
                                         A_MAX, V_MAX, 0.0, reach=1.0)[0, 0]
    assert with_reach == pytest.approx(arrive((9.0, 0.0)), abs=1e-3)
    inside = pc.physics_arrival_time(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0.5, 0.0]]),
                                     A_MAX, V_MAX, 0.25, reach=1.0)[0, 0]
    assert inside == pytest.approx(0.25)


def test_reaction_time_coasts_on_current_velocity():
    assert arrive((10, 0), vel=(4.0, 0.0), reaction=0.25) == pytest.approx(
        0.25 + arrive((9.0, 0), vel=(4.0, 0.0)), abs=1e-3)
    assert arrive((0.0, 0.0)) == 0.0


def test_physics_pitch_control_favours_the_player_who_can_stop():
    # Equidistant from the target, the defender is sprinting away from it.
    att, dfn = pc.pitch_control_at(np.array([[0.0, 0.0]]), np.zeros(2), np.array([[-10.0, 0.0]]), np.zeros((1, 2)),
                                   np.array([[10.0, 0.0]]), np.array([[7.0, 0.0]]))
    assert att[0] / (att[0] + dfn[0]) > 0.8
    with pytest.raises(ValueError):
        pc.time_to_intercept(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), pc.PitchControlParams(motion="x"))


def rising_xt(nx=12, ny=8):
    """xT that grows towards the attacked goal (+x) and is flat across the pitch."""
    vals = np.tile(np.linspace(0.001, 0.2, nx), (ny, 1))
    z = np.zeros_like(vals)
    return threat.ExpectedThreat(vals, z, z, z, 1)


def test_xt_interpolation_hits_cell_centres_and_is_linear_between():
    xt = rising_xt()
    cx = -52.5 + 105 / 12 * (np.arange(12) + 0.5)
    assert xt.interpolate(cx, np.zeros(12)) == pytest.approx(xt.values[0])
    mid = xt.interpolate((cx[3] + cx[4]) / 2, 0.0)
    assert mid == pytest.approx((xt.values[0, 3] + xt.values[0, 4]) / 2)
    assert xt.interpolate(-60.0, 0.0) == pytest.approx(xt.values[0, 0])   # clamped at the edge
    assert np.isnan(xt.interpolate(np.nan, 0.0))


def test_epv_is_control_times_threat_in_the_attacking_frame():
    xt, g = rising_xt(), Grid(12, 8)
    ctrl = np.full(g.ny * g.nx, 0.5)
    ctrl[:5] = 0.0
    epv = threat.epv_surface(ctrl, g.cells, xt, direction=1)
    assert epv == pytest.approx(ctrl * xt.interpolate(g.cells[:, 0], g.cells[:, 1]))
    assert np.all(epv[:5] == 0)
    # Attacking towards -x the same control is worth most on the left.
    flipped = threat.epv_surface(np.full(g.ny * g.nx, 0.5), g.cells, xt, direction=-1).reshape(g.ny, g.nx)
    assert flipped[:, 0].mean() > flipped[:, -1].mean()
    assert np.argmax(epv) % g.nx == g.nx - 1


def state(home, away, ball, home_vel=None, away_vel=None):
    home, away = np.array(home, float), np.array(away, float)
    return FrameState([f"home_{i + 1}" for i in range(len(home))], home,
                      np.zeros_like(home) if home_vel is None else np.array(home_vel, float),
                      [f"away_{i + 101}" for i in range(len(away))], away,
                      np.zeros_like(away) if away_vel is None else np.array(away_vel, float), np.array(ball, float))


def test_interception_risk():
    lane = np.array([[10.0, 0.0]])
    on_lane = po.interception_risk(lane, np.zeros(2), np.array([[10.0, 0.5]]), np.zeros((1, 2)))
    near = po.interception_risk(lane, np.zeros(2), np.array([[10.0, 5.0]]), np.zeros((1, 2)))
    far = po.interception_risk(lane, np.zeros(2), np.array([[10.0, 30.0]]), np.zeros((1, 2)))
    # Within reach he only needs his 0.25 s reaction against the ball's 0.67 s of flight.
    assert on_lane[0] > 0.8 > near[0] > far[0] and far[0] < 0.01
    two = po.interception_risk(lane, np.zeros(2), np.array([[10.0, 5.0], [10.0, -5.0]]), np.zeros((2, 2)))
    assert two[0] == pytest.approx(1 - (1 - near[0]) ** 2)  # independent chances combine
    assert po.interception_risk(lane, np.zeros(2), np.empty((0, 2)), np.empty((0, 2)))[0] == 0


def test_pass_options_rank_open_forward_teammates_first():
    # Passer #1 on the ball at (0, 0). #2 open 20 m ahead, #3 ahead but behind a defender on the lane,
    # #4 open but 15 m backwards.
    s = state(home=[(0, 0), (20, 0), (20, 20), (-15, 0)],
              away=[(10, 10.5), (45, -25)], ball=(0, 0))
    opts = po.evaluate_options(s, "home", "home_1", 1, rising_xt())
    by = {o["number"]: o for o in opts}
    assert set(by) == {2, 3, 4}
    assert opts[0]["number"] == 2
    assert by[2]["completion"] > by[3]["completion"]
    assert by[3]["lane_risk"] > by[2]["lane_risk"]
    assert by[4]["gain"] < 0 < by[2]["gain"]
    assert all(0 <= o["completion"] <= 1 for o in opts)
    assert po.evaluate_options(s, "home", "home_1", 1, rising_xt(), pc.PitchControlParams())[0] == opts[0]
    no_ball = state(home=[(0, 0), (5, 0)], away=[(9, 9)], ball=(np.nan, np.nan))
    assert po.evaluate_options(no_ball, "home", "home_1", 1, rising_xt()) == []


def test_pass_decisions_table():
    frame = {"home": [(0, 0), (20, 0), (20, 20)], "away": [(10, 10.5), (45, -25)], "ball": (0, 0)}
    t = make_track(repeat(frame, 30))
    ev = make_events([("home", "PASS", "", 1, 5, 15, 1, 3, (0, 0), (20, 20)),
                      ("home", "PASS", "", 1, 20, 28, 1, 2, (0, 0), (20, 0))])
    dec = po.pass_decisions(ev, t, None, DIRECTIONS, rising_xt())
    assert len(dec) == 2
    assert dec["chose_best"].tolist() == [False, True]
    assert (dec["rank"] == 1).tolist() == dec["chose_best"].tolist()
    assert (dec["best_gain"] >= dec["actual_gain"]).all()
    s = po.decision_summary(dec)
    assert s["home"]["best_option_share"] == 0.5 and s["away"]["passes_evaluated"] == 0
