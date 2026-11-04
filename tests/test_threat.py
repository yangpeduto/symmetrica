"""Expected threat, carries, progressive passes, defensive lines and packing."""
import numpy as np
import pandas as pd
import pytest

from conftest import DIRECTIONS, make_events, make_track
from symmetrica import threat


def test_geometric_xg_is_sensible():
    xs = np.array([47.0, 41.5, 36.0, 27.5])  # 5.5 m, penalty spot, edge of box, 25 m
    xg = threat.geometric_xg(xs, np.zeros(4))
    assert np.all(np.diff(xg) < 0)
    assert 0.15 < xg[1] < 0.35 and xg[3] < 0.06
    assert threat.geometric_xg(40.0, 8.0) == pytest.approx(threat.geometric_xg(40.0, -8.0))
    assert threat.geometric_xg(40.0, 25.0) < threat.geometric_xg(40.0, 0.0)


def _acts(rows):
    cols = ["team", "period", "frame", "end_frame", "kind", "player", "receiver",
            "x", "y", "end_x", "end_y", "success", "goal"]
    return pd.DataFrame([dict(zip(cols, ["home", 1, 0, 0] + list(r))) for r in rows])


def test_value_iteration_on_a_designed_chain():
    # 4 x 2 grid (cells 26.25 x 34 m). Cell A (final quarter) only ever shoots.
    # Cell B only passes successfully into A. Cell C passes into B, half of them fail.
    A, B, C = (40.0, 10.0), (15.0, 10.0), (-10.0, 10.0)
    rows = [("shot", "p", None, *A, np.nan, np.nan, True, False)] * 5
    rows += [("pass", "p", "q", *B, *A, True, False)] * 4
    rows += [("pass", "p", "q", *C, *B, True, False)] * 3 + [("lost", "p", None, *C, *A, False, False)] * 3
    xt = threat.ExpectedThreat.fit(_acts(rows), nx=4, ny=2, bandwidth=0)
    vA, vB, vC = xt.value(*np.array([A, B, C]).T)
    xg_A = xt.xg.ravel()[np.argmax(xt.shot_prob.ravel())]
    assert vA == pytest.approx(xg_A, rel=1e-6)
    assert vB == pytest.approx(vA, rel=1e-6)
    assert vC == pytest.approx(0.5 * vB, rel=1e-6)
    # Cells with no actions are worth nothing; unknown locations are NaN.
    assert xt.value(-40.0, -10.0)[()] == 0.0
    assert np.isnan(xt.value(np.nan, 0.0)[()])


def test_xt_fitted_on_a_match(synthetic_match):
    track, ev = synthetic_match
    from symmetrica.pitch import attack_direction
    acts = threat.actions(ev, attack_direction(track))
    xt = threat.ExpectedThreat.fit(acts)
    assert xt.values.shape == (8, 12)
    assert np.all((xt.values >= 0) & (xt.values <= 1))
    assert xt.iterations < 500
    # Threat lives in the final third: the best final-third cell beats the best own-third cell.
    assert xt.values[:, 8:].max() > 3 * xt.values[:, :4].max()
    assert set(acts["kind"]) <= {"pass", "carry", "lost", "shot"}


def test_carries_are_inferred_between_touches():
    ev = make_events([
        ("home", "PASS", "", 1, 10, 20, 4, 8, (-20, 0), (0, 0)),
        ("home", "PASS", "", 1, 60, 70, 8, 9, (12, 5), (30, 5)),     # #8 ran 13 m with the ball
        ("home", "PASS", "", 1, 80, 90, 9, 10, (31, 5), (40, 0)),    # #9 moved 1 m: just a touch
    ])
    acts = threat.actions(ev, DIRECTIONS)
    carries = acts[acts["kind"] == "carry"]
    assert len(carries) == 1
    c = carries.iloc[0]
    assert c["player"] == "home_8" and (c["x"], c["end_x"], c["end_y"]) == pytest.approx((0, 12, 5))
    assert (c["frame"], c["end_frame"]) == (20, 60)


def test_actions_use_the_attacking_frame():
    ev = make_events([("home", "PASS", "", 2, 10, 20, 4, 8, (20, 10), (0, 0))])  # home attacks -x in P2
    a = threat.actions(ev, DIRECTIONS).iloc[0]
    assert (a["x"], a["y"], a["end_x"]) == pytest.approx((-20, -10, 0))


@pytest.mark.parametrize("start,end,expected", [
    ((-40, 0), (-5, 0), True),     # own half, 35 m closer
    ((-40, 0), (-15, 0), False),   # own half, only 25 m
    ((-10, 0), (8, 0), True),      # crosses halfway, 18 m
    ((10, 0), (21, 0), True),      # opponent half, 11 m
    ((10, 0), (18, 0), False),     # opponent half, 8 m
    ((30, 0), (10, 0), False),     # backwards
])
def test_progressive_rule(start, end, expected):
    assert bool(threat.is_progressive(np.array([start[0]]), np.array([start[1]]),
                                      np.array([end[0]]), np.array([end[1]]))[0]) is expected


def test_defensive_lines_of_a_442():
    rng = np.random.default_rng(1)
    shape = np.r_[np.full(2, 0.0), np.full(4, 15.0), np.full(4, 30.0)] + rng.uniform(-1, 1, 10)
    lines = threat.defensive_lines(shape)
    assert lines == pytest.approx([0, 15, 30], abs=1.0)
    assert threat.defensive_lines(np.array([5.0])) == pytest.approx([5.0])
    assert len(threat.defensive_lines(np.array([]))) == 0


def test_lines_broken_and_packing():
    lines = np.array([0.0, 15.0, 30.0])
    assert threat.lines_broken(-5, 20, lines) == 2
    assert threat.lines_broken(-1, 20, lines) == 1           # starts only 1 m behind the first line
    assert threat.lines_broken(-5, 31, lines) == 2           # arrives only 1 m beyond the back line
    assert threat.lines_broken(20, -5, lines) == 0
    opp_before = np.array([0, 5, 15, 16, 30, 45])
    opp_after = np.array([0, 5, 15, 16, 30, 45])
    assert threat.packing(opp_before, -2, opp_after, 20) == 4   # took out the 4 players at x <= 16
    assert threat.packing(opp_before, 20, opp_after, -2) == 0   # never negative


def test_pass_table_with_tracking():
    # Home attacks +x. Away: keeper at 50, back line at 20, midfield at 5, a striker at -15.
    away = [(50, 0), (20, -15), (20, -5), (20, 5), (20, 15), (5, -10), (5, 10), (-15, 0)]
    frames = [{"home": [(-20, 0), (25, 0)], "away": away, "ball": (-20, 0)} for _ in range(40)]
    t = make_track(frames)
    ev = make_events([("home", "PASS", "", 1, 5, 30, 1, 2, (-20, 0), (25, 0))])
    acts = threat.actions(ev, DIRECTIONS)
    xt = threat.ExpectedThreat.fit(pd.concat([acts, _acts([("shot", "home_2", None, 45.0, 0.0,
                                                            np.nan, np.nan, True, True)])]))
    p = threat.pass_table(ev, DIRECTIONS, xt, t, acts).iloc[0]
    assert p["lines_broken"] == 3          # striker, midfield and back line
    assert p["packing"] == 7               # all outfielders; the keeper stays goal-side
    assert p["progressive"] and p["metres_gained"] == pytest.approx(45)
    assert p["xt_added"] == pytest.approx(p["xt_end"] - p["xt_start"])
    assert p["line_breaking"]

    board = threat.player_threat(threat.pass_table(ev, DIRECTIONS, xt, t, acts))
    assert board.iloc[0]["player"] == "home_1" and board.iloc[0]["line_breaking"] == 1
    summary = threat.threat_summary(threat.pass_table(ev, DIRECTIONS, xt, t, acts), acts)
    assert summary["home"]["line_breaking_passes"] == 1 and summary["away"]["passes"] == 0
