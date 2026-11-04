"""Possession chains, PPDA, counter-pressing, high turnovers and tracking pressure."""
import numpy as np
import pytest

from conftest import DIRECTIONS, make_events, make_track
from symmetrica import pressing


def feed():
    """Home builds up, away wins it high and loses it, home counter-presses, ball goes out."""
    return make_events([
        ("away", "SET PIECE", "KICK OFF", 1, 1, 1, 19, None, (0, 0), None),
        ("away", "PASS", "", 1, 1, 20, 19, 21, (0, 0), (-10, 5)),
        ("away", "BALL LOST", "INTERCEPTION", 1, 30, 40, 21, None, (-10, 5), (-20, 0)),
        ("home", "RECOVERY", "INTERCEPTION", 1, 40, 40, 4, None, (-20, 0), None),           # turnover 1
        ("home", "PASS", "", 1, 45, 60, 4, 8, (-20, 0), (0, 10)),
        ("away", "CHALLENGE", "TACKLE-WON", 1, 70, 70, 15, None, (0, 10), None),
        ("home", "CHALLENGE", "TACKLE-LOST", 1, 70, 70, 8, None, (0, 10), None),
        ("home", "BALL LOST", "THEFT", 1, 70, 70, 8, None, (0, 10), None),
        ("away", "RECOVERY", "THEFT", 1, 70, 70, 15, None, (0, 10), None),                    # turnover 2
        ("home", "CHALLENGE", "GROUND-WON", 1, 100, 100, 6, None, (5, 10), None),
        ("home", "RECOVERY", "THEFT", 1, 100, 100, 6, None, (5, 10), None),                   # regained in 1.2 s
        ("home", "PASS", "", 1, 110, 130, 6, 9, (5, 10), (40, 0)),
        ("home", "SHOT", "ON TARGET-GOAL", 1, 140, 150, 9, None, (42, 0), None),
        ("away", "SET PIECE", "KICK OFF", 1, 300, 300, 19, None, (0, 0), None),
        ("away", "PASS", "", 1, 300, 320, 19, 22, (0, 0), (5, 30)),
        ("away", "BALL OUT", "", 1, 330, 330, 22, None, (5, 34), None),
        ("home", "SET PIECE", "THROW IN", 1, 400, 400, 2, None, (5, 34), None),
        ("home", "PASS", "", 1, 400, 410, 2, 7, (5, 34), (10, 20)),
    ])


def test_chains_split_on_team_change_and_stoppages():
    chains, before = pressing.possession_chains(feed())
    assert [c.team for c in chains] == ["away", "home", "away", "home", "away", "home"]
    assert [c.open_play for c in chains] == [False, True, True, True, False, False]
    assert chains[3].goal and chains[3].shot and chains[3].passes == 1
    # The tackle by away happened while home had the ball.
    assert before.iloc[5] == "home"
    # Nobody is in possession after the ball goes out.
    assert before.iloc[16] is None


def test_ppda_counts_only_defensive_actions_in_the_pressing_zone():
    ev = make_events([
        ("away", "PASS", "", 1, 1, 5, 15, 16, (20, 0), (25, 0)),
        ("away", "PASS", "", 1, 10, 15, 16, 17, (25, 0), (30, 0)),
        ("away", "PASS", "", 1, 20, 25, 17, 18, (30, 0), (35, 0)),
        ("away", "PASS", "", 1, 30, 35, 18, 15, (-30, 0), (-25, 0)),     # outside home's pressing zone
        ("home", "CHALLENGE", "AERIAL-WON", 1, 40, 40, 9, None, (30, 0), None),  # aerials don't count
        ("home", "CHALLENGE", "GROUND-LOST", 1, 41, 41, 9, None, (30, 0), None),
        ("away", "PASS", "", 1, 50, 55, 15, 16, (30, 0), (32, 0)),
        ("away", "BALL LOST", "INTERCEPTION", 1, 60, 65, 16, None, (32, 0), (35, 0)),
        ("home", "RECOVERY", "INTERCEPTION", 1, 65, 65, 10, None, (35, 0), None),
    ])
    out = pressing.ppda(ev, DIRECTIONS)
    # Away passes in home's zone (x >= -10.5 in home's attacking frame): 4 completed + 1 intercepted.
    # Home defensive actions there: 1 ground challenge + 1 interception.
    assert out["home"] == pytest.approx(5 / 2)
    assert np.isnan(out["away"])  # away never defended


def test_ppda_timeline_windows():
    tl = pressing.ppda_timeline(feed(), DIRECTIONS, window_minutes=0.1)  # 150-frame windows
    assert list(tl.columns) == ["start_minute", "home", "away"]
    assert tl["start_minute"].iloc[1] == pytest.approx(0.1)


def test_counter_press_success_and_failure():
    tov = pressing.turnovers(feed(), DIRECTIONS, window=5.0)
    assert list(tov["team"]) == ["away", "home", "away"]
    lost_by_home = tov[tov["team"] == "home"].iloc[0]
    assert lost_by_home["regained"]
    assert lost_by_home["seconds_to_regain"] == pytest.approx(30 / 25)
    # Away lost the ball at x = -10 in the global frame, which is +10 in its own attacking frame,
    # and won it straight back with the tackle at frame 70.
    first = tov.iloc[0]
    assert first["x"] == pytest.approx(10) and first["regained"]
    assert first["seconds_to_regain"] == pytest.approx(30 / 25)
    # After the goal away restarts with a kick-off: a set piece is not a counter-press regain.
    assert tov["regained"].tolist() == [True, True, False]
    # Tight window: the 1.2 s regain no longer counts.
    assert not pressing.turnovers(feed(), DIRECTIONS, window=1.0)["regained"].any()


def test_counter_press_pressers_from_tracking():
    ev = feed()
    frames = [{"home": [(1, 10), (3, 10), (40, 0)], "away": [(0, 10)], "ball": (0, 10)} for _ in range(450)]
    tov = pressing.turnovers(ev, DIRECTIONS, track=make_track(frames), radius=10.0)
    assert tov[tov["team"] == "home"]["pressers"].iloc[0] == pytest.approx(2.0)


def test_recoveries_high_turnovers_and_zones():
    rec = pressing.recoveries(feed(), DIRECTIONS)
    home = rec[rec["team"] == "home"]
    # Home won it at (-20, 0) (defensive third) and at (5, 10): neither is within 40 m of goal.
    assert not home["high"].any()
    assert home["shot"].tolist() == [False, True]
    away = rec[rec["team"] == "away"].iloc[0]
    # Away attacks -x: (0, 10) becomes (0, -10) in its frame, middle third, centre channel.
    assert (away["x"], away["y"]) == pytest.approx((0.0, -10.0))
    assert (away["third"], away["channel"]) == (1, 1)
    z = pressing.recovery_zones(rec, "home")
    assert z.sum() == 2 and z[0, 1] == 1 and z[1, 1] == 1

    high = pressing.recoveries(make_events([
        ("away", "PASS", "", 1, 1, 5, 15, 16, (30, 0), (35, 0)),
        ("home", "RECOVERY", "INTERCEPTION", 1, 10, 10, 9, None, (40, 0), None),
    ]), DIRECTIONS)
    assert high[high["team"] == "home"]["high"].item()


def test_pressure_series_counts_only_the_defending_team():
    frames = [{"home": [(0, 0), (2, 0), (20, 0)], "away": [(1, 1), (30, 0)], "ball": (0, 0)} for _ in range(4)]
    t = make_track(frames)
    poss = np.array([1, 1, -1, 0])
    p = pressing.pressure_series(t, poss, radius=5.0)
    assert p["away_pressers"].tolist()[:2] == [1.0, 1.0]
    assert np.isnan(p["away_pressers"].iloc[2]) and np.isnan(p["home_pressers"].iloc[0])
    assert p["home_pressers"].iloc[2] == 2.0
    assert p["away_nearest"].iloc[0] == pytest.approx(np.sqrt(2))
    assert p.iloc[3].isna().all()


def test_pressing_summary_keys(synthetic_match):
    from symmetrica import labels
    from symmetrica.pitch import attack_direction
    track, ev = synthetic_match
    poss = labels.possession_labels(ev, track.index, track["period"])
    s = pressing.pressing_summary(ev, attack_direction(track), track, poss)
    for team in ("home", "away"):
        row = s[team]
        assert 0 <= row["counterpress_rate"] <= 1
        assert row["counterpress_regains"] <= row["turnovers"]
        assert row["high_turnovers"] <= row["recoveries"]
        assert np.array(row["recovery_zones"]).sum() == row["recoveries"]
        assert row["ppda"] > 0
        assert row["nearest_presser_m"] >= 0
