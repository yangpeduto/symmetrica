"""Momentum timeline, passing networks, the possession baseline and the insight rules."""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import DIRECTIONS, make_events, make_track
from symmetrica import insights, labels, models, momentum, networks
from symmetrica.data import FPS

MIN = 60 * FPS


def test_momentum_follows_threat():
    passes = pd.DataFrame({"team": ["home", "away", "away"], "frame": [1 + 2 * MIN, 1 + 6 * MIN, 1 + 6 * MIN + 5],
                           "xt_added": [0.2, 0.1, -0.5]})
    track = make_track([{"home": [(0, 0)], "away": [(5, 0)], "ball": (0, 0)}] * (8 * MIN))
    poss = np.where(np.arange(len(track)) < 4 * MIN, 1, -1)
    tl = momentum.momentum_timeline(passes, track, poss, span=1)
    assert len(tl) == 8
    assert tl.loc[2, "home_xt"] == pytest.approx(0.2)
    assert tl.loc[6, "away_xt"] == pytest.approx(0.1)   # negative xT added is not "threat created"
    assert tl.loc[2, "momentum"] == pytest.approx(1.0)  # scaled to the biggest swing
    assert tl.loc[6, "momentum"] < 0
    assert tl.loc[1, "home_possession"] == 1.0 and tl.loc[5, "home_possession"] == 0.0
    start, value = momentum.best_spell(tl, "away", length=2)
    assert start in (5, 6) and value == pytest.approx(0.1)


def test_passing_network_uses_average_position_in_possession():
    ev = make_events([("home", "PASS", "", 1, 1 + i, 2 + i, 1, 2, (0, 0), (5, 0)) for i in range(4)] +
                     [("home", "PASS", "", 1, 10, 11, 2, 1, (5, 0), (0, 0)),
                      ("home", "PASS", "", 1, 12, 13, 2, 3, (5, 0), (9, 0))])
    from symmetrica import threat
    passes = threat.actions(ev, DIRECTIONS)
    frames = [{"home": [(-10, 0), (10, 5), (20, -5)], "away": [(30, 0)], "ball": (0, 0), "period": 1}] * 10
    frames += [{"home": [(10, 0), (-10, -5), (-20, 5)], "away": [(30, 0)], "ball": (0, 0), "period": 2}] * 10
    t = make_track(frames)
    poss = np.ones(len(t), dtype=int)
    net = networks.passing_network(passes, t, poss, "home", DIRECTIONS, min_edge=2)
    nodes = {n["id"]: n for n in net["nodes"]}
    # Period 2 positions are rotated into the attacking frame, so both halves agree.
    assert (nodes["home_1"]["x"], nodes["home_1"]["y"]) == pytest.approx((-10, 0))
    assert (nodes["home_2"]["x"], nodes["home_2"]["y"]) == pytest.approx((10, 5))
    assert net["edges"] == [{"a": "home_1", "b": "home_2", "count": 5}]
    assert net["key_link"] == {"a": "home_1", "b": "home_2", "count": 5}
    assert net["hub"] == "home_2" and net["passes"] == 6


def test_possession_labels_and_memory_model():
    ev = make_events([
        ("home", "PASS", "", 1, 1, 5, 1, 2, (0, 0), (5, 0)),
        ("away", "RECOVERY", "INTERCEPTION", 1, 10, 10, 15, None, (5, 0), None),
        ("away", "BALL OUT", "", 1, 15, 15, 15, None, (5, 34), None),
        ("home", "SET PIECE", "THROW IN", 1, 20, 20, 2, None, (5, 34), None),
    ])
    frames = pd.Index(range(1, 25))
    y = labels.possession_labels(ev, frames, pd.Series(1, index=frames))
    assert y[0] == 1 and y[9] == -1 and y[14] == 0 and y[19] == 1

    d_home = np.array([0.3, 5, 5, 5, 0.3, 0.3])
    d_away = np.array([5, 5, 0.3, 5, 5, np.inf])
    est = models.control_with_memory(d_home, d_away, np.zeros(6), radius=1, max_speed=99, min_frames=1)
    # Keeps home while the ball travels (frame 1), switches at away control, and frame 5 is still finite.
    assert est.tolist() == [1, 1, -1, -1, 1, 1]
    assert models.nearest_player(d_home, d_away).tolist() == [1, -1, -1, -1, 1, 1]


def _summary(**over):
    s = {
        "space": {"home_share": 0.56, "home_final_third": 0.3, "away_final_third": 0.2},
        "pressing": {t: {"ppda": p, "counterpress_rate": c, "counterpress_regains": 5, "turnovers": 10,
                         "high_turnovers": h, "high_turnover_shots": 1, "counterpress_pressers": 2.0}
                     for t, p, c, h in (("home", 6.0, 0.5, 8), ("away", 12.0, 0.3, 2))},
        "threat": {t: {"xt_generated": x, "xt_carries": 0.1, "shots": 10, "goals": g, "xg": 1.0,
                       "line_breaking_passes": lb, "packing": 100}
                   for t, x, g, lb in (("home", 2.0, 3, 40), ("away", 1.0, 0, 20))},
        "networks": {t: {"key_link": {"a": f"{t}_4", "b": f"{t}_8", "count": 20}, "hub": f"{t}_8",
                         "top3_link_share": 0.2} for t in ("home", "away")},
        "momentum": {"best_spell": {"home": [20, 0.5], "away": [70, 0.0]}},
    }
    s["threat"]["leaders"] = [{"team": "home", "player": "home_8", "xt": 0.5, "xt_passes": 0.4,
                               "xt_carries": 0.1, "progressive": 4, "line_breaking": 9}]
    for k, v in over.items():
        s[k] = v
    return s


def test_insights_name_the_right_team_with_evidence():
    out = {i.category: i for i in insights.generate(_summary())}
    assert out["space"].team == "home" and "56%" in out["space"].detail
    assert out["pressing"].team == "home" and "6.0" in out["pressing"].detail
    assert out["counter-press"].team == "home" and "50%" in out["counter-press"].detail
    assert out["high turnovers"].team == "home"
    assert out["threat"].team == "home" and "2.00" in out["threat"].detail
    assert out["line-breaking"].team == "home" and "#8" in out["line-breaking"].detail
    assert out["players"].headline == "Home #8 was the main progressor"
    assert out["momentum"].headline == "Home's strongest spell: 20'-30'"
    assert all(0 <= i.strength <= 1 for i in out.values())
    finishing = [i for i in insights.generate(_summary()) if i.category == "finishing"]
    assert [i.headline for i in finishing] == ["Home outperformed their chances",   # 3 goals from 1.0 xG
                                               "Away underperformed their chances"]  # 0 goals from 1.0 xG
    s = _summary()
    s["threat"]["away"]["goals"] = 1
    assert "Away finished in line with their chances" in [i.headline for i in insights.generate(s)]


def test_insights_call_close_contests_even_and_survive_missing_values():
    s = _summary(space={"home_share": 0.51, "home_final_third": None, "away_final_third": 0.2})
    s["pressing"]["away"]["ppda"] = None          # no defensive actions -> NaN -> null in JSON
    s["threat"]["away"]["xt_generated"] = 1.95
    out = {i.category: i for i in insights.generate(s)}
    assert out["space"].headline == "Space was shared evenly" and out["space"].team is None
    assert "pressing" not in out
    assert out["threat"].headline == "Threat creation was evenly matched"
    assert all(isinstance(i.to_dict()["detail"], str) for i in out.values())
    assert not any(math.isnan(i.strength) for i in out.values())
