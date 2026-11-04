"""Parsing of Metrica's raw CSV layout and event normalisation."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_events
from symmetrica import data


def _write_tracking(path, team, players):
    # Metrica layout: two title rows, a row of player names (each spanning x and y), then data.
    names = ["", "", ""] + sum([[p, ""] for p in players], []) + ["Ball", ""]
    lines = [",,," + ",".join([team] * (len(names) - 3)),
             ",,," + ",".join(["", ""] * (len(players) + 1)),
             "Period,Frame,Time [s]," + ",".join(n for n in names[3:]),
             "1,1,0.04," + ",".join(["0.5,0.5", "0,0"][: len(players)]) + ",0.25,0.75",
             "1,2,0.08," + ",".join(["1,1", "NaN,NaN"][: len(players)]) + ",0.5,0.5"]
    path.write_text("\n".join(lines) + "\n")


def test_read_tracking_converts_to_metres_and_joins_teams(tmp_path):
    _write_tracking(tmp_path / "home.csv", "Home", ["Player1", "Player11"])
    _write_tracking(tmp_path / "away.csv", "Away", ["Player15", "Player25"])
    t = data.read_tracking(tmp_path / "home.csv", tmp_path / "away.csv")

    assert list(t.index) == [1, 2]
    assert {"home_1_x", "home_11_y", "away_15_x", "away_25_y", "ball_x", "period", "time"} <= set(t.columns)
    # (0.5, 0.5) is the centre spot; (0, 0) the top-left corner; (1, 1) bottom-right.
    assert t.loc[1, "home_1_x"] == pytest.approx(0.0)
    assert t.loc[1, "home_1_y"] == pytest.approx(0.0)
    assert (t.loc[1, "home_11_x"], t.loc[1, "home_11_y"]) == pytest.approx((-52.5, 34.0))
    assert (t.loc[2, "home_1_x"], t.loc[2, "home_1_y"]) == pytest.approx((52.5, -34.0))
    assert (t.loc[1, "ball_x"], t.loc[1, "ball_y"]) == pytest.approx((-26.25, -17.0))
    assert np.isnan(t.loc[2, "home_11_x"])
    assert data.player_ids(t, "away") == ["away_15", "away_25"]


def test_player_arrays_shape():
    t = pd.DataFrame({"home_1_x": [1.0, 2.0], "home_1_y": [0.0, 0.0],
                      "home_2_x": [3.0, np.nan], "home_2_y": [1.0, np.nan], "ball_x": [0, 0]})
    x, y = data.player_arrays(t, "home")
    assert x.shape == y.shape == (2, 2)
    assert np.isnan(x[1, 1])


def test_prepare_events_metres_and_player_ids():
    ev = make_events([
        ("away", "PASS", "", 1, 10, 20, 19, 21, (0.0, 0.0), (10.0, -5.0)),
        ("home", "BALL LOST", "INTERCEPTION", 1, 30, 40, 4, None, (-52.5, 34.0), None),
    ])
    assert list(ev["team"]) == ["away", "home"]
    assert ev.loc[0, "player"] == "away_19" and ev.loc[0, "receiver"] == "away_21"
    assert ev.loc[1, "receiver"] is None
    assert (ev.loc[0, "end_x"], ev.loc[0, "end_y"]) == pytest.approx((10.0, -5.0))
    assert (ev.loc[1, "start_x"], ev.loc[1, "start_y"]) == pytest.approx((-52.5, 34.0))
    assert np.isnan(ev.loc[1, "end_x"])
    assert ev["subtype"].tolist() == ["", "INTERCEPTION"]
