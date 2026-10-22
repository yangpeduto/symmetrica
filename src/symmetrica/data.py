"""Load Metrica Sports sample data (CSV format, games 1 and 2).

Coordinates come normalised to [0, 1] with (0, 0) top left. We convert to
metres on a 105 x 68 pitch with (0, 0) at the centre spot.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FPS = 25

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"


def _game_dir(game: int) -> Path:
    return DATA_DIR / f"Sample_Game_{game}"


def download(game: int) -> None:
    """Fetch the three CSV files of one sample game into data/."""
    from urllib.request import urlretrieve

    target = _game_dir(game)
    target.mkdir(parents=True, exist_ok=True)
    for part in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team"):
        name = f"Sample_Game_{game}_{part}.csv"
        if not (target / name).exists():
            urlretrieve(f"{BASE_URL}/Sample_Game_{game}/{name}", target / name)


def _to_metres(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
    return (np.asarray(x) - 0.5) * PITCH_LENGTH, (0.5 - np.asarray(y)) * PITCH_WIDTH


def _read_tracking_file(path: Path, team: str) -> pd.DataFrame:
    # Row 3 holds "Player11,,Player1,,..." — every player spans an x and a y column.
    header = pd.read_csv(path, skiprows=2, nrows=0).columns
    names = ["period", "frame", "time"]
    for col in header[3:]:
        if not col.startswith("Unnamed"):
            prefix = "ball" if col == "Ball" else f"{team}_{col.replace('Player', '')}"
            names += [f"{prefix}_x", f"{prefix}_y"]
    df = pd.read_csv(path, skiprows=3, header=None, names=names)
    for col in [c for c in df.columns if c.endswith("_x")]:
        stem = col[:-2]
        df[f"{stem}_x"], df[f"{stem}_y"] = _to_metres(df[f"{stem}_x"], df[f"{stem}_y"])
    return df.set_index("frame")


def tracking(game: int) -> pd.DataFrame:
    """One row per frame: period, time, home_*/away_* player positions and ball."""
    folder = _game_dir(game)
    home = _read_tracking_file(folder / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "home")
    away = _read_tracking_file(folder / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "away")
    away = away.drop(columns=["period", "time", "ball_x", "ball_y"])
    return home.join(away)


def events(game: int) -> pd.DataFrame:
    df = pd.read_csv(_game_dir(game) / f"Sample_Game_{game}_RawEventsData.csv")
    df.columns = ["team", "type", "subtype", "period", "start_frame", "start_time",
                  "end_frame", "end_time", "from", "to", "start_x", "start_y", "end_x", "end_y"]
    df["team"] = df["team"].str.lower()
    df["subtype"] = df["subtype"].fillna("")
    return df


def player_arrays(track: pd.DataFrame, team: str) -> tuple[np.ndarray, np.ndarray]:
    """(frames x players) arrays of x and y for one team, NaN where a player is off the pitch."""
    xs = [c for c in track.columns if c.startswith(f"{team}_") and c.endswith("_x")]
    ys = [c[:-2] + "_y" for c in xs]
    return track[xs].to_numpy(), track[ys].to_numpy()
