"""Frame-level possession labels derived from Metrica event data.

The event feed only tells us when something happens on the ball, so the
label for frame f is "the team of the last on-ball event before f", with
two exceptions where the ball is dead and nobody has possession:
  * after the ball goes out (BALL OUT, or a shot/pass subtype ending in OUT)
  * after a foul (FAULT RECEIVED)
until the next on-ball event restarts play.

Labels: 1 = home, -1 = away, 0 = dead ball / unknown.
"""
import numpy as np
import pandas as pd

ON_BALL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}
TEAM_CODE = {"home": 1, "away": -1}


def possession_labels(events: pd.DataFrame, frames: pd.Index, periods: pd.Series) -> np.ndarray:
    n_frames = int(frames.max()) + 1
    labels = np.zeros(n_frames, dtype=np.int8)

    marks = []  # (frame, value) — value is the team code, or 0 for "ball dead from here"
    for ev in events.itertuples(index=False):
        end = ev.end_frame if ev.end_frame > 0 else ev.start_frame
        if ev.type in ON_BALL:
            marks.append((ev.start_frame, TEAM_CODE[ev.team]))
            if "OUT" in ev.subtype:
                marks.append((end, 0))
        elif ev.type == "BALL OUT":
            marks.append((end, 0))
        elif ev.type == "FAULT RECEIVED":
            marks.append((ev.start_frame, 0))

    # Stable sort: at equal frames a later event (e.g. the OUT after a pass) wins.
    marks.sort(key=lambda m: m[0])
    for (frame, value), nxt in zip(marks, marks[1:] + [(n_frames, 0)]):
        labels[frame:nxt[0]] = value

    # Nothing carries over half time.
    period = periods.reindex(range(n_frames)).to_numpy()
    second = np.nanargmax(period == 2) if np.any(period == 2) else n_frames
    first_event_second_half = min((f for f, _ in marks if f >= second), default=n_frames)
    labels[second:first_event_second_half] = 0
    return labels[frames.to_numpy()]
