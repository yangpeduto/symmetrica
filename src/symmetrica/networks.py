"""Passing networks: who combines with whom, and where they stand when doing it.

Nodes are players placed at their **average position while their team has
the ball**, taken from tracking (so it reflects the shape in possession,
not where they happened to pass from), in the team's attacking frame:
every team attacks left to right, both halves folded together. Edges count
completed passes between a pair in either direction.
"""
import numpy as np
import pandas as pd

from .data import FPS
from .pitch import direction_per_frame


def passing_network(passes: pd.DataFrame, track: pd.DataFrame, possession: np.ndarray,
                    team: str, directions: dict, min_edge: int = 3) -> dict:
    p = passes[(passes["team"] == team) & (passes["kind"] == "pass") & passes["receiver"].notna()]
    code = 1 if team == "home" else -1
    in_pos = possession == code
    d = direction_per_frame(track, team, directions).astype(float)

    players = sorted(set(p["player"]) | set(p["receiver"]), key=lambda s: int(s.split("_")[1]))
    made = p["player"].value_counts()
    received = p["receiver"].value_counts()
    total = max(len(p), 1)
    nodes = []
    for pl in players:
        x = track[f"{pl}_x"].to_numpy() * d
        y = track[f"{pl}_y"].to_numpy() * d
        on = np.isfinite(x) & (d != 0)
        mask = on & in_pos
        if not mask.any():
            mask = on
        if not mask.any():
            continue
        nodes.append({
            "id": pl, "number": int(pl.split("_")[1]),
            "x": float(np.mean(x[mask])), "y": float(np.mean(y[mask])),
            "minutes": float(on.sum() / FPS / 60),
            "passes": int(made.get(pl, 0)), "received": int(received.get(pl, 0)),
            "involvement": float((made.get(pl, 0) + received.get(pl, 0)) / (2 * total)),
        })

    pairs = [tuple(sorted(ab, key=lambda s: int(s.split("_")[1]))) for ab in zip(p["player"], p["receiver"])]
    counts = pd.Series(pairs, dtype=object).value_counts()
    edges = [{"a": a, "b": b, "count": int(c)} for (a, b), c in counts.items() if c >= min_edge]
    top = counts.index[0] if len(counts) else None
    hub = max(nodes, key=lambda n: n["involvement"]) if nodes else None
    return {
        "team": team, "nodes": nodes, "edges": edges, "passes": int(len(p)),
        "key_link": {"a": top[0], "b": top[1], "count": int(counts.iloc[0])} if top else None,
        "hub": hub["id"] if hub else None,
        # Share of the network's passes carried by its three busiest links.
        "top3_link_share": float(counts.iloc[:3].sum() / total) if len(counts) else 0.0,
    }
