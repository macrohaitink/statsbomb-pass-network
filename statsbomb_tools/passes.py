"""Passes: open-play passes with flat coordinates, receiver positions and expected threat (xT)."""

import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

SET_PIECES = ["Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"]

PASS_COLUMNS = [
    "id", "match_id", "minute", "second", "period", "possession",
    "player", "player_id", "position", "pass_recipient", "pass_recipient_id",
    "pass_type", "pass_outcome", "pass_length",
    "x", "y", "end_x", "end_y",
]

# Karun Singh's open xT grid: 8 rows (pitch width) x 12 columns (pitch length)
XT_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"


def open_play_passes(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """A team's open-play passes, with x/y start and end coordinates and a `completed` flag.

    Set pieces and injury clearances (ball kicked out for an injury) are excluded.
    `passer_position` is the passer's position at the time of the pass.
    """
    passes = events[(events["type"] == "Pass") & (events["team_id"] == team_id)].copy()
    passes[["x", "y"]] = pd.DataFrame(passes["location"].tolist(), index=passes.index)
    passes[["end_x", "end_y"]] = pd.DataFrame(passes["pass_end_location"].tolist(), index=passes.index)
    passes = passes[PASS_COLUMNS].rename(columns={"position": "passer_position"})

    open_play = passes[
        ~passes["pass_type"].isin(SET_PIECES)
        & (passes["pass_outcome"] != "Injury Clearance")
    ].copy()
    # StatsBomb leaves pass_outcome empty for completed passes
    open_play["completed"] = open_play["pass_outcome"].isna()
    return open_play


def receiver_positions(passes: pd.DataFrame, events: pd.DataFrame, team_id: int) -> pd.Series:
    """Position of each pass's receiver, from the Ball Receipt* event linked to the pass.

    Incomplete passes get the intended receiver's position when StatsBomb logged one;
    passes with no linked receipt (e.g. at the final whistle) get NaN.
    """
    receipts = (
        events.loc[(events["type"] == "Ball Receipt*") & (events["team_id"] == team_id),
                   ["related_events", "position"]]
        .explode("related_events")
    )
    receipts = receipts[receipts["related_events"].isin(passes["id"])].set_index("related_events")
    return passes["id"].map(receipts["position"]).rename("receiver_position")


def load_xT_grid(path: Path) -> np.ndarray:
    """Karun Singh's open xT grid (8 x 12), downloaded to `path` the first time."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(XT_URL, path)
    return np.array(json.loads(path.read_text()))


def xT_at(xT_grid: np.ndarray, x: pd.Series, y: pd.Series) -> np.ndarray:
    """xT of the grid cell containing each (x, y) point on StatsBomb's 120 x 80 pitch."""
    rows, cols = xT_grid.shape
    col = np.clip((x.to_numpy() / 120 * cols).astype(int), 0, cols - 1)
    row = np.clip((y.to_numpy() / 80 * rows).astype(int), 0, rows - 1)
    return xT_grid[row, col]


def pass_xT(passes: pd.DataFrame, xT_grid: np.ndarray) -> pd.Series:
    """Threat added by each pass: only completed passes count, and backward passes count as 0, not negative."""
    gained = xT_at(xT_grid, passes["end_x"], passes["end_y"]) - xT_at(xT_grid, passes["x"], passes["y"])
    return pd.Series(np.where(passes["completed"], gained.clip(min=0), 0.0), index=passes.index, name="xT")
