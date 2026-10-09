"""Per-player counts of on-ball actions: the raw material for per-90 metrics."""

import numpy as np
import pandas as pd

SHOOTOUT = 5
GOAL = np.array([120, 40])  # centre of the attacking goal; StatsBomb pitches are 120 x 80
BOX_X, BOX_HALF_WIDTH = 102, 22  # penalty box: x >= 102 and y within 40 +/- 22
ON_TARGET = ["Goal", "Saved", "Saved to Post"]
SET_PIECES = ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"]
PROGRESSIVE = 0.75  # a progressive pass or carry ends at most 75% as far from goal as it started
TOUCH_TYPES = ["Pass", "Ball Receipt*", "Carry", "Shot", "Dribble"]  # on-ball events used for heatmaps
PER_90 = ["np_goals", "npxg", "np_shots", "assists", "xa", "key_passes", "box_receptions",
          "successful_dribbles", "progressive_passes", "progressive_carries"]

def _xy(locations: pd.Series) -> np.ndarray:
    """A column of [x, y] lists as an (n, 2) array."""
    return np.array(locations.tolist(), dtype=float).reshape(-1, 2)


def _progressive(start: pd.Series, end: pd.Series) -> np.ndarray:
    """True where the ball ends at most PROGRESSIVE times as far from goal as it started."""
    distance = lambda locations: np.linalg.norm(_xy(locations) - GOAL, axis=1)
    return distance(end) <= PROGRESSIVE * distance(start)


def _in_box(locations: pd.Series) -> np.ndarray:
    """True for locations inside the opposition penalty box."""
    xy = _xy(locations)
    return (xy[:, 0] >= BOX_X) & (np.abs(xy[:, 1] - GOAL[1]) <= BOX_HALF_WIDTH)


def event_counts(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Counts of shooting, creating and possession actions per player of a team, indexed by player_id.

    Shots exclude penalties (np_*). Key passes are passes leading to a shot (assists included);
    xa is the xG of those shots. Progressive passes are completed open-play passes.
    """
    ev = events[(events["team_id"] == team_id) & (events["period"] < SHOOTOUT)]
    of_type = lambda name: ev[ev["type"] == name]

    shots = of_type("Shot")
    np_shots = shots[shots["shot_type"] != "Penalty"]
    passes = of_type("Pass")
    completed = passes[passes["pass_outcome"].isna()]
    open_play = completed[~completed["pass_type"].isin(SET_PIECES)]
    key_passes = passes[(passes["pass_shot_assist"] == True) | (passes["pass_goal_assist"] == True)]
    carries = of_type("Carry")
    dribbles = of_type("Dribble")
    received = of_type("Ball Receipt*")
    received = received[received["ball_receipt_outcome"].isna()]

    count = lambda rows: rows.groupby("player_id").size()
    shot_xg = shots.set_index("id")["shot_statsbomb_xg"]

    counts = pd.DataFrame({
        "np_goals": count(np_shots[np_shots["shot_outcome"] == "Goal"]),
        "np_shots": count(np_shots),
        "np_shots_on_target": count(np_shots[np_shots["shot_outcome"].isin(ON_TARGET)]),
        "npxg": np_shots.groupby("player_id")["shot_statsbomb_xg"].sum(),
        "key_passes": count(key_passes),
        "xa": key_passes["pass_assisted_shot_id"].map(shot_xg).groupby(key_passes["player_id"]).sum(),
        "box_receptions": count(received[_in_box(received["location"])]),
        "dribbles": count(dribbles),
        "successful_dribbles": count(dribbles[dribbles["dribble_outcome"] == "Complete"]),
        "passes": count(passes),
        "completed_passes": count(completed),
        "progressive_passes": count(open_play[_progressive(open_play["location"], open_play["pass_end_location"])]),
        "progressive_carries": count(carries[_progressive(carries["location"], carries["carry_end_location"])]),
    })
    counts.index = counts.index.astype(int)
    return counts.fillna(0)

def player_metrics(totals: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates and ratios from season totals (after combine_clubs), one row per player."""
    metrics = totals[PER_90].div(totals["minutes"], axis=0).mul(90)
    metrics["xg_per_shot"] = totals["npxg"] / totals["np_shots"]
    metrics["shots_on_target_pct"] = 100 * totals["np_shots_on_target"] / totals["np_shots"]
    metrics["pass_completion_pct"] = 100 * totals["completed_passes"] / totals["passes"]
    return metrics


def percentiles(metrics: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-100) of each player on each metric, within the players in `metrics`."""
    return metrics.rank(pct=True).mul(100)


def touch_locations(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """x, y of a player's on-ball events (StatsBomb coordinates, attacking towards x = 120)."""
    on_ball = events[(events["player_id"] == player_id) & events["type"].isin(TOUCH_TYPES)]
    return pd.DataFrame(_xy(on_ball["location"]), columns=["x", "y"])
