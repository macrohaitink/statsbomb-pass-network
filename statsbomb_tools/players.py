"""Season totals per player: appearances, minutes, goals and assists."""

import pandas as pd

from statsbomb_tools.lineups import player_appearances

SHOOTOUT = 5  # period number of penalty shootouts, which don't count as goals


def season_totals(events: pd.DataFrame, team_id: int, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player of a team, indexed by player_id.

    Columns: player, appearances, starts, minutes, first_match, last_match, goals, penalty_goals, assists.
    Own goals are separate events in StatsBomb, so they are not counted. Assists are StatsBomb's
    `pass_goal_assist`: only the pass straight before the goal, so they can be lower than other sources.
    """
    match_dates = matches[["match_id", "match_date"]].assign(match_date=lambda m: pd.to_datetime(m["match_date"]))
    apps = player_appearances(events, team_id).merge(match_dates, on="match_id")
    totals = apps.groupby("player_id").agg(
        player=("player", "first"),
        appearances=("match_id", "nunique"),
        starts=("position", "count"),  # count skips missing values: substitutes have no starting position
        minutes=("minutes", "sum"),
        first_match=("match_date", "min"),
        last_match=("match_date", "max"),
    )

    team_events = events[(events["team_id"] == team_id) & (events["period"] < SHOOTOUT)]
    goals = team_events[(team_events["type"] == "Shot") & (team_events["shot_outcome"] == "Goal")]
    counts = pd.DataFrame({
        "goals": goals.groupby("player_id").size(),
        "penalty_goals": goals[goals["shot_type"] == "Penalty"].groupby("player_id").size(),
        "assists": team_events[team_events["pass_goal_assist"] == True].groupby("player_id").size(),
    })
    counts.index = counts.index.astype(int)  # event player ids are floats (missing for some event types)

    totals = totals.join(counts)
    totals[counts.columns] = totals[counts.columns].fillna(0).astype(int)
    return totals.sort_values("minutes", ascending=False)
