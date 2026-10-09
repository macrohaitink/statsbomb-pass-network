"""Season totals per player: appearances, minutes, goals and assists."""

import pandas as pd

from statsbomb_tools.lineups import main_positions, player_appearances
from statsbomb_tools.metrics import event_counts

SHOOTOUT = 5  # period number of penalty shootouts, which don't count as goals


def season_totals(events: pd.DataFrame, team_id: int, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player of a team, indexed by player_id.

    Columns: player, appearances, starts, minutes, first_match, last_match, goals, penalty_goals, assists, position, position_share, position_group
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
    totals = totals.join(main_positions(events, team_id), on="player")
    return totals.sort_values("minutes", ascending=False)

def season_stats(events: pd.DataFrame, team_id: int, matches: pd.DataFrame) -> pd.DataFrame:
    """season_totals plus event_counts: one row per player of a team, indexed by player_id."""
    counts = event_counts(events, team_id)
    totals = season_totals(events, team_id, matches).join(counts)
    return totals.fillna(dict.fromkeys(counts.columns, 0))  # players with no counted events at all


def combine_clubs(league: pd.DataFrame) -> pd.DataFrame:
    """One row per player, for a league table where players who moved clubs have a row per club.

    Counts are summed and the match dates span both spells. Team and position come from the
    club where the player played most minutes; `teams` lists all clubs in date order.
    """
    main = league.sort_values("minutes", ascending=False)
    main = main[~main.index.duplicated()].copy()

    summed = league.select_dtypes("number").drop(columns=["team_id", "position_share"]).groupby(level=0).sum()
    main[summed.columns] = summed.reindex(main.index)
    main["first_match"] = league.groupby(level=0)["first_match"].min()
    main["last_match"] = league.groupby(level=0)["last_match"].max()
    main["teams"] = league.sort_values("first_match").groupby(level=0)["team"].agg(" / ".join)
    return main


def match_log(events: pd.DataFrame, team_id: int, player_id: int, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match a player appeared in: date, opponent, home, minutes, goals and npxG."""
    apps = player_appearances(events, team_id)
    apps = apps[apps["player_id"] == player_id]
    shots = events[(events["player_id"] == player_id) & (events["type"] == "Shot") & (events["period"] < SHOOTOUT)]
    np_shots = shots[shots["shot_type"] != "Penalty"]
    per_match = pd.DataFrame({
        "goals": shots[shots["shot_outcome"] == "Goal"].groupby("match_id").size(),
        "npxg": np_shots.groupby("match_id")["shot_statsbomb_xg"].sum(),
    })
    log = (
        apps[["match_id", "minutes"]]
        .merge(matches[["match_id", "match_date", "home_team_id", "home_team", "away_team"]], on="match_id")
        .join(per_match, on="match_id")
    )
    log[["goals", "npxg"]] = log[["goals", "npxg"]].fillna(0)
    log["home"] = log["home_team_id"] == team_id
    log["opponent"] = log["away_team"].where(log["home"], log["home_team"])
    log["match_date"] = pd.to_datetime(log["match_date"])
    return (log[["match_date", "opponent", "home", "minutes", "goals", "npxg"]]
            .sort_values("match_date").reset_index(drop=True))
