"""Load StatsBomb open data: competitions, matches and match events (cached on disk)."""

import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb


def _quiet(func, **kwargs):
    """Call a statsbombpy function without its 'no credentials' warning (expected for open data)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*credentials.*")
        return func(**kwargs)


def load_competitions() -> pd.DataFrame:
    """All competitions and seasons available in the open data."""
    return _quiet(sb.competitions)


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """All matches of one competition season."""
    return _quiet(sb.matches, competition_id=competition_id, season_id=season_id)


def get_team_id(matches: pd.DataFrame, team_name: str) -> int:
    """StatsBomb id of a team, looked up by name in a matches table."""
    home = matches.loc[matches["home_team"] == team_name, "home_team_id"]
    away = matches.loc[matches["away_team"] == team_name, "away_team_id"]
    ids = pd.concat([home, away]).unique()
    if len(ids) != 1:
        raise ValueError(f"Expected one id for {team_name!r}, found {list(ids)}")
    return int(ids[0])


def team_matches(matches: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Matches played by one team (home and away), in date order."""
    played = (matches["home_team_id"] == team_id) | (matches["away_team_id"] == team_id)
    return matches[played].sort_values("match_date").reset_index(drop=True)


def load_match_events(match_id: int, cache_dir: Path) -> pd.DataFrame:
    """Events of one match, downloaded only if not already cached in `cache_dir`."""
    path = Path(cache_dir) / f"{match_id}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = _quiet(sb.events, match_id=match_id)
    events.to_pickle(path)
    return events


def load_events(match_ids, cache_dir: Path) -> pd.DataFrame:
    """Events of several matches in one table, in match and event order."""
    match_ids = list(match_ids)
    frames = []
    for i, match_id in enumerate(match_ids, start=1):
        frames.append(load_match_events(match_id, cache_dir))
        print(f"{i}/{len(match_ids)} matches loaded", end="\r")
    print()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["match_id", "index"])
        .reset_index(drop=True)
    )

def load_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    """Lineups of one match, as {team name: players table} (with player_nickname, country, jersey_number)."""
    return _quiet(sb.lineups, match_id=match_id)