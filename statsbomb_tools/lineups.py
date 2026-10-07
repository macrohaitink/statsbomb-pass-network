"""Who was on the pitch, in which formation and position, and for how long."""

import pandas as pd

SENDING_OFF = ["Red Card", "Second Yellow"]

POSITION_GROUPS = {
    position: group
    for group, positions in {
        "Goalkeeper": ["Goalkeeper"],
        "Centre-back": ["Right Center Back", "Center Back", "Left Center Back"],
        "Full-back": ["Right Back", "Left Back", "Right Wing Back", "Left Wing Back"],
        "Defensive midfielder": ["Right Defensive Midfield", "Center Defensive Midfield", "Left Defensive Midfield"],
        "Central midfielder": ["Right Center Midfield", "Center Midfield", "Left Center Midfield"],
        "Attacking midfielder": ["Right Attacking Midfield", "Center Attacking Midfield", "Left Attacking Midfield"],
        "Winger": ["Right Midfield", "Left Midfield", "Right Wing", "Left Wing"],
        "Striker": ["Striker", "Right Center Forward", "Left Center Forward", "Secondary Striker"],
    }.items()
    for position in positions
}

def _sent_off(events: pd.DataFrame) -> pd.Series:
    """True for events where a player was sent off (cards are on Foul Committed or Bad Behaviour events)."""
    card = events["foul_committed_card"].fillna(events["bad_behaviour_card"])
    return card.isin(SENDING_OFF)


def player_appearances(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """One row per player per match: minute on, minute off and minutes played.

    Players leave the pitch when substituted or sent off; otherwise at the last minute of the match.
    `position` is the starting position (missing for substitutes).
    """
    team_events = events[events["team_id"] == team_id]
    start_xi = team_events[team_events["type"] == "Starting XI"]
    subs = team_events[team_events["type"] == "Substitution"]

    starters = pd.DataFrame([
        {"match_id": row["match_id"],
         "player_id": p["player"]["id"],
         "player": p["player"]["name"],
         "position": p["position"]["name"],
         "on": 0}
        for _, row in start_xi.iterrows()
        for p in row["tactics"]["lineup"]
    ])
    sub_on = pd.DataFrame({
        "match_id": subs["match_id"],
        "player_id": subs["substitution_replacement_id"].astype(int),
        "player": subs["substitution_replacement"],
        "on": subs["minute"],
    })

    off = (
        pd.concat([subs, team_events[_sent_off(team_events)]])[["match_id", "player_id", "minute"]]
        .groupby(["match_id", "player_id"], as_index=False)["minute"].min()
        .rename(columns={"minute": "off"})
    )
    match_end = events.groupby("match_id")["minute"].max().rename("match_end")

    appearances = (
        pd.concat([starters, sub_on], ignore_index=True)
        .merge(off, on=["match_id", "player_id"], how="left")
        .merge(match_end, on="match_id")
    )
    appearances["off"] = appearances["off"].fillna(appearances["match_end"])
    appearances["minutes"] = appearances["off"] - appearances["on"]
    return appearances


def formation_at_each_event(events: pd.DataFrame, team_id: int) -> pd.Series:
    """The team's formation (e.g. "4-2-3-1") in force at every event, of either team.

    Read from the team's Starting XI and Tactical Shift events, then carried forward
    until the next change in that match. `events` must be in match and event order.
    """
    is_setup = (events["team_id"] == team_id) & events["type"].isin(["Starting XI", "Tactical Shift"])
    formation = events.loc[is_setup, "tactics"].map(lambda t: "-".join(str(t["formation"])))
    return formation.reindex(events.index).groupby(events["match_id"]).ffill()


def event_durations(events: pd.DataFrame) -> pd.Series:
    """Minutes from each event to the next one in the same half (0 for the last event of a half)."""
    time = pd.to_timedelta(events["timestamp"])
    next_time = time.groupby([events["match_id"], events["period"]]).shift(-1)
    return (next_time - time).dt.total_seconds().fillna(0) / 60


def minutes_by_formation(events: pd.DataFrame) -> pd.DataFrame:
    """Minutes and matches per formation, most used first. Needs `formation` and `duration` columns."""
    return (
        events.groupby("formation")
        .agg(minutes=("duration", "sum"), matches=("match_id", "nunique"))
        .sort_values("minutes", ascending=False)
        .round()
    )


def minutes_by_position(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Minutes each player spent in each position of each formation. Needs `formation` and `duration` columns.

    Lineups are rebuilt from the team's lineup-changing events: Starting XI and Tactical Shift
    give the full lineup, a substitute takes the position of the player going off, and a
    sent-off player leaves without a replacement.
    """
    changes = events[(events["team_id"] == team_id) & (
        events["type"].isin(["Starting XI", "Tactical Shift", "Substitution"]) | _sent_off(events)
    )]

    lineups = []
    for idx, ev in changes.iterrows():
        if ev["type"] in ("Starting XI", "Tactical Shift"):
            on_pitch = {p["player"]["name"]: p["position"]["name"] for p in ev["tactics"]["lineup"]}
        elif ev["type"] == "Substitution":
            on_pitch[ev["substitution_replacement"]] = on_pitch.pop(ev["player"])
        else:
            on_pitch.pop(ev["player"], None)  # None: ignore cards shown to players on the bench
        lineups += [{"lineup_id": idx, "player": player, "position": position}
                    for player, position in on_pitch.items()]
    lineups = pd.DataFrame(lineups)

    # Tag every event with the lineup in force, then add up time per lineup
    lineup_id = pd.Series(changes.index, index=changes.index).reindex(events.index)
    lineup_id = lineup_id.groupby(events["match_id"]).ffill()
    lineup_minutes = events.groupby(lineup_id).agg(
        formation=("formation", "first"),
        minutes=("duration", "sum"),
    )

    return (
        lineups.merge(lineup_minutes, left_on="lineup_id", right_index=True)
        .groupby(["formation", "position", "player"], as_index=False)["minutes"].sum()
    )


def main_player_by_position(position_minutes: pd.DataFrame, formation_minutes: pd.DataFrame) -> pd.DataFrame:
    """Player with the most minutes in each position of each formation, and their share of its minutes."""
    top = (
        position_minutes.sort_values("minutes", ascending=False)
        .drop_duplicates(["formation", "position"])
        .sort_values(["formation", "position"])
        .reset_index(drop=True)
    )
    top["share"] = top["minutes"] / top["formation"].map(formation_minutes["minutes"])
    return top

def main_positions(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Each player's most-played position and position group, by minutes, indexed by player name.

    The group is chosen by total minutes across its positions, so a forward split between
    Left and Right Center Forward still counts as a Striker. `events` must be in match and event order.
    """
    events = events.assign(
        formation=formation_at_each_event(events, team_id),
        duration=event_durations(events),
    )
    minutes = minutes_by_position(events, team_id).groupby(["player", "position"])["minutes"].sum()
    minutes = minutes.reset_index().assign(group=lambda m: m["position"].map(POSITION_GROUPS))

    def most_minutes(by: str) -> pd.DataFrame:
        per_player = minutes.groupby(["player", by])["minutes"].sum().reset_index()
        return per_player.sort_values("minutes", ascending=False).drop_duplicates("player").set_index("player")

    position, group = most_minutes("position"), most_minutes("group")
    return pd.DataFrame({
        "position": position["position"],
        "position_share": position["minutes"] / minutes.groupby("player")["minutes"].sum(),
        "position_group": group["group"],
    })
