"""Pass network tables: one row per node (position) and per edge (pair of positions)."""

import pandas as pd


def build_nodes(passes: pd.DataFrame, formation_minutes: pd.DataFrame, main_players: pd.DataFrame) -> pd.DataFrame:
    """One row per position per formation: on-ball location, passing volume, completion and xT.

    `passes` needs formation, passer_position, receiver_position, completed, x, y, end_x, end_y and xT.
    The location averages where the position passed from and where it received completed passes.
    Totals are also given per 90 minutes of the formation, so formations can be compared.
    """
    completed = passes[passes["completed"]]
    on_ball = pd.concat([
        passes[["formation", "passer_position", "x", "y"]]
            .rename(columns={"passer_position": "position"}),
        completed[["formation", "receiver_position", "end_x", "end_y"]]
            .rename(columns={"receiver_position": "position", "end_x": "x", "end_y": "y"}),
    ])

    nodes = (
        passes.groupby(["formation", "passer_position"])
        .agg(passes=("completed", "size"), completed=("completed", "sum"), xT=("xT", "sum"))
        .rename_axis(["formation", "position"])
        .join(on_ball.groupby(["formation", "position"])[["x", "y"]].mean())
        .reset_index()
        .merge(main_players[["formation", "position", "player", "share"]],
               on=["formation", "position"], how="left")
    )
    nodes["minutes"] = nodes["formation"].map(formation_minutes["minutes"])
    nodes["completed_p90"] = nodes["completed"] / nodes["minutes"] * 90
    nodes["completion"] = nodes["completed"] / nodes["passes"]
    nodes["xT_p90"] = nodes["xT"] / nodes["minutes"] * 90
    return nodes


def build_edges(passes: pd.DataFrame, nodes: pd.DataFrame, formation_minutes: pd.DataFrame) -> pd.DataFrame:
    """Completed passes between each pair of positions, both directions combined, with node locations.

    `passes` needs formation, passer_position, receiver_position, completed and xT.
    Each pair is stored once, alphabetically (pos_a < pos_b), with the x/y of both ends for drawing.
    """
    links = passes[passes["completed"]].dropna(subset=["receiver_position"])
    links = links[links["passer_position"] != links["receiver_position"]]

    # Alphabetical order, so A -> B and B -> A get the same key
    pair = links[["passer_position", "receiver_position"]]
    links = links.assign(pos_a=pair.min(axis=1), pos_b=pair.max(axis=1))

    edges = (
        links.groupby(["formation", "pos_a", "pos_b"])
        .agg(completed=("xT", "size"), xT_per_pass=("xT", "mean"))
        .reset_index()
    )
    edges["completed_p90"] = edges["completed"] / edges["formation"].map(formation_minutes["minutes"]) * 90

    xy = nodes.set_index(["formation", "position"])[["x", "y"]]
    return (
        edges
        .join(xy, on=["formation", "pos_a"])
        .join(xy, on=["formation", "pos_b"], rsuffix="_b")
        .rename(columns={"x": "x_a", "y": "y_a"})
        .sort_values(["formation", "completed_p90"], ascending=[True, False])
        .reset_index(drop=True)
    )
