"""Draw pass networks on vertical pitches, one per formation, with shared scales and a legend."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from mplsoccer import VerticalPitch

# Colours
BACKGROUND = ["#fbf3f4", "#f1d9de"]   # top -> bottom gradient behind everything
GRASS = "#aabb97"
PITCH_LINES = "#f7f7f2"
TEXT = "#2b2b2b"
MUTED = "#6e6e6e"
LEGEND_GREY = "#d9d0d2"
CMAP = LinearSegmentedColormap.from_list("xT", ["#f4a582", "#d6604d", "#8e2c6b", "#3b1c4a"])

# Sizes
MAX_NODE_SIZE = 1400    # marker area (points²) of the busiest node
MAX_EDGE_WIDTH = 9      # line width (points) of the busiest link
LABEL_MARGIN = 30       # space beside the pitch for labels (pitch units; pitch is 80 wide)
LABEL_GAP = 9           # minimum vertical distance between two labels (pitch units)

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/{}"


def use_space_mono(font_dir: Path) -> None:
    """Make Space Mono the default matplotlib font, downloading it to `font_dir` the first time."""
    font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)
    for file in ["SpaceMono-Regular.ttf", "SpaceMono-Bold.ttf"]:
        path = font_dir / file
        if not path.exists():
            urlretrieve(FONT_URL.format(file), path)
        font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = "Space Mono"


@dataclass
class Scales:
    """Colour, size and width scales shared by every pitch, so the plots can be compared directly."""
    node_norm: Normalize
    edge_norm: Normalize
    max_node_p90: float
    max_edge_p90: float

    @classmethod
    def from_tables(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> "Scales":
        return cls(
            node_norm=Normalize(0, nodes["xT_p90"].max()),
            edge_norm=Normalize(0, edges["xT_per_pass"].max()),
            max_node_p90=nodes["completed_p90"].max(),
            max_edge_p90=edges["completed_p90"].max(),
        )

    def node_size(self, completed_p90):
        return completed_p90 / self.max_node_p90 * MAX_NODE_SIZE

    def edge_width(self, completed_p90):
        return completed_p90 / self.max_edge_p90 * MAX_EDGE_WIDTH


def _spread(heights, gap, low, high):
    """Move label heights apart so neighbours are at least `gap` apart, staying within [low, high]."""
    order = np.argsort(heights)
    placed = np.array(heights, dtype=float)[order]
    for i in range(1, len(placed)):                 # push up from the bottom
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed[-1] = min(placed[-1], high)
    for i in range(len(placed) - 2, -1, -1):        # then pull down from the top if we overshot
        placed[i] = min(placed[i], placed[i + 1] - gap)
    placed = np.maximum(placed, low)
    result = np.empty_like(placed)
    result[order] = placed
    return result


def _label_nodes(ax, nodes, scales, display_names):
    """Labels in columns beside the pitch, joined to their node by a thin leader line."""
    nodes = nodes.assign(
        left=nodes["position"].str.contains("Left")
             | (~nodes["position"].str.contains("Right") & (nodes["y"] < 40))
    )
    for left, side in nodes.groupby("left"):
        label_x = -3 if left else 83                # just outside the touchlines (y = 0 and 80)
        heights = _spread(side["x"].to_numpy(), LABEL_GAP, 4, 116)
        for n, label_y in zip(side.itertuples(), heights):
            radius = np.sqrt(scales.node_size(n.completed_p90) / np.pi)
            name = display_names.get(n.player, n.player)
            # A vertical pitch draws StatsBomb (x, y) at screen position (y, x)
            ax.annotate(name, xy=(n.y, n.x), xytext=(label_x, label_y),
                        ha="right" if left else "left", va="bottom",
                        fontsize=9, fontweight="bold", color=TEXT, zorder=4,
                        arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.6,
                                        shrinkA=0, shrinkB=radius + 1))
            ax.text(label_x, label_y - 0.8,
                    f"{n.share:.0%} of mins\n{n.completion:.0%} cmp · {n.xT_p90:.2f} xT",
                    ha="right" if left else "left", va="top",
                    fontsize=7, color=MUTED, linespacing=1.3, zorder=4)


def plot_network(ax, pitch, nodes, edges, scales, title, subtitle="", display_names=None):
    """Draw one pass network (nodes and edges of a single formation) on an already-drawn pitch."""
    display_names = display_names or {}

    # Grass only inside the touchlines, so the label margins keep the background
    ax.add_patch(plt.Rectangle((0, 0), 80, 120, color=GRASS, zorder=0))

    # Edges first so nodes sit on top; most threatening links drawn last
    for e in edges.sort_values("xT_per_pass").itertuples():
        pitch.lines(e.x_a, e.y_a, e.x_b, e.y_b, ax=ax, lw=scales.edge_width(e.completed_p90),
                    color=CMAP(scales.edge_norm(e.xT_per_pass)), alpha=0.85, zorder=2)

    pitch.scatter(nodes["x"], nodes["y"], ax=ax, s=scales.node_size(nodes["completed_p90"]),
                  c=nodes["xT_p90"], cmap=CMAP, norm=scales.node_norm,
                  edgecolors="white", linewidth=1.5, zorder=3)

    _label_nodes(ax, nodes, scales, display_names)

    ax.text(40, 128, title, ha="center", va="bottom", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(40, 123, subtitle, ha="center", va="bottom", fontsize=8, color=MUTED)


def _draw_legend(ax, scales):
    """Footer: two colour bars, node sizes, link widths and credits (axes coordinates 0-1)."""
    def heading(x, text):
        ax.text(x, 0.95, text, fontsize=8, color=TEXT, va="top", transform=ax.transAxes)

    # Colour bars: node and link xT use different scales
    for x, norm, text in [(0.0, scales.node_norm, "Passing xT p90 (node)"),
                          (0.22, scales.edge_norm, "xT per pass (link)")]:
        heading(x, text)
        cax = ax.inset_axes([x, 0.45, 0.17, 0.12])
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
        cax.xaxis.set_major_locator(MaxNLocator(3))   # a few round-numbered ticks
        cbar.outline.set_visible(False)
        cax.tick_params(labelsize=7, colors=MUTED, length=0)

    # Node sizes and link widths, drawn with the same scales as the plot
    heading(0.46, "Passes completed p90")
    for i, value in enumerate([15, 35, 55]):
        x = 0.48 + i * 0.06
        ax.scatter(x, 0.5, s=scales.node_size(value), color=LEGEND_GREY, edgecolors="white",
                   linewidth=1.5, transform=ax.transAxes, clip_on=False)
        ax.text(x, 0.05, f"{value}", ha="center", fontsize=7, color=MUTED, transform=ax.transAxes)

    heading(0.68, "Passes exchanged p90")
    for i, value in enumerate([5, 10, 20]):
        x = 0.68 + i * 0.065
        ax.plot([x, x + 0.045], [0.5, 0.5], lw=scales.edge_width(value), color=LEGEND_GREY,
                solid_capstyle="butt", transform=ax.transAxes)
        ax.text(x + 0.0225, 0.05, f"{value}", ha="center", fontsize=7, color=MUTED, transform=ax.transAxes)

    ax.text(1, 0.05, "Data: StatsBomb open data\nxT grid: Karun Singh", ha="right", va="bottom",
            fontsize=7, color=MUTED, linespacing=1.5, transform=ax.transAxes)


def plot_formations(nodes, edges, formation_minutes, formations, title, subtitle,
                    display_names=None, min_edge_p90=4):
    """Side-by-side pass networks, one per formation, with a title block and legend. Returns the figure.

    Links with fewer than `min_edge_p90` completed passes per 90 are hidden.
    `display_names` maps StatsBomb player names to the shorter names shown in labels.
    """
    edges = edges[edges["completed_p90"] >= min_edge_p90]
    scales = Scales.from_tables(nodes, edges)

    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color="none", line_color=PITCH_LINES,
                          linewidth=1, spot_scale=0.01, line_zorder=1,
                          pad_left=LABEL_MARGIN, pad_right=LABEL_MARGIN, pad_top=12, pad_bottom=2)
    fig, axs = pitch.grid(nrows=1, ncols=len(formations), figheight=12, grid_height=0.76,
                          title_height=0.1, endnote_height=0.08, space=0.02, axis=False)

    # Background gradient: an image filling the whole figure, behind every other axes
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto", extent=[0, 1, 0, 1],
              cmap=LinearSegmentedColormap.from_list("bg", BACKGROUND))
    bg.axis("off")

    for ax, formation in zip(np.atleast_1d(axs["pitch"]), formations):
        minutes = formation_minutes.loc[formation, "minutes"]
        plot_network(ax, pitch,
                     nodes[nodes["formation"] == formation],
                     edges[edges["formation"] == formation],
                     scales, title=formation, subtitle=f"{minutes:.0f} minutes",
                     display_names=display_names)

    # Title block
    title_ax = axs["title"]
    title_ax.text(0, 0.72, title, fontsize=24, fontweight="bold", color=TEXT, va="center")
    title_ax.text(0, 0.3, subtitle, fontsize=10, color=MUTED, va="center")
    title_ax.axhline(0.05, xmin=0, xmax=1, color=TEXT, lw=0.8)

    _draw_legend(axs["endnote"], scales)
    return fig
