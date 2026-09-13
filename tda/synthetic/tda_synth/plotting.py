"""Figure style and plotting helpers, sized for journal submission."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Fixed categorical order. Do not reorder: the color follows the series.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

#: Single-column and double-column widths in inches, the usual journal grid.
COLUMN = 3.46
DOUBLE = 7.20

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.2,
    "legend.columnspacing": 1.2,
    "lines.linewidth": 1.6,
    "lines.markersize": 3.5,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
mpl.rcParams.update(RC)


def series_color(index: int) -> str:
    """Color for series ``index``. A fifth series is an error, not a wrap."""
    if index >= len(PALETTE):
        raise ValueError(
            f"{index + 1} series but only {len(PALETTE)} colors; use small "
            "multiples instead of cycling the palette"
        )
    return PALETTE[index]


def _finish(ax, *, axis="y"):
    """Recessive grid behind the marks. ``axis`` is "y" or "both"."""
    ax.grid(axis=axis, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)


def _label_last(ax, x, y, text, color, dy=0.0):
    """Direct label at the last point; identity is never color-alone."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=7,
        color=color,
        va="center",
        ha="left",
        clip_on=False,
        annotation_clip=False,
    )


def sensitivity_plot(
    x_values,
    series: dict[str, list[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path=None,
    reference_x=None,
    percent: bool = True,
):
    """Signed or absolute error against a swept nuisance parameter."""
    fig, ax = plt.subplots(figsize=(COLUMN * 1.18, COLUMN * 0.78),
                           constrained_layout=True)
    scale = 100.0 if percent else 1.0
    ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=1)
    drawn = []
    for index, (label, values) in enumerate(series.items()):
        color = series_color(index)
        y = [scale * v for v in values]
        ax.plot(x_values, y, marker="o", color=color, label=label,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        drawn.append((y[-1], label, color))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, pad=6)
    ax.set_xticks(list(x_values))
    ax.set_xlim(min(x_values) - 0.04 * (max(x_values) - min(x_values)),
                max(x_values) + 0.18 * (max(x_values) - min(x_values)))
    _finish(ax)
    ax.legend(loc="upper left", ncol=len(series), bbox_to_anchor=(0.0, 1.0))
    # Stack the end labels in value order so overlapping series stay readable.
    span = max(1e-12, ax.get_ylim()[1] - ax.get_ylim()[0])
    drawn.sort(key=lambda r: r[0])
    previous = None
    for value, label, color in drawn:
        dy = 0.0
        if previous is not None and abs(value - previous) < 0.05 * span:
            dy = 7.0
        _label_last(ax, list(x_values)[-1], value, label, color, dy)
        previous = value + (dy / 100.0) * span
    if reference_x is not None:
        ax.axvline(reference_x, color=GRID, linestyle=(0, (3, 3)),
                   linewidth=0.8, zorder=1)
        ax.annotate("assumed", xy=(reference_x, ax.get_ylim()[1]),
                    xytext=(2, -1), textcoords="offset points",
                    fontsize=6.5, color=MUTED, va="top", ha="left")
    if output_path is not None:
        fig.savefig(output_path)
    return fig, ax


def plot_loss_map(table: pd.DataFrame, output_path=None):
    """Coarse loss over the candidate grid."""
    pivot = table.pivot(index="rhat", columns="dhat", values="loss")
    fig, ax = plt.subplots(figsize=(COLUMN * 1.15, COLUMN * 0.9), constrained_layout=True)
    image = ax.imshow(
        np.log10(1.0 + pivot.to_numpy()),
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()],
        cmap="viridis_r",
        interpolation="nearest",
    )
    best = table.loc[table["loss"].idxmin()]
    ax.plot(best["dhat"], best["rhat"], marker="+", color="white",
            markersize=7, markeredgewidth=1.2, zorder=3)
    ax.set_xlabel(r"$\widehat{D}$")
    ax.set_ylabel(r"$\widehat{R}$")
    ax.set_title("Coarse loss surface", loc="left", pad=6)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(r"$\log_{10}(1+\mathrm{loss})$", fontsize=7)
    bar.ax.tick_params(labelsize=6.5, width=0.6, length=2)
    bar.outline.set_linewidth(0.6)
    if output_path is not None:
        fig.savefig(output_path)
    return fig, ax


def plot_diagram(diagram: dict[int, np.ndarray], output_path=None):
    """Persistence diagram, one panel per homology dimension."""
    n = len(diagram)
    fig, axes = plt.subplots(1, n, figsize=(COLUMN * 0.92 * n, COLUMN * 0.95),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for index, (ax, (dimension, points)) in enumerate(zip(axes, sorted(diagram.items()))):
        color = series_color(index)
        if len(points):
            low, high = float(np.min(points)), float(np.max(points))
            pad = max((high - low) * 0.08, 0.01)
            ax.plot([low - pad, high + pad], [low - pad, high + pad],
                    linestyle=(0, (3, 3)), color=MUTED, linewidth=0.7, zorder=1)
            ax.scatter(points[:, 0], points[:, 1], s=9, color=color,
                       edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"$H_{dimension}$   n = {len(points)}", loc="left", pad=5)
        ax.set_xlabel("birth")
        if index == 0:
            ax.set_ylabel("death")
        _finish(ax, axis="both")
    if output_path is not None:
        fig.savefig(output_path)
    return fig, axes


PANEL = "abcdefghij"


def field_panel(ax, field, *, title, cmap, vmin, vmax, panel=None):
    """One density field. Colorbars are shared per group, not per panel."""
    handle = ax.imshow(field, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
    label = f"({panel})  {title}" if panel is not None else title
    ax.set_title(label, loc="left", pad=4, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor(GRID)
    return handle


def shared_colorbar(fig, handle, axes, label):
    """One colorbar for a group of panels that share a scale."""
    bar = fig.colorbar(handle, ax=list(axes), fraction=0.028, pad=0.012)
    bar.set_label(label, fontsize=7)
    bar.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    bar.outline.set_linewidth(0.5)
    return bar


def grouped_bars(ax, groups, series, *, ylabel, title):
    """Grouped bars with a 2px surface gap and direct value labels."""
    positions = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width * 0.92,
                      color=series_color(index), label=label,
                      edgecolor="white", linewidth=0.8)
        for rect, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", xy=(rect.get_x() + rect.get_width() / 2,
                                            rect.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(positions, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6)
    _finish(ax)
    ax.legend(ncol=len(series))
    return ax
