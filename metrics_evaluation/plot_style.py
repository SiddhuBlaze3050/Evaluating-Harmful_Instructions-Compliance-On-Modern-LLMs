"""
Publication-Quality Plot Styling — LLM Safety Evaluation

Centralises all matplotlib rcParams, color palettes, figure sizes and
helper utilities so every figure across the pipeline is visually
consistent and meets Nature / IEEE journal standards.

Usage
-----
    from plot_style import (
        apply_publication_style, style_axes, format_legend,
        condition_color, condition_marker, condition_label,
        short_model_name, FIGURE_DPI,
        FIG_SINGLE, FIG_SCATTER, FIG_WIDE, FIG_GRID_3x3,
        LINE_WIDTH, MARKER_SIZE, FONT_SIZE_ANNOT,
    )

    # Call once at the start of a pipeline run
    apply_publication_style()

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(x, y, color=condition_color('baseline'), ...)
    style_axes(ax)
    format_legend(ax, loc='upper right')
    fig.savefig('out.png', dpi=FIGURE_DPI, bbox_inches='tight')
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# ─── Color Palette ────────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette
CONDITION_COLORS: dict[str, str] = {
    "baseline":     "#D55E00",   # Terracotta / red
    "safedecoding": "#0072B2",   # Blue
    "llamaguard3":  "#009E73",   # Green
    "guardrail":    "#009E73",   # Alias for llamaguard3
}

CONDITION_MARKERS: dict[str, str] = {
    "baseline":     "o",
    "safedecoding": "s",
    "llamaguard3":  "^",
    "guardrail":    "^",
}

CONDITION_LABELS: dict[str, str] = {
    "baseline":     "Baseline",
    "safedecoding": "SafeDecoding",
    "llamaguard3":  "LlamaGuard-3",
    "guardrail":    "LlamaGuard-3",
}

MODEL_SHORT_NAMES: dict[str, str] = {
    "Gemini-2.5 Flash": "Gemini-2.5",
    "Llama-3.1-8B":     "Llama-3.1",
    "Mistral-7B":       "Mistral-7B",
}

# ─── Typography ───────────────────────────────────────────────────────────────
FONT_FAMILY     = "DejaVu Sans"
FONT_SIZE_TITLE = 13          # axes title (pt) — kept for optional internal use
FONT_SIZE_LABEL = 13          # axis label (pt)
FONT_SIZE_TICK  = 11          # tick label (pt)
FONT_SIZE_LEGEND = 11         # legend text (pt)
FONT_SIZE_ANNOT  = 11         # inline annotations (pt)

# ─── Lines & Markers ──────────────────────────────────────────────────────────
LINE_WIDTH  = 2.2
MARKER_SIZE = 7

# ─── Resolution ───────────────────────────────────────────────────────────────
FIGURE_DPI = 300              # print-ready raster DPI

# ─── Standard Figure Sizes (inches) ───────────────────────────────────────────
FIG_SINGLE   = (7.0, 5.0)    # double-column width, general use
FIG_SCATTER  = (6.5, 5.5)    # square-ish: Pareto / scatter plots
FIG_WIDE     = (12.0, 5.0)   # two panels side-by-side
FIG_GRID_3x3 = (13.0, 9.0)   # 3 × N facet grids


# ─── Core Style Application ───────────────────────────────────────────────────

def apply_publication_style() -> None:
    """
    Apply global matplotlib rcParams for publication-quality figures.

    Should be called **once** at the start of a pipeline run or script.
    Individual `style_axes()` calls handle per-axes cleanup in addition
    to these global defaults.
    """
    mpl.rcParams.update(
        {
            # ── Font ──────────────────────────────────────────────────────────
            "font.family":              FONT_FAMILY,
            "font.size":                FONT_SIZE_TICK,
            "axes.titlesize":           FONT_SIZE_TITLE,
            "axes.titleweight":         "bold",
            "axes.titlepad":            8,
            "axes.labelsize":           FONT_SIZE_LABEL,
            "axes.labelweight":         "normal",
            "xtick.labelsize":          FONT_SIZE_TICK,
            "ytick.labelsize":          FONT_SIZE_TICK,
            "legend.fontsize":          FONT_SIZE_LEGEND,
            "legend.title_fontsize":    FONT_SIZE_LEGEND,
            # ── Lines / Markers ───────────────────────────────────────────────
            "lines.linewidth":          LINE_WIDTH,
            "lines.markersize":         MARKER_SIZE,
            # ── Axes ──────────────────────────────────────────────────────────
            "axes.linewidth":           0.8,
            "axes.spines.top":          False,
            "axes.spines.right":        False,
            "axes.grid":                False,   # controlled per-plot via style_axes()
            # ── Ticks ─────────────────────────────────────────────────────────
            "xtick.direction":          "out",
            "ytick.direction":          "out",
            "xtick.major.width":        0.8,
            "ytick.major.width":        0.8,
            "xtick.major.size":         4.0,
            "ytick.major.size":         4.0,
            "xtick.minor.size":         2.0,
            "ytick.minor.size":         2.0,
            # ── Legend ────────────────────────────────────────────────────────
            "legend.frameon":           True,
            "legend.framealpha":        0.9,
            "legend.edgecolor":         "#CCCCCC",
            "legend.fancybox":          False,
            "legend.handlelength":      1.5,
            "legend.handletextpad":     0.5,
            "legend.labelspacing":      0.4,
            "legend.columnspacing":     1.0,
            # ── Saving ────────────────────────────────────────────────────────
            "savefig.dpi":              FIGURE_DPI,
            "savefig.bbox":             "tight",
            "savefig.pad_inches":       0.05,
            # ── Figure ────────────────────────────────────────────────────────
            "figure.dpi":               100,     # screen preview DPI
            "figure.facecolor":         "white",
            "axes.facecolor":           "white",
            "savefig.facecolor":        "white",
        }
    )


# ─── Per-Axes Helpers ─────────────────────────────────────────────────────────

def style_axes(ax: plt.Axes, grid: str = "y") -> None:
    """
    Apply clean, publication-style formatting to a single Axes object.

    Removes the top and right spines, sets outward ticks, and optionally
    adds a subtle dotted grid on the requested axis.

    Args:
        ax:   The matplotlib ``Axes`` instance to style.
        grid: Which axis to draw grid lines on — ``'x'``, ``'y'``,
              ``'both'``, or ``'none'``.
    """
    # Spines (belt-and-suspenders: rcParams + explicit)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)

    # Ticks
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)

    # Grid
    ax.grid(False)
    if grid != "none":
        ax.grid(True, axis=grid, alpha=0.22, linestyle=":", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def format_legend(
    ax: plt.Axes,
    loc: str = "best",
    title: Optional[str] = None,
    ncols: int = 1,
) -> None:
    """
    Apply consistent publication-style legend formatting.

    Args:
        ax:    The ``Axes`` whose legend to format.
        loc:   Legend location string (e.g. ``'upper right'``).
        title: Optional legend title.
        ncols: Number of columns in the legend.
    """
    legend = ax.legend(
        loc=loc,
        title=title,
        ncols=ncols,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        fancybox=False,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.6)


# ─── Lookup Utilities ─────────────────────────────────────────────────────────

def condition_color(condition: str) -> str:
    """Return the canonical publication color for a condition name."""
    return CONDITION_COLORS.get(condition.lower(), "#7F7F7F")


def condition_marker(condition: str) -> str:
    """Return the canonical marker style for a condition name."""
    return CONDITION_MARKERS.get(condition.lower(), "o")


def condition_label(condition: str) -> str:
    """Return a human-readable display label for a condition name."""
    return CONDITION_LABELS.get(condition.lower(), condition.replace("_", " ").title())


def short_model_name(model: str) -> str:
    """Return a shortened model name suitable for inline plot annotations."""
    return MODEL_SHORT_NAMES.get(model, model.split("-")[0])
