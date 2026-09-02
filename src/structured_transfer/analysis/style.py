"""
Shared plot styling.

One place for palette and chrome, so every figure in the repository reads as one
system. Values are taken unchanged from a validated reference palette rather than
picked by eye.

Color assignment follows the job the color does:

* **Categorical** (identity: ID vs OOD, selfer vs transfer) -- fixed slot order,
  never cycled by rank. At most three slots are used anywhere in this repo, which
  is the subset that clears colorblind separation on the all-pairs test.
* **Sequential** (magnitude: OOD balanced accuracy in the partition heatmap,
  scratch depth in the alignment scatter) -- one hue, light to dark. Never a
  rainbow; a multi-hue ramp makes equal steps look unequal.
* **Status** -- reserved for the collapse annotations, never reused as a series.

Note this departs from the paper's own figures, which used a multi-hue magma ramp
for the heatmap. Single-hue is the correct encoding for continuous magnitude; the
underlying numbers are identical.

Figures are light-mode only, deliberately: they are publication artifacts destined
for a PDF and a README, not a themed web page.
"""

from __future__ import annotations

# ── Categorical slots (identity). Assign in order; never cycle. ───────────────
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA = SERIES

# ── Sequential ramp (magnitude): one hue, light -> dark. ─────────────────────
SEQUENTIAL = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: Index of the lightest step usable for *ordinal* encoding on the light surface.
#: The full ramp is for continuous magnitude on large filled areas (heatmap
#: cells), where the lightest step legitimately recedes toward the surface. Thin
#: marks -- scatter dots, lines -- need a contrast floor, so discrete ordered
#: series start here instead. Below this step a dot is barely distinguishable
#: from the background.
_ORDINAL_FLOOR = 3

# ── Status (reserved; never a series color). ─────────────────────────────────
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD     = "#0ca30c"

# ── Chrome and ink. Grid and axes stay one shade off the surface. ────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#898781"
GRIDLINE  = "#e1e0d9"
BASELINE  = "#c3c2b7"

#: Gap between adjacent bars, in bar-width units. A surface-colored gap separates
#: fills without drawing a border around every mark.
BAR_GAP = 0.04


def apply_style() -> None:
    """
    Install the house matplotlib style.

    Thin marks, solid hairline gridlines on the value axis only (dashed grid
    reads as "threshold" when it is just a grid), no top/right spines, generous
    padding, and tabular figures on tick labels so columns of numbers align.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
        "figure.facecolor":    SURFACE,
        "axes.facecolor":      SURFACE,
        "savefig.facecolor":   SURFACE,

        "font.family":         "sans-serif",
        "font.sans-serif":     ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size":           9,
        "axes.titlesize":      11,
        "axes.labelsize":      9.5,
        "xtick.labelsize":     8.5,
        "ytick.labelsize":     8.5,
        "legend.fontsize":     8.5,

        "text.color":          INK,
        "axes.labelcolor":     INK_2,
        "xtick.color":         INK_MUTED,
        "ytick.color":         INK_MUTED,
        "axes.titlecolor":     INK,

        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.edgecolor":      BASELINE,
        "axes.linewidth":      0.8,

        "axes.grid":           True,
        "axes.grid.axis":      "y",
        "grid.color":          GRIDLINE,
        "grid.linewidth":      0.8,
        "grid.linestyle":      "-",     # solid hairline, never dashed
        "grid.alpha":          1.0,
        "axes.axisbelow":      True,    # grid behind the marks

        "lines.linewidth":     2.0,
        "lines.markersize":    6.0,

        "legend.frameon":      False,
        "axes.titlepad":       10,
    })


def sequential_color(value: float, vmin: float, vmax: float) -> str:
    """
    Map a magnitude onto the sequential ramp.

    Clamped, so out-of-range values saturate at the ramp ends instead of wrapping.
    """
    if vmax <= vmin:
        return SEQUENTIAL[len(SEQUENTIAL) // 2]
    t = (value - vmin) / (vmax - vmin)
    idx = round(max(0.0, min(1.0, t)) * (len(SEQUENTIAL) - 1))
    return SEQUENTIAL[idx]


def ordinal_color(value: float, vmin: float, vmax: float) -> str:
    """
    Map a discrete ordered value onto the contrast-safe part of the ramp.

    Use this for thin marks -- scatter points, lines -- and for small discrete
    scales such as "number of scratch blocks, 0 to 5". Use
    :func:`sequential_color` only for large filled areas.
    """
    ramp = SEQUENTIAL[_ORDINAL_FLOOR:]
    if vmax <= vmin:
        return ramp[len(ramp) // 2]
    t = (value - vmin) / (vmax - vmin)
    idx = round(max(0.0, min(1.0, t)) * (len(ramp) - 1))
    return ramp[idx]


def sequential_cmap():
    """The full sequential ramp as a matplotlib colormap, for filled areas."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("st_blue", SEQUENTIAL)


def ordinal_cmap():
    """
    The contrast-safe sub-range as a colormap.

    A colorbar beside ordinal marks must be built from the same range the marks
    use, or the legend and the data disagree about what a colour means.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("st_blue_ord", SEQUENTIAL[_ORDINAL_FLOOR:])


def text_on(color_hex: str) -> str:
    """
    Readable ink for a label drawn on a filled mark.

    Uses relative luminance rather than a fixed threshold on the ramp index, so
    it stays correct if the ramp is ever swapped for a brand palette.
    """
    r, g, b = (int(color_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)
    return INK if luminance > 0.45 else "#ffffff"
