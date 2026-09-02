"""
Figure generation for both experiments.

Every figure reads either the archived run artifacts or the reference result
tables; a figure whose inputs are unavailable is skipped with an explicit message
rather than approximated.

Figure coverage
---------------
iWildCam (from the 21 archived runs or the reference leaderboard):

  ``partition_heatmap``    paper Figure 1b -- OOD balanced accuracy over F/T/S
  ``early_layer_generality``  Figure 7  (H1) -- what freezing early blocks costs
  ``stage4_specificity``      Figure 8  (H2) -- Stage4 as F vs S vs T
  ``t_buffer_benefit``        Figure 9  (H4) -- direct F->S vs an inserted T buffer
  ``id_ood_alignment``        Figure 10 -- ID and OOD move together
  ``training_curves``         (extra, run artifacts only) -- per-epoch OOD trajectories

TinyImageNet (from the reference selfer and transfer tables):

  ``yosinski_replication``    Figure 1a -- the four depth curves
  ``coadaptation_fragility``  Figure 3  (H1) -- selfer pure-frozen dip at n=3
  ``specificity_collapse``    Figure 4  (H3) -- transfer frozen collapse at n=4
  ``finetuning_recovery``     Figure 5  (H2) -- fine-tuning recovers, except TTTS

Two further figures require per-run data rather than aggregates:

  Figure 2 (A/B symmetry check)   needs per-*direction* results (AnB vs BnA).
  Figure 6 (ID/OOD-C correlation) needs all 56 individual runs.

The reference tables report the mean over directions, so both are generated only
once ``scripts/tinyimagenet_run.py`` has populated ``results/tinyimagenet/runs/``
-- see :func:`tinyimagenet_per_run_figures`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .load import (
    RESULTS_DIR,
    load_leaderboard,
    load_run_curves,
    load_tinyimagenet_runs,
    load_tinyimagenet_tables,
)

#: Backbone block names, used for axis labels on the depth curves.
_WRN_BLOCK_LABELS = ["0\n(scratch)", "1\n(+Conv1)", "2\n(+Group1)",
                     "3\n(+Group2)", "4\n(+Group3)"]


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    # Print a repo-relative path when we can; a custom --results_dir may sit
    # outside the repository, in which case relative_to would raise.
    try:
        shown = path.relative_to(RESULTS_DIR.parent)
    except ValueError:
        shown = path
    print(f"  wrote {shown}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# iWildCam
# ═══════════════════════════════════════════════════════════════════════════

def partition_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 1b -- the structured transfer map.

    Rows are frozen depth ``a``, columns fine-tuned depth ``b``; scratch depth is
    implied, ``c = 5 - a - b``, so the triangle below the anti-diagonal is the
    whole valid configuration space. Cell fill is OOD balanced accuracy on a
    single-hue sequential ramp, since it encodes continuous magnitude.

    Reading it: performance rises to the right (more adaptation) far more
    steeply than it rises downward (more freezing) -- adaptation is what this
    domain shift demands.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    vmin, vmax = df["ood_bal_acc"].min(), df["ood_bal_acc"].max()

    for config, row in df.iterrows():
        a, b = int(row["frozen_depth"]), int(row["tuned_depth"])
        value = row["ood_bal_acc"]

        color = style.sequential_color(value, vmin, vmax)
        # A 2px surface gap separates fills instead of a border on every cell.
        ax.add_patch(plt.Rectangle((b - 0.48, a - 0.48), 0.96, 0.96,
                                   facecolor=color, edgecolor=style.SURFACE,
                                   linewidth=2.0))
        ink = style.text_on(color)
        ax.text(b, a - 0.10, config, ha="center", va="center",
                fontsize=8, fontweight="bold", color=ink)
        ax.text(b, a + 0.16, f"{value:.1f}", ha="center", va="center",
                fontsize=8.5, color=ink)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)          # frozen depth increases downward
    ax.set_xticks(range(6), [f"T={i}" for i in range(6)])
    ax.set_yticks(range(6), [f"F={i}" for i in range(6)])
    ax.set_xlabel("Fine-tuned backbone blocks  (T)")
    ax.set_ylabel("Frozen backbone blocks  (F)")
    ax.set_title("Structured transfer map: OOD balanced accuracy over F/T/S partitions",
                 loc="left")
    ax.grid(False)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Scale legend: a continuous encoding needs one.
    scalar_map = plt.cm.ScalarMappable(cmap=style.sequential_cmap(),
                                       norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("OOD balanced accuracy (%)", color=style.INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=style.INK_MUTED, labelcolor=style.INK_MUTED, length=0)

    ax.text(0, 1.045,
            "Each cell is one monotonic partition F^a T^b S^c; scratch blocks are "
            "implicit, S = 5 - F - T. The head is always scratch.",
            transform=ax.transAxes, fontsize=7.8, color=style.INK_MUTED)

    return _save(fig, out_dir, "fig1b_partition_heatmap")


def early_layer_generality(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 7 (H1) -- how much does freezing early blocks cost?

    Left: hold the later stages adapted and push the frozen prefix deeper
    (T5S1 -> F1T4S1 -> F2T3S1). Accuracy barely moves, so early features really
    are generic and reusable.

    Right: the pure-frozen series, replacing each frozen block with a scratch one
    (F5S1 -> F4S2 -> ... -> S6). This is the counter-story: freezing is only
    cheap when the *later* stages are still free to adapt. The fully frozen
    backbone is the worst configuration in the sweep, below training from scratch.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    panels = [
        (axes[0], ["T5S1", "F1T4S1", "F2T3S1"],
         "Freezing early blocks is nearly free\n(Stage2-4 always fine-tuned)"),
        (axes[1], ["F5S1", "F4S2", "F3S3", "F2S4", "F1S5", "S6"],
         "Replacing frozen blocks with scratch\n(pure-frozen series)"),
    ]

    for ax, configs, title in panels:
        configs = [c for c in configs if c in df.index]
        x = np.arange(len(configs))
        width = 0.42 - style.BAR_GAP

        id_vals  = [df.loc[c, "id_acc"] for c in configs]
        ood_vals = [df.loc[c, "ood_acc"] for c in configs]

        ax.bar(x - width / 2 - style.BAR_GAP / 2, id_vals, width,
               label="ID accuracy", color=style.SERIES_BLUE)
        ax.bar(x + width / 2 + style.BAR_GAP / 2, ood_vals, width,
               label="OOD accuracy", color=style.SERIES_ORANGE)

        # Direct-label only the bar tops -- the values are the point of the panel
        # and there are few enough that this does not become noise.
        for xi, (v_id, v_ood) in zip(x, zip(id_vals, ood_vals)):
            ax.text(xi - width / 2 - style.BAR_GAP / 2, v_id + 1.2, f"{v_id:.1f}",
                    ha="center", fontsize=7.5, color=style.INK_2)
            ax.text(xi + width / 2 + style.BAR_GAP / 2, v_ood + 1.2, f"{v_ood:.1f}",
                    ha="center", fontsize=7.5, color=style.INK_2)

        # The all-scratch control, as a reference line rather than a rival series.
        if "S6" in df.index:
            ax.axhline(df.loc["S6", "ood_acc"], color=style.INK_MUTED,
                       linewidth=1.0, zorder=1)
            ax.text(len(configs) - 0.45, df.loc["S6", "ood_acc"] + 0.8,
                    "S6 OOD baseline", ha="right", fontsize=7.5,
                    color=style.INK_MUTED)

        ax.set_xticks(x, configs)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 85)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", ncols=2)

    fig.suptitle("H1 - Early layers are general; deep layers are not",
                 x=0.008, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_early_layer_generality")


def stage4_specificity(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 8 (H2) -- Stage4 decides the outcome.

    Every configuration, grouped by what happens to the last residual stage:
    frozen, scratch, or fine-tuned. The ordering within each group is by frozen
    depth, so the comparison across groups is like-for-like.

    The result is the paper's sharpest finding: fine-tuning Stage4 dominates both
    alternatives. Frozen locks the model into ImageNet semantics; scratch throws
    away 63% of the pretrained parameters and must relearn high-level features
    from camera-trap data alone. Only fine-tuning combines pretrained structure
    with target adaptation.
    """
    # Stage4's state = the 5th backbone block. F when frozen_depth == 5,
    # T when frozen_depth + tuned_depth == 5, else S.
    def stage4_state(row) -> str:
        if row["frozen_depth"] == 5:
            return "F"
        return "T" if row["frozen_depth"] + row["tuned_depth"] == 5 else "S"

    df = df.copy()
    df["stage4"] = df.apply(stage4_state, axis=1)

    groups = [("F", "Stage4 frozen"), ("S", "Stage4 scratch"), ("T", "Stage4 fine-tuned")]
    colors = dict(zip("FST", (style.SERIES_AQUA, style.SERIES_ORANGE, style.SERIES_BLUE)))

    fig, ax = plt.subplots(figsize=(12.5, 5.0))

    positions, labels, gap = [], [], 0.9
    cursor = 0.0
    for state, _ in groups:
        subset = df[df["stage4"] == state].sort_values("frozen_depth")
        for config in subset.index:
            positions.append(cursor)
            labels.append(config)
            cursor += 1.0
        cursor += gap

    for pos, config in zip(positions, labels):
        row   = df.loc[config]
        color = colors[row["stage4"]]
        ax.bar(pos, row["ood_acc"], 1.0 - style.BAR_GAP * 2,
               color=color, zorder=3)
        ax.text(pos, row["ood_acc"] + 0.9, f"{row['ood_acc']:.1f}",
                ha="center", fontsize=7.5, color=style.INK_2)

    if "S6" in df.index:
        ax.axhline(df.loc["S6", "ood_acc"], color=style.INK_MUTED, linewidth=1.0, zorder=1)
        ax.text(positions[-1] + 0.6, df.loc["S6", "ood_acc"] + 0.7,
                "S6 (all scratch)", ha="right", fontsize=7.5, color=style.INK_MUTED)

    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("OOD accuracy (%)")
    ax.set_ylim(0, 62)
    ax.set_title("H2 - Top-layer specificity: how Stage4 is treated decides OOD performance",
                 loc="left", fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s, _ in groups]
    ax.legend(handles, [name for _, name in groups], loc="upper left", ncols=3)

    fig.tight_layout()
    return _save(fig, out_dir, "fig8_stage4_specificity")


def t_buffer_benefit(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 9 (H4) -- does a fine-tuned buffer between F and S help?

    One panel per frozen depth ``a``. Within a panel, ``b = 0`` is a direct
    F -> S boundary; ``b > 0`` inserts fine-tuned blocks between the frozen
    prefix and the scratch suffix. If the buffer hypothesis holds, adding T
    blocks should improve OOD accuracy monotonically -- which is what the panels
    show, with the gain concentrated in the step that converts the last scratch
    stage into a fine-tuned one.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    for ax, a in zip(axes.flat, (1, 2, 3, 4)):
        subset = df[df["frozen_depth"] == a].sort_values("tuned_depth")
        if subset.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(subset))
        ax.bar(x, subset["ood_acc"], 1.0 - style.BAR_GAP * 6,
               color=[style.SERIES_ORANGE if b == 0 else style.SERIES_BLUE
                      for b in subset["tuned_depth"]], zorder=3)

        # Sorting by tuned_depth puts b = 0 first, and it always exists for
        # a in 1..4 (since a + b + c = 5 with c >= 0), so this is the direct
        # F -> S reference the panel measures gains against.
        baseline = subset.iloc[0]["ood_acc"]
        for xi, (_, row) in zip(x, subset.iterrows()):
            ax.text(xi, row["ood_acc"] + 0.8, f"{row['ood_acc']:.1f}",
                    ha="center", fontsize=8, color=style.INK_2)
            if row["tuned_depth"] > 0:
                delta = row["ood_acc"] - baseline
                ax.text(xi, row["ood_acc"] + 3.4, f"{delta:+.1f} pp",
                        ha="center", fontsize=7.5,
                        color=style.STATUS_GOOD if delta > 0 else style.STATUS_CRITICAL)

        ax.set_xticks(x, [f"{c}\nb={int(r['tuned_depth'])}"
                          for c, r in subset.iterrows()], fontsize=8)
        ax.set_ylabel("OOD accuracy (%)")
        ax.set_ylim(0, 62)
        frozen_names = ", ".join(["Stem", "Stage1", "Stage2", "Stage3"][:a])
        ax.set_title(f"a = {a} frozen  ({frozen_names})", loc="left")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=style.SERIES_ORANGE),
        plt.Rectangle((0, 0), 1, 1, color=style.SERIES_BLUE),
    ]
    fig.legend(handles, ["b = 0  (direct F -> S)", "b > 0  (T buffer inserted)"],
               loc="upper right", ncols=2, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("H4 - A fine-tuned buffer between frozen and scratch regions helps",
                 x=0.008, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "fig9_t_buffer_benefit")


def id_ood_alignment(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 10 -- ID and OOD performance move together.

    Balanced accuracy, ID against OOD, one point per configuration. Color encodes
    scratch depth -- an *ordered* quantity, hence the sequential ramp -- and
    marker size encodes trainable parameter share.

    Every point sits below the identity line: OOD is uniformly harder. What the
    sweep does *not* show is any point above it, or any trade-off regime where a
    configuration buys OOD robustness with ID accuracy. Weak partitions simply
    fail at both, which says the F/T/S choice controls representation quality
    rather than an overfitting/robustness balance.
    """
    fig, ax = plt.subplots(figsize=(7.6, 6.8))

    lo = min(df["id_bal_acc"].min(), df["ood_bal_acc"].min()) - 2.5
    hi = max(df["id_bal_acc"].max(), df["ood_bal_acc"].max()) + 3.0
    ax.plot([lo, hi], [lo, hi], color=style.BASELINE, linewidth=1.0, zorder=1)
    ax.text(hi - 0.5, hi - 0.5, "ID = OOD", rotation=45, ha="right", va="bottom",
            fontsize=8, color=style.INK_MUTED)

    smin, smax = df["scratch_depth"].min(), df["scratch_depth"].max()
    for _, row in df.iterrows():
        # Scratch depth is a small discrete ordered scale (0-5) shown on thin
        # marks, so it uses the contrast-safe part of the ramp: the lightest
        # sequential steps vanish against the surface at scatter-dot size.
        color = style.ordinal_color(row["scratch_depth"], smin, smax)
        size  = 60 + 260 * (row["trainable_pct"] / 100.0)
        # A 2px surface ring keeps overlapping points readable without a border
        # around every mark.
        ax.scatter(row["id_bal_acc"], row["ood_bal_acc"], s=size, color=color,
                   edgecolors=style.SURFACE, linewidths=2.0, zorder=3)

    # Direct-label only the configurations the discussion names; the rest are
    # readable from the leaderboard CSV (the table view).
    for config in ("T5S1", "F3T2S1", "F2T3S1", "F4T1S1", "F5S1", "S6"):
        if config not in df.index:
            continue
        row = df.loc[config]
        ax.annotate(config, (row["id_bal_acc"], row["ood_bal_acc"]),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=8.5, fontweight="bold", color=style.INK)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ID balanced accuracy (%)")
    ax.set_ylabel("OOD balanced accuracy (%)")
    ax.set_title("ID and OOD performance move together, but OOD stays harder",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="both")
    ax.set_aspect("equal")

    # Built from the same sub-range the marks use, so the legend and the data
    # agree about what a colour means.
    scalar_map = plt.cm.ScalarMappable(cmap=style.ordinal_cmap(),
                                       norm=plt.Normalize(smin, smax))
    cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Scratch backbone blocks", color=style.INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=style.INK_MUTED, labelcolor=style.INK_MUTED, length=0)

    ax.text(0, -0.13, "Marker size is the share of trainable parameters (1.6% - 100%).",
            transform=ax.transAxes, fontsize=7.8, color=style.INK_MUTED)

    fig.tight_layout()
    return _save(fig, out_dir, "fig10_id_ood_alignment")


def training_curves(
    out_dir: Path,
    runs_dir: Path | None = None,
    configs: tuple[str, ...] = ("T5S1", "F3T2S1", "F4T1S1", "F4S2", "F5S1", "S6"),
) -> Path | None:
    """
    Per-epoch OOD trajectories -- an extra, not a paper figure.

    Only the archived run artifacts carry per-epoch logs, so this is the one
    figure that ignores the paper tables entirely. It is worth having because it
    shows what an endpoint table cannot: F5S1 plateaus almost immediately (it has
    1.6% trainable parameters and nothing left to learn), while the adaptive
    configurations are still climbing when the time budget cuts them off.
    """
    try:
        curves = load_run_curves(runs_dir)
    except FileNotFoundError:
        print("  skipped training_curves: no run artifacts found")
        return None
    if curves.empty:
        print("  skipped training_curves: run artifacts contain no eval_log entries")
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    available = [c for c in configs if c in set(curves["config"])]
    # Ordered emphasis: one ramp over a fixed, meaningful ordering of
    # configurations, not one hue per series cycled by rank. The ordinal
    # sub-range is required here -- a 2px line in the lightest sequential step
    # is unreadable on the light surface.
    for i, config in enumerate(available):
        subset = curves[curves["config"] == config].sort_values("epoch")
        color = style.ordinal_color(i, 0, max(len(available) - 1, 1))
        ax.plot(subset["epoch"], subset["ood_bal_acc"], color=color, zorder=3)
        last = subset.iloc[-1]
        ax.annotate(config, (last["epoch"], last["ood_bal_acc"]),
                    textcoords="offset points", xytext=(6, -2),
                    fontsize=8.5, fontweight="bold", color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("OOD balanced accuracy (%)")
    ax.set_title("OOD trajectories under the shared time budget",
                 loc="left", fontweight="bold")
    ax.text(0, 1.03,
            "Runs stop on wall clock, not epoch count, so curves end at different "
            "epochs. T5S1 is the slowest and stops earliest.",
            transform=ax.transAxes, fontsize=7.8, color=style.INK_MUTED)

    fig.tight_layout()
    return _save(fig, out_dir, "extra_training_curves")


# ═══════════════════════════════════════════════════════════════════════════
# TinyImageNet
# ═══════════════════════════════════════════════════════════════════════════

def yosinski_replication(selfer: pd.DataFrame, transfer: pd.DataFrame,
                         out_dir: Path) -> Path:
    """
    Paper Figure 1a -- the four depth curves, Yosinski-style.

    x is the number of blocks copied from the source network; y is clean top-1.
    Four curves, encoded compositionally rather than with four hues: color
    separates selfer from transfer, line style separates frozen from fine-tuned.
    That keeps the palette at two slots and makes the 2x2 design of the
    experiment legible from the chart itself.

    The whole story is in the right-hand end. Three curves return to baseline at
    n = 4; the transfer-frozen curve collapses to 53.1%. Copying the deepest
    residual group works only when the source and target label sets match.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    series = [
        (selfer,   ["SSSS", "FSSS", "FFSS", "FFFS", "FFFF"],
         style.SERIES_BLUE,   "-",  "Selfer - frozen"),
        (selfer,   ["SSSS", "TSSS", "TTSS", "TTTS", "TTTT"],
         style.SERIES_BLUE,   "--", "Selfer - fine-tuned"),
        (transfer, ["SSSS", "FSSS", "FFSS", "FFFS", "FFFF"],
         style.SERIES_ORANGE, "-",  "Transfer - frozen"),
        (transfer, ["SSSS", "TSSS", "TTSS", "TTTS", "TTTT"],
         style.SERIES_ORANGE, "--", "Transfer - fine-tuned"),
    ]

    for table, configs, color, linestyle, label in series:
        ys = [table.loc[c, "clean_acc"] for c in configs if c in table.index]
        xs = list(range(len(ys)))
        ax.plot(xs, ys, color=color, linestyle=linestyle, marker="o",
                markersize=6, markeredgecolor=style.SURFACE, markeredgewidth=2.0,
                label=label, zorder=3)

    baseline = selfer.loc["SSSS", "clean_acc"]
    ax.axhline(baseline, color=style.INK_MUTED, linewidth=1.0, zorder=1)
    ax.text(0.02, baseline + 0.5, f"Scratch baseline ({baseline:.1f}%)",
            fontsize=7.8, color=style.INK_MUTED)

    # One direct label, on the single point the figure exists to show.
    collapse = transfer.loc["FFFF", "clean_acc"]
    ax.annotate(f"FFFF transfer collapse\n{collapse:.1f}%",
                xy=(4, collapse), xytext=(3.15, collapse + 4.5),
                fontsize=8.5, color=style.STATUS_CRITICAL, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-", color=style.STATUS_CRITICAL, lw=1.2))

    ax.set_xticks(range(5), _WRN_BLOCK_LABELS)
    ax.set_xlabel("Blocks copied from the source network  (n)")
    ax.set_ylabel("Top-1 validation accuracy (%)")
    ax.set_ylim(50, 78)
    ax.set_title("Yosinski replication - WRN-28-10 on TinyImageNet",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left", ncols=2)

    fig.tight_layout()
    return _save(fig, out_dir, "fig1a_yosinski_replication")


def coadaptation_fragility(selfer: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 3 (H1) -- co-adaptation fragility in the selfer frozen curve.

    Source and target tasks are identical here, so nothing about the *domain* can
    explain a drop. The dip at n = 3 (71.8%, from a 73.7% baseline) is purely
    optimization: freezing through Group2 splits a set of co-adapted layers, and
    the still-scratch Group3 has to relearn against a fixed representation it did
    not co-evolve with. Freezing all four recovers to 73.5%, because then no
    co-adapted pair is split at all.

    The effect is real but small -- roughly 2 points. WRN's residual connections
    appear to soften it relative to the original AlexNet result.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    configs = ["SSSS", "FSSS", "FFSS", "FFFS", "FFFF"]
    ys = [selfer.loc[c, "clean_acc"] for c in configs]
    xs = list(range(len(ys)))

    ax.plot(xs, ys, color=style.SERIES_BLUE, marker="o", markersize=7,
            markeredgecolor=style.SURFACE, markeredgewidth=2.0, zorder=3)

    baseline = selfer.loc["SSSS", "clean_acc"]
    ax.axhline(baseline, color=style.INK_MUTED, linewidth=1.0, zorder=1)
    ax.text(0.02, baseline + 0.12, f"Scratch baseline ({baseline:.1f}%)",
            fontsize=8, color=style.INK_MUTED)

    dip = int(np.argmin(ys))
    ax.annotate(f"min at n = {dip}\n{ys[dip]:.1f}%",
                xy=(dip, ys[dip]), xytext=(dip + 0.35, ys[dip] - 0.9),
                fontsize=8.5, color=style.SERIES_BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=style.SERIES_BLUE, lw=1.2))

    ax.set_xticks(range(5), _WRN_BLOCK_LABELS)
    ax.set_xlabel("Blocks frozen from the source network  (n)")
    ax.set_ylabel("Top-1 validation accuracy (%)")
    # A single series needs no legend box; the title names it.
    ax.set_title("H1 - Co-adaptation fragility (selfer, pure frozen)",
                 loc="left", fontweight="bold")
    ax.text(0, 1.03,
            "Source and target tasks are identical, so this dip is optimization, "
            "not domain mismatch.",
            transform=ax.transAxes, fontsize=7.8, color=style.INK_MUTED)

    fig.tight_layout()
    return _save(fig, out_dir, "fig3_coadaptation_fragility")


def specificity_collapse(selfer: pd.DataFrame, transfer: pd.DataFrame,
                         out_dir: Path) -> Path:
    """
    Paper Figure 4 (H3) -- feature specialization, isolated.

    The same frozen curve, run same-domain and cross-domain. They track each
    other almost exactly through n = 3 and then diverge by 20 points at n = 4.
    Since the two conditions differ only in whether the source and target class
    sets match, the shaded gap *is* the task-specificity of Group3's features.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    configs = ["SSSS", "FSSS", "FFSS", "FFFS", "FFFF"]
    xs = list(range(len(configs)))
    selfer_ys   = [selfer.loc[c, "clean_acc"] for c in configs]
    transfer_ys = [transfer.loc[c, "clean_acc"] for c in configs]

    ax.fill_between(xs, transfer_ys, selfer_ys, color=style.SERIES_ORANGE,
                    alpha=0.12, zorder=2)
    ax.plot(xs, selfer_ys, color=style.SERIES_BLUE, marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=2.0,
            label="Selfer  (A->A, B->B)", zorder=3)
    ax.plot(xs, transfer_ys, color=style.SERIES_ORANGE, marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=2.0,
            label="Transfer  (A->B, B->A)", zorder=3)

    gap = selfer_ys[-1] - transfer_ys[-1]
    ax.annotate(f"task-specificity gap\n{gap:.1f} pp",
                xy=(4, (selfer_ys[-1] + transfer_ys[-1]) / 2),
                xytext=(3.0, (selfer_ys[-1] + transfer_ys[-1]) / 2),
                fontsize=8.5, color=style.STATUS_CRITICAL, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-", color=style.STATUS_CRITICAL, lw=1.2))

    ax.set_xticks(range(5), _WRN_BLOCK_LABELS)
    ax.set_xlabel("Blocks frozen from the source network  (n)")
    ax.set_ylabel("Top-1 validation accuracy (%)")
    ax.set_ylim(50, 78)
    ax.set_title("H3 - Frozen transfer degrades with feature specificity",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left")

    fig.tight_layout()
    return _save(fig, out_dir, "fig4_specificity_collapse")


def finetuning_recovery(selfer: pd.DataFrame, out_dir: Path) -> Path:
    """
    Paper Figure 5 (H2) -- fine-tuning recovers co-adaptation, with one exception.

    Fine-tuning the copied blocks removes the n = 3 frozen dip, as expected: the
    layers can re-coadapt. But the fine-tuned curve has a *deeper* dip of its own
    at n = 3 (TTTS, 68.0%), and that one is not co-adaptation.

    TTTS means Group2 is fine-tuned while Group3 -- 76% of the model's parameters
    -- is scratch and training at 10x the learning rate. The scratch block is
    chasing an input distribution that is itself still moving. The paper finds the
    same failure in FTTS and FFTS regardless of the frozen prefix, and in both
    selfer and transfer runs, which is what rules out a domain explanation.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    frozen_cfgs   = ["SSSS", "FSSS", "FFSS", "FFFS", "FFFF"]
    finetune_cfgs = ["SSSS", "TSSS", "TTSS", "TTTS", "TTTT"]
    xs = list(range(5))
    frozen_ys   = [selfer.loc[c, "clean_acc"] for c in frozen_cfgs]
    finetune_ys = [selfer.loc[c, "clean_acc"] for c in finetune_cfgs]

    ax.plot(xs, frozen_ys, color=style.SERIES_BLUE, marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=2.0,
            label="Frozen   F^n S^(4-n)", zorder=3)
    ax.plot(xs, finetune_ys, color=style.SERIES_ORANGE, marker="o", markersize=6,
            markeredgecolor=style.SURFACE, markeredgewidth=2.0,
            label="Fine-tuned  T^n S^(4-n)", zorder=3)

    baseline = selfer.loc["SSSS", "clean_acc"]
    ax.axhline(baseline, color=style.INK_MUTED, linewidth=1.0, zorder=1)

    dip = int(np.argmin(finetune_ys))
    ax.annotate(f"TTTS: {finetune_ys[dip]:.1f}%\nscratch Group3 above a\nfine-tuned Group2",
                xy=(dip, finetune_ys[dip]), xytext=(dip - 0.65, finetune_ys[dip] - 1.4),
                fontsize=8.5, color=style.STATUS_CRITICAL, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-", color=style.STATUS_CRITICAL, lw=1.2))

    ax.set_xticks(range(5), _WRN_BLOCK_LABELS)
    ax.set_xlabel("Blocks copied from the source network  (n)")
    ax.set_ylabel("Top-1 validation accuracy (%)")
    ax.set_title("H2 - Fine-tuning recovers co-adaptation, but exposes the G2->G3 instability",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left")

    fig.tight_layout()
    return _save(fig, out_dir, "fig5_finetuning_recovery")


def tinyimagenet_per_run_figures(out_dir: Path,
                                 runs_dir: Path | None = None) -> list[Path]:
    """
    Figures 2 and 6, which require per-run Experiment 1 artifacts.

    Neither is derivable from the reference tables, which report the mean over
    source/target directions. Returns an empty list until
    ``scripts/tinyimagenet_run.py`` has populated ``results/tinyimagenet/runs/``.
    """
    runs = load_tinyimagenet_runs(runs_dir)
    if runs is None:
        print(
            "  skipped fig2 (A/B symmetry) and fig6 (ID-OOD-C correlation): "
            "both require per-run TinyImageNet artifacts. Run "
            "scripts/tinyimagenet_run.py to generate them."
        )
        return []

    paths = []

    # ── Figure 2: A/B symmetry check ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (left, right, title) in zip(axes, [
        ("AtoB", "BtoA", "Transfer symmetry: A->B vs B->A"),
        ("AtoA", "BtoB", "Selfer symmetry: A->A vs B->B"),
    ]):
        merged = runs[runs["direction"] == left].merge(
            runs[runs["direction"] == right], on="config", suffixes=("_l", "_r")
        )
        if merged.empty:
            ax.set_visible(False)
            continue

        ax.scatter(merged["clean_acc_l"], merged["clean_acc_r"],
                   s=70, color=style.SERIES_BLUE,
                   edgecolors=style.SURFACE, linewidths=2.0, zorder=3)

        lo = min(merged["clean_acc_l"].min(), merged["clean_acc_r"].min()) - 2
        hi = max(merged["clean_acc_l"].max(), merged["clean_acc_r"].max()) + 2
        ax.plot([lo, hi], [lo, hi], color=style.BASELINE, linewidth=1.0, zorder=1)

        r = float(np.corrcoef(merged["clean_acc_l"], merged["clean_acc_r"])[0, 1])
        mae = float(np.abs(merged["clean_acc_l"] - merged["clean_acc_r"]).mean())
        ax.text(0.04, 0.94, f"Pearson r = {r:.3f}\nMAE = {mae:.2f} pp",
                transform=ax.transAxes, va="top", fontsize=8.5, color=style.INK_2)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{left} accuracy (%)")
        ax.set_ylabel(f"{right} accuracy (%)")
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.grid(True, axis="both")

    fig.suptitle("Split validity - points near the identity line mean A and B are equally hard",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths.append(_save(fig, out_dir, "fig2_split_symmetry"))

    # ── Figure 6: clean vs corrupted accuracy ────────────────────────────────
    usable = runs.dropna(subset=["clean_acc", "corrupt_acc"])
    if not usable.empty:
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        for is_selfer, label, color in [
            (True,  "Selfer",   style.SERIES_BLUE),
            (False, "Transfer", style.SERIES_ORANGE),
        ]:
            subset = usable[usable["is_selfer"] == is_selfer]
            ax.scatter(subset["clean_acc"], subset["corrupt_acc"], s=55,
                       color=color, edgecolors=style.SURFACE, linewidths=2.0,
                       label=label, zorder=3)

        r = float(np.corrcoef(usable["clean_acc"], usable["corrupt_acc"])[0, 1])
        rho = float(usable["clean_acc"].rank().corr(usable["corrupt_acc"].rank()))
        ax.text(0.04, 0.94,
                f"n = {len(usable)} runs\nPearson r = {r:.3f}\nSpearman rho = {rho:.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=style.INK_2)

        ax.set_xlabel("Clean top-1 accuracy (%)")
        ax.set_ylabel("Corrupted top-1 accuracy (TinyImageNet-C style, %)")
        ax.set_title("Corruption accuracy tracks clean accuracy",
                     loc="left", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, axis="both")
        fig.tight_layout()
        paths.append(_save(fig, out_dir, "fig6_clean_vs_corrupt"))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_all(
    results_dir: str | Path | None = None,
    prefer: str = "paper",
) -> list[Path]:
    """
    Regenerate every figure whose inputs are available.

    Args:
        results_dir: root of the results tree; defaults to ``<repo>/results``.
        prefer:      ``"paper"`` plots the transcribed tables (default, so the
                     figures agree with the write-up); ``"runs"`` plots the
                     archived run artifacts.

    Figures with missing inputs are skipped with a printed explanation rather
    than approximated. Returns the paths written.
    """
    style.apply_style()

    results_dir = Path(results_dir or RESULTS_DIR)
    iwildcam_out = results_dir / "iwildcam" / "figures"
    tiny_out     = results_dir / "tinyimagenet" / "figures"

    written: list[Path] = []

    print(f"iWildCam figures (source: {prefer}):")
    df = load_leaderboard(prefer=prefer, results_dir=results_dir)
    written.append(partition_heatmap(df, iwildcam_out))
    written.append(early_layer_generality(df, iwildcam_out))
    written.append(stage4_specificity(df, iwildcam_out))
    written.append(t_buffer_benefit(df, iwildcam_out))
    written.append(id_ood_alignment(df, iwildcam_out))
    curves = training_curves(iwildcam_out, results_dir / "iwildcam" / "runs")
    if curves is not None:
        written.append(curves)

    print("\nTinyImageNet figures (source: paper Tables 4 and 5):")
    selfer, transfer = load_tinyimagenet_tables(results_dir)
    written.append(yosinski_replication(selfer, transfer, tiny_out))
    written.append(coadaptation_fragility(selfer, tiny_out))
    written.append(specificity_collapse(selfer, transfer, tiny_out))
    written.append(finetuning_recovery(selfer, tiny_out))
    written.extend(
        tinyimagenet_per_run_figures(tiny_out, results_dir / "tinyimagenet" / "runs")
    )

    print(f"\n{len(written)} figures written.")
    return written
