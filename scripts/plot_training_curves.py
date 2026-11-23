"""Training-curve figures for the report (issue #12), in IEEE figure format.

IEEE conventions used:
- single-column width (3.5 in), serif (Times) text at 8 pt, no title inside
  the figure (the caption goes under it in the paper as "Fig. N.");
- vector PDF for the paper plus a 600-dpi PNG;
- every series also differs by marker shape / line style, so the figures still
  read when printed in grayscale.

Reads docs/results/*.json (written by scripts/train.py) and writes
docs/figures/fig_val_f1_epochs.{pdf,png} and fig_loss_curves.{pdf,png}.
Suggested captions are in docs/figures/README.md.

Usage:
    python scripts/plot_training_curves.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "docs" / "results"
OUT = REPO_ROOT / "docs" / "figures"

# Categorical slots 1-3 of the validated reference palette (all-pairs CVD-safe).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

COL_W = 3.5  # IEEE single column, inches

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,  # embed TrueType fonts (IEEE PDF eXpress requirement)
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def epoch_curve(info, key="eval_f1"):
    e = info["per_epoch_val"]
    return np.array([x["epoch"] for x in e]), np.array([x[key] for x in e])


def seed_stats(names, key="eval_f1"):
    curves = [epoch_curve(load(n), key) for n in names]
    x = curves[0][0]
    ys = np.stack([c[1] for c in curves])
    return x, ys.mean(0), ys.min(0), ys.max(0)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote docs/figures/{stem}.pdf/.png")


def fig_val_f1():
    """Validation entity-level F1 per epoch: cased vs uncased (3 seeds), plus cased 8 epochs."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    series = [
        (["full_baseline", "base_s43", "base_s44"], "Cased, 5 epochs (3 seeds)", BLUE, "o", "-"),
        (["uncased", "uncased_s43", "uncased_s44"], "Uncased, 5 epochs (3 seeds)", ORANGE, "s", "-"),
    ]
    for names, label, color, marker, ls in series:
        x, mean, lo, hi = seed_stats(names)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, ls, color=color, marker=marker, markersize=4, markeredgecolor="white",
                markeredgewidth=0.6, label=label)
    x8, y8 = epoch_curve(load("ep8"))
    ax.plot(x8, y8, "--", color=AQUA, marker="^", markersize=4, markeredgecolor="white",
            markeredgewidth=0.6, label="Cased, 8 epochs (1 seed)")

    # Final selected checkpoint: uncased, seed 42, epoch 2.
    xf, yf = epoch_curve(load("uncased"))
    i = int(np.argmax(yf))
    ax.plot(xf[i], yf[i], marker="*", markersize=8, color=INK, markeredgecolor="white",
            markeredgewidth=0.5, linestyle="none", zorder=5, label="Final model (seed 42, epoch 2)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation entity-level F1")
    ax.set_xticks(range(1, 9))
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0.93, 0.99)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderaxespad=0.2)
    save(fig, "fig_val_f1_epochs")


def fig_loss():
    """Training loss (every 50 steps) and validation loss (per epoch), log scale."""
    # One panel per model so training vs validation stays readable in grayscale.
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.6), sharex=True, sharey=True)
    for ax, (name, panel, color, marker) in zip(axes, (
        ("full_baseline", "(a) Cased", BLUE, "o"),
        ("uncased", "(b) Uncased", ORANGE, "s"),
    )):
        info = load(name)
        tl = [(r["epoch"], r["loss"]) for r in info["train_loss_log"]]
        ax.plot(*zip(*tl), "-", color=color, linewidth=0.9, label="Training (every 50 steps)")
        ve, vl = epoch_curve(info, "eval_loss")
        ax.plot(ve, vl, linestyle=(0, (4, 2)), color=INK, linewidth=1.0, marker=marker, markersize=4,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
                label="Validation (per epoch)")
        ax.set_yscale("log")
        ax.set_ylabel("Cross-entropy loss")
        ax.text(0.98, 0.93, panel, transform=ax.transAxes, ha="right", va="top", color=INK)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), frameon=False, handlelength=2.6,
                  borderaxespad=0.2)
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(0, 5.1)
    axes[-1].set_xticks(range(0, 6))
    fig.align_ylabels(axes)
    save(fig, "fig_loss_curves")


if __name__ == "__main__":
    fig_val_f1()
    fig_loss()
