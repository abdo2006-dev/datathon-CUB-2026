"""
plotting.py
-----------
Reusable figure functions for the injection molding causal analysis.
All functions save to the figures/ directory and return the figure object.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.utils import FIGURES_DIR, PASS_FAIL_THRESHOLD

# ── Style ──────────────────────────────────────────────────────────────────────
TEAL = "#2a9d8f"
RED  = "#e76f51"
BLUE = "#264653"
GOLD = "#e9c46a"
GREY = "#adb5bd"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def savefig(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved → {path}")
    return path


# ── EDA Figures ────────────────────────────────────────────────────────────────
def plot_scrap_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["scrap_rate_pct"], bins=60, color=TEAL, edgecolor="white", alpha=0.85)
    ax.axvline(PASS_FAIL_THRESHOLD, color=RED, lw=1.8, ls="--", label=f"Pass/fail threshold ({PASS_FAIL_THRESHOLD}%)")
    ax.axvline(df["scrap_rate_pct"].mean(), color=BLUE, lw=1.8, ls="-",
               label=f"Mean ({df['scrap_rate_pct'].mean():.2f}%)")
    pct_fail = (df["scrap_rate_pct"] > PASS_FAIL_THRESHOLD).mean() * 100
    ax.set_title(f"Scrap Rate Distribution  (n={len(df):,} intervals  |  {pct_fail:.0f}% failing)")
    ax.set_xlabel("scrap_rate_pct (%)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "scrap_distribution.png")
    return fig


def plot_defect_breakdown(df: pd.DataFrame) -> plt.Figure:
    counts = df["defect_type"].value_counts()
    mean_scrap = df.groupby("defect_type")["scrap_rate_pct"].mean().reindex(counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: count
    axes[0].barh(counts.index[::-1], counts.values[::-1], color=TEAL)
    axes[0].set_xlabel("Interval count")
    axes[0].set_title("Defect Type Frequency")

    # Right: mean scrap
    colors = [RED if v > df["scrap_rate_pct"].mean() else TEAL for v in mean_scrap.values[::-1]]
    axes[1].barh(mean_scrap.index[::-1], mean_scrap.values[::-1], color=colors)
    axes[1].axvline(df["scrap_rate_pct"].mean(), color=GREY, lw=1.4, ls="--", label="Overall mean")
    axes[1].set_xlabel("Mean scrap rate (%)")
    axes[1].set_title("Mean Scrap Rate by Defect Type")
    axes[1].legend(frameon=False)

    fig.suptitle("Defect Burden", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "defect_breakdown.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list) -> plt.Figure:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 8}, linewidths=0.3,
    )
    ax.set_title("Pairwise Pearson Correlations (process levers + target)")
    fig.tight_layout()
    savefig(fig, "correlation_heatmap.png")
    return fig


# ── Causal Analysis Figures ────────────────────────────────────────────────────
def plot_cooling_sign_reversal(df: pd.DataFrame) -> plt.Figure:
    """Scatter plots showing raw vs. mold-temp-adjusted association for cooling time."""
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Cooling Time and Scrap Rate: Sign Reversal Under Adjustment", fontsize=13)

    # (A) Raw
    ax = axes[0]
    ax.scatter(df["cooling_time_s"], df["scrap_rate_pct"],
               alpha=0.12, s=12, color=GREY, rasterized=True)
    rho, _ = stats.pearsonr(df["cooling_time_s"], df["scrap_rate_pct"])
    m, b = np.polyfit(df["cooling_time_s"], df["scrap_rate_pct"], 1)
    xs = np.linspace(df["cooling_time_s"].min(), df["cooling_time_s"].max(), 100)
    ax.plot(xs, m * xs + b, color=RED, lw=2)
    ax.set_title(f"(A)  Raw association\nρ = {rho:+.2f}  →  spurious positive")
    ax.set_xlabel("cooling_time_s (s)")
    ax.set_ylabel("scrap_rate_pct (%)")

    # (B) Partial residuals after regressing out mold_temperature_c
    from sklearn.linear_model import LinearRegression
    X = df[["mold_temperature_c"]].values
    y = df["scrap_rate_pct"].values
    c = df["cooling_time_s"].values

    reg_y = LinearRegression().fit(X, y)
    reg_c = LinearRegression().fit(X, c)
    resid_y = y - reg_y.predict(X)
    resid_c = c - reg_c.predict(X)

    ax = axes[1]
    ax.scatter(resid_c, resid_y, alpha=0.12, s=12, color=GREY, rasterized=True)
    rho2, _ = stats.pearsonr(resid_c, resid_y)
    m2, b2 = np.polyfit(resid_c, resid_y, 1)
    xs2 = np.linspace(resid_c.min(), resid_c.max(), 100)
    ax.plot(xs2, m2 * xs2 + b2, color=TEAL, lw=2)
    ax.set_title(f"(B)  Adjusted for mold temperature\nPartial ρ = {rho2:+.2f}  →  true negative effect")
    ax.set_xlabel("cooling_time_s (partial residual)")
    ax.set_ylabel("scrap_rate_pct (partial residual)")

    fig.tight_layout()
    savefig(fig, "cooling_sign_reversal.png")
    return fig


def plot_forest(effect_df: pd.DataFrame) -> plt.Figure:
    """
    Forest plot of standardized β with 95% CI.

    effect_df columns: variable, beta_std, ci_lo, ci_hi
    """
    df = effect_df.sort_values("beta_std").reset_index(drop=True)
    colors = [TEAL if b < 0 else RED for b in df["beta_std"]]

    fig, ax = plt.subplots(figsize=(8, len(df) * 0.45 + 1.5))
    y = np.arange(len(df))

    ax.scatter(df["beta_std"], y, color=colors, s=60, zorder=3)
    for i, row in df.iterrows():
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                color=colors[i], lw=1.6, alpha=0.7)

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["variable"])
    ax.set_xlabel("Standardized β on scrap_rate_pct (adjusted; z-scored features)")
    ax.set_title("Adjusted Causal Effect Estimates with 95% Bootstrap CIs")

    teal_patch = mpatches.Patch(color=TEAL, label="Lowers scrap (β < 0)")
    red_patch  = mpatches.Patch(color=RED,  label="Raises scrap (β > 0)")
    ax.legend(handles=[teal_patch, red_patch], frameon=False, loc="lower right")

    fig.tight_layout()
    savefig(fig, "forest_plot.png")
    return fig


# ── Intervention Figures ───────────────────────────────────────────────────────
def plot_intervention_impact(actions: dict, combined_delta: float) -> plt.Figure:
    """
    Horizontal bar chart of estimated scrap reductions per action.

    actions: {label: delta_pp}
    combined_delta: combined package delta (pp)
    """
    labels = list(actions.keys()) + ["Combined package\n(all five actions)"]
    deltas = list(actions.values()) + [combined_delta]
    colors = [TEAL] * len(actions) + [BLUE]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(labels) + 2))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, height=0.6)
    ax.axvline(combined_delta, color=GOLD, lw=1.5, ls=":", alpha=0.9,
               label=f"Package total ({combined_delta:.2f} p.p.)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated Δ mean scrap_rate_pct (percentage points)")
    ax.set_title("Estimated Scrap Reduction per Intervention\n(Chain-Propagated GBR Counterfactual Simulation)")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "intervention_impact.png")
    return fig


def plot_warpage_interaction(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of mean scrap in warpage subset by mold-temp × cooling-time bins."""
    warp = df[df["defect_type"] == "warpage"].copy()
    warp["mold_bin"] = pd.cut(
        warp["mold_temperature_c"],
        bins=[-np.inf, 65, 75, np.inf],
        labels=["Mold ≤ 65 °C", "Mold 65–75 °C", "Mold > 75 °C"],
    )
    warp["cool_bin"] = pd.cut(
        warp["cooling_time_s"],
        bins=[-np.inf, 12, 18, np.inf],
        labels=["Cooling ≤ 12 s", "Cooling 12–18 s", "Cooling > 18 s"],
    )
    pivot = warp.pivot_table(values="scrap_rate_pct", index="mold_bin", columns="cool_bin", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=3, vmax=9, ax=ax, linewidths=0.5,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Mean Scrap Rate (%) — Warpage Subset\nMold Temperature × Cooling Time Interaction")
    ax.set_xlabel("Cooling time bin")
    ax.set_ylabel("Mold temperature bin")
    fig.tight_layout()
    savefig(fig, "warpage_interaction.png")
    return fig
