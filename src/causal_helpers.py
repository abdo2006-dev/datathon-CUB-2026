"""
causal_helpers.py
-----------------
DAG-informed regression, bootstrap confidence intervals, and effect estimation
for the injection molding causal analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils import (
    ADJUSTMENT_SETS, FIXED_EFFECTS, ALL_LEVERS, zscore
)


# ── Fixed-effect encoding ──────────────────────────────────────────────────────
def add_fixed_effects(df: pd.DataFrame, fe_cols: list = FIXED_EFFECTS) -> pd.DataFrame:
    """One-hot encode fixed-effect columns, drop first to avoid collinearity."""
    return pd.get_dummies(df, columns=fe_cols, drop_first=True)


# ── Adjusted regression for a single lever ────────────────────────────────────
def estimate_adjusted_effect(
    df: pd.DataFrame,
    lever: str,
    outcome: str = "scrap_rate_pct",
    adjustment_set: list = None,
    n_bootstrap: int = 300,
    random_state: int = 42,
) -> dict:
    """
    Estimate the DAG-adjusted causal effect of `lever` on `outcome`.

    Returns
    -------
    dict with keys: lever, beta_std, ci_lo, ci_hi, beta_unstd, sigma_lever, sigma_outcome
    """
    if adjustment_set is None:
        adjustment_set = ADJUSTMENT_SETS.get(lever, [])

    # Build feature matrix
    covariates = [lever] + adjustment_set
    available = [c for c in covariates if c in df.columns]
    df_model = df[available + [outcome] + FIXED_EFFECTS].dropna()

    # One-hot fixed effects
    df_enc = add_fixed_effects(df_model)
    fe_dummies = [c for c in df_enc.columns if any(c.startswith(fe) for fe in FIXED_EFFECTS)]

    # Z-score continuous inputs
    sigma_lever   = df_model[lever].std()
    sigma_outcome = df_model[outcome].std()

    df_z = df_enc.copy()
    for col in available:
        if pd.api.types.is_numeric_dtype(df_z[col]):
            df_z[col] = zscore(df_z[col])

    X_cols = [c for c in available if c != lever] + fe_dummies
    X = df_z[X_cols].values.astype(float)
    t = df_z[lever].values
    y = df_z[outcome].values

    # Frisch-Waugh: partial out covariates from both lever and outcome
    reg_t = LinearRegression().fit(X, t)
    reg_y = LinearRegression().fit(X, y)
    resid_t = t - reg_t.predict(X)
    resid_y = y - reg_y.predict(X)

    beta_std = np.dot(resid_t, resid_y) / np.dot(resid_t, resid_t)
    beta_unstd = beta_std * (sigma_outcome / sigma_lever)

    # Bootstrap CIs
    rng = np.random.default_rng(random_state)
    n = len(df_z)
    betas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rt = resid_t[idx]
        ry = resid_y[idx]
        b = np.dot(rt, ry) / (np.dot(rt, rt) + 1e-12)
        betas.append(b)
    betas = np.array(betas)
    ci_lo, ci_hi = np.percentile(betas, [2.5, 97.5])

    return {
        "lever": lever,
        "beta_std": round(beta_std, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "beta_unstd": round(beta_unstd, 4),
        "sigma_lever": round(sigma_lever, 4),
        "sigma_outcome": round(sigma_outcome, 4),
        "n_obs": len(df_z),
    }


def estimate_all_effects(
    df: pd.DataFrame,
    levers: list = None,
    outcome: str = "scrap_rate_pct",
    n_bootstrap: int = 300,
) -> pd.DataFrame:
    """
    Run adjusted regression for every lever in `levers` and return a summary DataFrame.
    """
    if levers is None:
        levers = [l for l in ALL_LEVERS if l in df.columns]

    results = []
    for lever in levers:
        try:
            r = estimate_adjusted_effect(df, lever, outcome=outcome, n_bootstrap=n_bootstrap)
            results.append(r)
            print(f"  {lever:35s}  β_std={r['beta_std']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        except Exception as e:
            print(f"  {lever}: ERROR — {e}")

    return pd.DataFrame(results)


# ── Natural-unit effect sizing ─────────────────────────────────────────────────
def natural_unit_effect(beta_std: float, sigma_lever: float, sigma_outcome: float,
                         lever_step: float) -> float:
    """
    Convert standardized β to expected outcome change for a given lever step.

    ΔY = beta_std * (sigma_outcome / sigma_lever) * lever_step
    """
    beta_unstd = beta_std * (sigma_outcome / sigma_lever)
    return beta_unstd * lever_step


# ── GBR training ──────────────────────────────────────────────────────────────
def train_gbr(df: pd.DataFrame, outcome: str = "scrap_rate_pct", random_state: int = 42):
    """
    Train a gradient-boosted regressor on the full feature set (excluding outcome columns
    and identifiers). Returns (model, feature_names, cv_r2).
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    # Exclude identifiers, outcomes (except the target), and cycle_time_s (leakage)
    from src.utils import IDENTIFIERS, OUTCOMES
    exclude = set(IDENTIFIERS + OUTCOMES + ["cycle_time_s"])
    exclude.discard(outcome)

    feature_cols = [c for c in df.columns if c not in exclude and c != outcome
                    and pd.api.types.is_numeric_dtype(df[c])]
    df_model = df[feature_cols + [outcome]].dropna()
    X = df_model[feature_cols].values
    y = df_model[outcome].values

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        random_state=random_state, subsample=0.8
    )
    cv_scores = cross_val_score(gbr, X, y, cv=5, scoring="r2")
    gbr.fit(X, y)
    print(f"GBR 5-fold CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return gbr, feature_cols, cv_scores.mean()
