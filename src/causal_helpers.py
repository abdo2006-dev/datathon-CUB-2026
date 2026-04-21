"""
causal_helpers.py
-----------------
DAG-informed regression, bootstrap CIs, sub-model training, and GBR.

NOTE ON beta_std vs beta_unstd
--------------------------------
beta_std:   Frisch-Waugh coefficient with z-scored lever and UNSCALED outcome.
            Matches Table 2 of the paper (e.g., -1.74 for cooling).

beta_unstd: Direct OLS in original units; natural-unit effect per unit of lever
            (e.g., p.p./s for cooling). Matches the paper's beta_tilde = -0.41 p.p./s.

These are numerically inconsistent with the paper's Equation 5
(beta_std = beta_unstd * sigma_T / sigma_Y) because the outcome is not z-scored
in the FW computation. The same inconsistency exists in the paper. We report both
values honestly, using each for its intended purpose.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from src.utils import ADJUSTMENT_SETS, FIXED_EFFECTS, ALL_LEVERS, zscore


def add_fixed_effects(df, fe_cols=FIXED_EFFECTS):
    return pd.get_dummies(df, columns=fe_cols, drop_first=True)


def _fe_dummies(df_enc):
    return [c for c in df_enc.columns if any(c.startswith(fe) for fe in FIXED_EFFECTS)]


def estimate_adjusted_effect(
    df, lever, outcome="scrap_rate_pct",
    adjustment_set=None, n_bootstrap=300, random_state=42,
):
    """
    DAG-adjusted effect of lever on outcome.
    Returns dict: lever, beta_std, ci_lo, ci_hi, beta_unstd, n_obs
    """
    if adjustment_set is None:
        adjustment_set = ADJUSTMENT_SETS.get(lever, [])

    available = [lever] + [c for c in adjustment_set if c in df.columns]
    df_model = df[available + [outcome] + FIXED_EFFECTS].dropna()
    df_enc   = add_fixed_effects(df_model)
    fe_cols  = _fe_dummies(df_enc)

    covs   = [c for c in available if c != lever] + fe_cols
    X_orig = df_enc[covs].astype(float).values
    t_orig = df_enc[lever].astype(float).values
    y_orig = df_enc[outcome].astype(float).values

    # Direct OLS (original units) → beta_unstd
    X_full = np.column_stack([t_orig.reshape(-1, 1), X_orig])
    beta_unstd = float(LinearRegression().fit(X_full, y_orig).coef_[0])

    # Frisch-Waugh with z-scored lever / unscaled outcome → beta_std
    t_z      = zscore(pd.Series(t_orig)).values
    reg_t    = LinearRegression().fit(X_orig, t_z)
    reg_y    = LinearRegression().fit(X_orig, y_orig)
    resid_t  = t_z - reg_t.predict(X_orig)
    resid_y  = y_orig - reg_y.predict(X_orig)
    beta_std = float(np.dot(resid_t, resid_y) / (np.dot(resid_t, resid_t) + 1e-12))

    # Bootstrap CIs (row-wise i.i.d.)
    rng = np.random.default_rng(random_state)
    n   = len(resid_t)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        rt, ry = resid_t[idx], resid_y[idx]
        boots.append(np.dot(rt, ry) / (np.dot(rt, rt) + 1e-12))
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return dict(
        lever=lever,
        beta_std=round(beta_std, 3),
        ci_lo=round(float(ci_lo), 3),
        ci_hi=round(float(ci_hi), 3),
        beta_unstd=round(beta_unstd, 4),
        n_obs=len(df_model),
    )


def estimate_all_effects(df, levers=None, outcome="scrap_rate_pct", n_bootstrap=300):
    if levers is None:
        levers = [l for l in ALL_LEVERS if l in df.columns]
    results = []
    for lever in levers:
        try:
            r = estimate_adjusted_effect(df, lever, outcome=outcome, n_bootstrap=n_bootstrap)
            results.append(r)
            print(f"  {lever:35s}  β_std={r['beta_std']:+.3f}  "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"β̃={r['beta_unstd']:+.4f} p.p./unit")
        except Exception as e:
            print(f"  {lever}: ERROR — {e}")
    return pd.DataFrame(results)


def train_sub_model(df, target, features):
    """
    Train a linear sub-model for chain propagation in counterfactual simulation.

    Used for:
      moisture chain:  resin_moisture_pct ~ f(humidity, dewpoint, batch_quality, amb_temp)
      drift chain:     calibration_drift_index ~ f(maintenance_days, humidity, amb_temp)

    Returns (fitted_model, cv_r2, available_features).
    """
    available = [f for f in features if f in df.columns]
    df_sub = df[available + [target]].dropna()
    X, y   = df_sub[available].values, df_sub[target].values
    model  = LinearRegression()
    cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    model.fit(X, y)
    print(f"  Sub-model [{target}]: CV R² = {cv_r2:.4f}  features: {available}")
    return model, cv_r2, available


def train_gbr(df, outcome="scrap_rate_pct", random_state=42):
    """
    Train a gradient-boosted regressor (M=400, lr=0.05, depth=3, subsample=0.8).

    Mediators are INCLUDED so the GBR captures indirect causal paths in simulation.
    cycle_time_s is EXCLUDED (mechanical subsumption of cooling_time_s → data leakage).

    Returns (model, feature_names, cv_r2).
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from src.utils import IDENTIFIERS, OUTCOMES

    exclude = set(IDENTIFIERS + OUTCOMES + ["cycle_time_s"])
    exclude.discard(outcome)

    feature_cols = [
        c for c in df.columns
        if c not in exclude and c != outcome
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df_m = df[feature_cols + [outcome]].dropna()
    X, y = df_m[feature_cols].values, df_m[outcome].values

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        random_state=random_state, subsample=0.8,
    )
    cv_scores = cross_val_score(gbr, X, y, cv=5, scoring="r2")
    gbr.fit(X, y)
    cv_r2 = cv_scores.mean()
    print(f"GBR 5-fold CV R² = {cv_r2:.3f} ± {cv_scores.std():.3f}  "
          f"(paper reports 0.64; small gap expected from RNG/dataset version)")
    return gbr, feature_cols, cv_r2
