"""
intervention_helpers.py
-----------------------
Counterfactual simulation for the injection molding causal analysis.

CHAIN PROPAGATION (delta-propagation method)
---------------------------------------------
For two interventions, the paper describes "chain-propagated GBR counterfactual
simulation" where a lever's effect passes through an intermediate mediator:

  1. DRYER DEWPOINT → MOISTURE → SCRAP (paper Section 3.2, eqn 6)
     Shift dewpoint → predict DELTA in resin_moisture_pct via sub-model → add
     delta to each row's observed moisture before re-evaluating the main GBR.
     "dewpoint is thus not treated as a direct cause of scrap" — dewpoint itself
     is NOT changed in the main GBR; only the propagated moisture delta is applied.

  2. MAINTENANCE → CALIBRATION DRIFT → SCRAP (paper Section 5.2)
     Cap maintenance → predict DELTA in calibration_drift_index → add delta to
     observed drift before re-evaluating the main GBR.

IMPORTANT: We apply the SUB-MODEL DELTA (predicted change from the intervention),
NOT the sub-model's absolute prediction. This correctly uses the sub-model to
quantify the lever's marginal effect on the mediator, while respecting the large
residual variance (R² = 0.09 for moisture) that the sub-model cannot explain.
"""

import numpy as np
import pandas as pd
from typing import Union


LEVER_RANGES = {
    "cooling_time_s":               (5,   40),
    "mold_temperature_c":           (40,  110),
    "barrel_temperature_c":         (180, 310),
    "injection_pressure_bar":       (600, 1800),
    "hold_pressure_bar":            (300, 1200),
    "screw_speed_rpm":              (20,  150),
    "dryer_dewpoint_c":             (-50, -20),
    "shot_size_g":                  (50,  2000),
    "clamp_force_kn":               (500, 4400),
    "maintenance_days_since_last":  (1,   60),
}

MOISTURE_FEATURES = [
    "ambient_humidity_pct", "dryer_dewpoint_c",
    "resin_batch_quality_index", "ambient_temperature_c",
]
DRIFT_FEATURES = [
    "maintenance_days_since_last", "ambient_humidity_pct", "ambient_temperature_c",
]


def _clip(value, lever):
    lo, hi = LEVER_RANGES.get(lever, (-np.inf, np.inf))
    return float(np.clip(value, lo, hi))


def _apply_moisture_chain(df, df_int, feature_cols, moisture_model, mask):
    """
    Compute the delta in resin_moisture_pct caused by the current dewpoint shift
    in df_int, and add it to the observed moisture values.

    Method: predict moisture with ORIGINAL dewpoints → predict with SHIFTED dewpoints
    → delta = shifted_prediction - original_prediction → add delta to observed moisture.

    This respects the R²=0.09 of the sub-model by only propagating the CHANGE,
    not replacing the measured moisture with the sub-model's (noisy) absolute prediction.
    """
    if "resin_moisture_pct" not in feature_cols:
        return
    m_feats = [f for f in MOISTURE_FEATURES if f in feature_cols]
    if not m_feats:
        return

    # Prediction with ORIGINAL dewpoint (from the original df, not df_int)
    X_orig = df[m_feats].fillna(df[m_feats].mean()).values
    # Prediction with SHIFTED dewpoint (from df_int which has been modified)
    X_int_m = df_int[m_feats].fillna(df[m_feats].mean()).values

    delta_moisture = moisture_model.predict(X_int_m) - moisture_model.predict(X_orig)

    # Only apply delta to rows that were actually intervened on
    m_idx = feature_cols.index("resin_moisture_pct")
    new_moisture = np.clip(df_int.iloc[:, m_idx].values + delta_moisture, 0, None)
    df_int.iloc[mask.values, m_idx] = new_moisture[mask.values]


def _apply_drift_chain(df, df_int, feature_cols, drift_model, mask):
    """
    Compute the delta in calibration_drift_index caused by the current maintenance
    shift in df_int, and add it to the observed drift values.
    """
    if "calibration_drift_index" not in feature_cols:
        return
    d_feats = [f for f in DRIFT_FEATURES if f in feature_cols]
    if not d_feats:
        return

    X_orig  = df[d_feats].fillna(df[d_feats].mean()).values
    X_int_d = df_int[d_feats].fillna(df[d_feats].mean()).values
    delta_drift = drift_model.predict(X_int_d) - drift_model.predict(X_orig)

    d_idx = feature_cols.index("calibration_drift_index")
    new_drift = np.clip(df_int.iloc[:, d_idx].values + delta_drift, 0, None)
    df_int.iloc[mask.values, d_idx] = new_drift[mask.values]


def counterfactual_shift(
    df: pd.DataFrame,
    gbr,
    feature_cols: list,
    lever: str,
    target_value: Union[float, None] = None,
    delta: Union[float, None] = None,
    cap_only: bool = False,
    condition_col: str = None,
    condition_threshold: float = None,
    condition_direction: str = "ge",
    moisture_model=None,
    drift_model=None,
) -> dict:
    """
    Estimate the PATE of shifting `lever` using the fitted GBR.

    For the dryer dewpoint intervention, pass moisture_model to chain-propagate
    the effect through resin_moisture_pct (delta method).

    For the maintenance intervention, pass drift_model to chain-propagate
    the effect through calibration_drift_index (delta method).
    """
    df_sim = df[feature_cols].copy().fillna(df[feature_cols].mean())
    y_obs  = gbr.predict(df_sim.values)

    # Row selection
    if condition_col and condition_col in df.columns:
        mask = (df[condition_col] >= condition_threshold
                if condition_direction == "ge"
                else df[condition_col] <= condition_threshold)
    else:
        mask = pd.Series(True, index=df.index)

    df_int = df_sim.copy()

    # Apply lever shift
    if lever in df_int.columns:
        if delta is not None:
            df_int.loc[mask, lever] = (
                df_sim.loc[mask, lever] + delta
            ).apply(lambda v: _clip(v, lever))
        elif target_value is not None:
            clipped = _clip(target_value, lever)
            if cap_only:
                cap_mask = mask & (df[lever] > clipped)
                df_int.loc[cap_mask, lever] = clipped
            else:
                df_int.loc[mask, lever] = clipped

    # Chain propagation (delta method)
    if moisture_model is not None:
        _apply_moisture_chain(df, df_int, feature_cols, moisture_model, mask)
    if drift_model is not None:
        _apply_drift_chain(df, df_int, feature_cols, drift_model, mask)

    y_int  = gbr.predict(df_int.values)
    deltas = y_int - y_obs
    pate   = float(deltas[mask].mean())

    return dict(
        lever=lever,
        pate=round(pate, 4),
        n_intervened=int(mask.sum()),
        pate_pct_relative=round(pate / df["scrap_rate_pct"].mean() * 100, 2),
        condition=(f"{condition_col} {condition_direction} {condition_threshold}"
                   if condition_col else "all rows"),
    )


def simulate_combined_package(
    df: pd.DataFrame,
    gbr,
    feature_cols: list,
    interventions: list,
    moisture_model=None,
    drift_model=None,
) -> dict:
    """
    Apply all five interventions simultaneously and compute the combined PATE.

    Chains are applied ONCE after all lever shifts accumulate.
    """
    df_int = df[feature_cols].copy().fillna(df[feature_cols].mean())
    y_obs  = gbr.predict(df_int.values)
    all_masks = pd.Series(False, index=df.index)

    for spec in interventions:
        lever    = spec["lever"]
        if lever not in df_int.columns:
            continue
        delta    = spec.get("delta")
        tgt      = spec.get("target_value")
        cap_only = spec.get("cap_only", False)
        cond_col = spec.get("condition_col")
        cond_thr = spec.get("condition_threshold")
        cond_dir = spec.get("condition_direction", "ge")

        if cond_col and cond_col in df.columns:
            mask = (df[cond_col] >= cond_thr if cond_dir == "ge"
                    else df[cond_col] <= cond_thr)
        else:
            mask = pd.Series(True, index=df.index)

        all_masks = all_masks | mask

        if delta is not None:
            df_int.loc[mask, lever] = (
                df_int.loc[mask, lever] + delta
            ).apply(lambda v: _clip(v, lever))
        elif tgt is not None:
            clipped = _clip(tgt, lever)
            if cap_only:
                cap_mask = mask & (df[lever] > clipped)
                df_int.loc[cap_mask, lever] = clipped
            else:
                df_int.loc[mask, lever] = clipped

    # Apply chain propagation once (using overall mask)
    if moisture_model is not None:
        _apply_moisture_chain(df, df_int, feature_cols, moisture_model, all_masks)
    if drift_model is not None:
        _apply_drift_chain(df, df_int, feature_cols, drift_model, all_masks)

    y_int = gbr.predict(df_int.values)
    pate  = float((y_int - y_obs).mean())
    base  = float(df["scrap_rate_pct"].mean())

    return dict(
        baseline_mean_scrap=round(base, 4),
        package_mean_scrap=round(base + pate, 4),
        absolute_delta_pp=round(pate, 4),
        relative_delta_pct=round(pate / base * 100, 2),
    )
