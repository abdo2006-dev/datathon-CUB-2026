"""
intervention_helpers.py
------------------------
Counterfactual simulation logic for the injection molding causal analysis.

The core method (do-calculus simulation via GBR):
  For each row i, shift lever T to target t*, hold all other features at x_i,
  re-evaluate the GBR. PATE = mean over all rows of (ŷ_intervened - ŷ_observed).
"""

import numpy as np
import pandas as pd
from typing import Union


# ── Ontology-declared physical ranges ─────────────────────────────────────────
LEVER_RANGES = {
    "cooling_time_s":          (5,   40),
    "mold_temperature_c":      (40,  110),
    "barrel_temperature_c":    (180, 310),
    "injection_pressure_bar":  (600, 1800),
    "hold_pressure_bar":       (300, 1200),
    "screw_speed_rpm":         (20,  150),
    "dryer_dewpoint_c":        (-50, -20),
    "shot_size_g":             (50,  2000),
    "maintenance_days_since_last": (1, 60),
}


def _clip_to_range(value: float, lever: str) -> float:
    lo, hi = LEVER_RANGES.get(lever, (-np.inf, np.inf))
    return float(np.clip(value, lo, hi))


# ── Core simulation function ───────────────────────────────────────────────────
def counterfactual_shift(
    df: pd.DataFrame,
    gbr,
    feature_cols: list,
    lever: str,
    target_value: Union[float, None] = None,
    delta: Union[float, None] = None,
    condition_col: str = None,
    condition_threshold: float = None,
    condition_direction: str = "ge",   # "ge" or "le"
    cap_only: bool = False,            # if True, only intervene on rows where lever > target_value
) -> dict:
    """
    Estimate the average treatment effect of shifting `lever` to `target_value`
    (or by `delta` from each row's observed value) using the fitted GBR.

    Optionally apply the intervention only to rows where `condition_col` meets
    the threshold condition (for conditional recommendations like wear-aware pressure).

    Returns
    -------
    dict with keys: lever, target_value_or_delta, n_intervened, pate, pate_percent
    """
    df_sim = df[feature_cols].copy().fillna(df[feature_cols].mean())
    y_obs = gbr.predict(df_sim.values)

    # Identify rows to intervene on
    if condition_col is not None and condition_col in df.columns:
        if condition_direction == "ge":
            mask = df[condition_col] >= condition_threshold
        else:
            mask = df[condition_col] <= condition_threshold
    else:
        mask = pd.Series(True, index=df.index)

    df_int = df_sim.copy()
    if lever in df_int.columns:
        if delta is not None:
            df_int.loc[mask, lever] = (df_sim.loc[mask, lever] + delta).apply(
                lambda v: _clip_to_range(v, lever)
            )
        elif target_value is not None:
            clipped = _clip_to_range(target_value, lever)
            # For cap interventions: only shift rows that exceed the cap value
            # For floor interventions: only shift rows that are below the floor value
            if cap_only:
                cap_mask = mask & (df[lever] > clipped)
                df_int.loc[cap_mask, lever] = clipped
            else:
                df_int.loc[mask, lever] = clipped

    y_int = gbr.predict(df_int.values)
    delta_per_row = y_int - y_obs
    pate = delta_per_row[mask].mean()

    return {
        "lever": lever,
        "specification": f"delta={delta}" if delta is not None else f"target={target_value}",
        "condition": f"{condition_col} {condition_direction} {condition_threshold}" if condition_col else "all rows",
        "n_intervened": int(mask.sum()),
        "pate": round(pate, 4),
        "pate_pct_relative": round(pate / df["scrap_rate_pct"].mean() * 100, 2),
    }


# ── Moisture sub-model propagation ────────────────────────────────────────────
def propagate_dewpoint_to_moisture(
    df: pd.DataFrame,
    gbr_moisture,
    moisture_feature_cols: list,
    dewpoint_delta: float,
    humidity_threshold: float = 65.0,
) -> pd.Series:
    """
    Simulate the effect of shifting dryer_dewpoint_c by `dewpoint_delta` on
    resin_moisture_pct via the structural moisture sub-model.

    Only applied to rows where ambient_humidity_pct >= humidity_threshold.
    Returns a Series of intervened resin_moisture_pct values.
    """
    mask = df["ambient_humidity_pct"] >= humidity_threshold
    df_sim = df[moisture_feature_cols].copy().fillna(df[moisture_feature_cols].mean())

    if "dryer_dewpoint_c" in df_sim.columns:
        df_sim.loc[mask, "dryer_dewpoint_c"] = (
            df_sim.loc[mask, "dryer_dewpoint_c"] + dewpoint_delta
        ).clip(lower=-50, upper=-20)

    return pd.Series(gbr_moisture.predict(df_sim.values), index=df.index)


# ── Combined package simulation ────────────────────────────────────────────────
def simulate_combined_package(
    df: pd.DataFrame,
    gbr,
    feature_cols: list,
    interventions: list,
) -> dict:
    """
    Apply all interventions simultaneously (sequentially modifying the same row)
    and compute the combined PATE.

    interventions: list of dicts with keys matching counterfactual_shift kwargs,
                   minus df/gbr/feature_cols.

    Returns summary dict.
    """
    df_int = df[feature_cols].copy().fillna(df[feature_cols].mean())
    y_obs = gbr.predict(df_int.values)

    for spec in interventions:
        lever = spec["lever"]
        if lever not in df_int.columns:
            continue

        # Build mask
        cond_col = spec.get("condition_col")
        cond_thr = spec.get("condition_threshold")
        cond_dir = spec.get("condition_direction", "ge")
        if cond_col and cond_col in df.columns:
            mask = (df[cond_col] >= cond_thr) if cond_dir == "ge" else (df[cond_col] <= cond_thr)
        else:
            mask = pd.Series(True, index=df.index)

        delta     = spec.get("delta")
        target_v  = spec.get("target_value")

        cap_only = spec.get("cap_only", False)
        if delta is not None:
            df_int.loc[mask, lever] = (df_int.loc[mask, lever] + delta).apply(
                lambda v: _clip_to_range(v, lever)
            )
        elif target_v is not None:
            clipped = _clip_to_range(target_v, lever)
            if cap_only:
                cap_mask = mask & (df[lever] > clipped)
                df_int.loc[cap_mask, lever] = clipped
            else:
                df_int.loc[mask, lever] = clipped

    y_int = gbr.predict(df_int.values)
    pate = (y_int - y_obs).mean()
    baseline = df["scrap_rate_pct"].mean()

    return {
        "baseline_mean_scrap": round(baseline, 4),
        "package_mean_scrap": round(baseline + pate, 4),
        "absolute_delta_pp": round(pate, 4),
        "relative_delta_pct": round(pate / baseline * 100, 2),
    }
