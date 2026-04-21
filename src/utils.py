"""
utils.py
--------
Data loading and preprocessing utilities for the injection molding causal analysis.
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "synthetic_injection_molding_demo.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Variable role definitions ──────────────────────────────────────────────────
IDENTIFIERS = [
    "timestamp", "plant_id", "machine_id", "mold_id",
    "product_variant", "resin_lot_id",
]

PROCESS_LEVERS = [
    "dryer_dewpoint_c", "barrel_temperature_c", "mold_temperature_c",
    "injection_pressure_bar", "hold_pressure_bar", "screw_speed_rpm",
    "cooling_time_s", "shot_size_g", "clamp_force_kn",
]

PLANNING_LEVER = ["maintenance_days_since_last"]

CONFOUNDERS = [
    "ambient_humidity_pct", "ambient_temperature_c", "operator_shift",
    "operator_experience_level",
]

MEDIATORS = [
    "resin_moisture_pct", "calibration_drift_index", "tool_wear_index",
    "resin_batch_quality_index",
]

CONTEXT = ["cavity_count", "part_weight_g"]

OUTCOMES = [
    "scrap_rate_pct", "scrap_count", "defect_type",
    "pass_fail_flag", "parts_produced", "energy_kwh_interval", "cycle_time_s",
]

ALL_LEVERS = PROCESS_LEVERS + PLANNING_LEVER

PASS_FAIL_THRESHOLD = 3.2  # % scrap rate threshold


# ── DAG-informed adjustment sets ───────────────────────────────────────────────
# For each lever T, these are the covariates Z that satisfy the backdoor criterion
# (blocking all non-causal paths from T to scrap_rate_pct, no descendants of T).
# Machine/mold/variant/shift fixed effects are added in every model on top of these.
ADJUSTMENT_SETS = {
    "cooling_time_s": [
        "mold_temperature_c", "part_weight_g", "shot_size_g",
        "ambient_humidity_pct", "ambient_temperature_c", "maintenance_days_since_last",
    ],
    "mold_temperature_c": [
        "cooling_time_s", "barrel_temperature_c", "part_weight_g", "ambient_humidity_pct",
        "ambient_temperature_c",
    ],
    "barrel_temperature_c": [
        "injection_pressure_bar", "mold_temperature_c",
        "resin_batch_quality_index", "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "injection_pressure_bar": [
        "tool_wear_index", "clamp_force_kn", "barrel_temperature_c",
        "resin_batch_quality_index", "part_weight_g", "hold_pressure_bar",
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "hold_pressure_bar": [
        "injection_pressure_bar", "tool_wear_index",
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "dryer_dewpoint_c": [
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "maintenance_days_since_last": [
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "screw_speed_rpm": [
        "shot_size_g", "barrel_temperature_c",
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
}

FIXED_EFFECTS = ["machine_id", "mold_id", "product_variant", "operator_shift"]


# ── Loading ────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the injection molding dataset with basic preprocessing applied."""
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Clip the small number of out-of-range clamp force readings
    df["clamp_force_kn"] = df["clamp_force_kn"].clip(upper=4400)

    # Encode pass/fail threshold consistently
    df["pass_fail_flag"] = (df["scrap_rate_pct"] > PASS_FAIL_THRESHOLD).astype(int)

    return df


# ── Feature engineering ────────────────────────────────────────────────────────
def get_dummies_fixed_effects(df: pd.DataFrame, fe_cols: list = FIXED_EFFECTS) -> pd.DataFrame:
    """One-hot encode fixed-effect columns (drop_first to avoid multicollinearity)."""
    return pd.get_dummies(df, columns=fe_cols, drop_first=True)


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a Series to mean=0, std=1."""
    return (series - series.mean()) / series.std()


def zscore_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a copy of df with the listed columns z-scored."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = zscore(df[c])
    return df


def get_continuous_levers(df: pd.DataFrame) -> list:
    """Return the list of numeric process/planning levers present in df."""
    return [c for c in ALL_LEVERS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
