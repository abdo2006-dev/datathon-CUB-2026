"""
utils.py
--------
Data loading, preprocessing, and variable-role definitions for the injection
molding causal analysis.

Variable roles match Section 2.1 of the paper exactly.
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(REPO_ROOT, "data", "synthetic_injection_molding_demo.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Variable roles (paper Section 2.1) ────────────────────────────────────────

# Real-time setpoints adjustable by the operator (paper §2.1, "Controllable process levers")
PROCESS_LEVERS = [
    "dryer_dewpoint_c", "barrel_temperature_c", "mold_temperature_c",
    "injection_pressure_bar", "hold_pressure_bar", "cooling_time_s",
    "screw_speed_rpm", "shot_size_g",
]

# Scheduling-policy decision, not a per-interval setpoint (paper §2.1)
PLANNING_LEVER = ["maintenance_days_since_last"]

# All intervention candidates
ALL_LEVERS = PROCESS_LEVERS + PLANNING_LEVER

# Variables with causal paths to both process levers and outcome (paper §2.1)
CONFOUNDERS = [
    "ambient_humidity_pct", "ambient_temperature_c", "operator_shift",
]

# Transmit effects of upstream causes toward the outcome;
# must NOT enter adjustment sets for total-effect estimation (paper §2.1)
MEDIATORS = [
    "resin_moisture_pct", "calibration_drift_index", "tool_wear_index",
]

# Adjusted for in all models but not an intervention target (paper §2.1, §4.5)
OPERATOR_COVARIATE = ["operator_experience_level"]

# Condition-on variables that define context; no intervention recommended (paper §2.1)
# Note: clamp_force_kn is NOT listed as a controllable lever in the paper.
CONTEXT = [
    "product_variant", "cavity_count", "part_weight_g", "clamp_force_kn",
]

# Batch-level quality index — used in some adjustment sets but not a lever.
# Paper Table 2 abbreviates it as "batch" in adjustment sets for injection_pressure
# and barrel_temperature.
BATCH_QUALITY = ["resin_batch_quality_index"]

IDENTIFIERS = [
    "timestamp", "plant_id", "machine_id", "mold_id",
    "product_variant", "resin_lot_id",
]

OUTCOMES = [
    "scrap_rate_pct", "scrap_count", "defect_type",
    "pass_fail_flag", "parts_produced", "energy_kwh_interval",
    "cycle_time_s",   # mechanically subsumes cooling_time_s → excluded from predictors
]

# Fixed effects absorbed in every regression (paper §2.3, eqn 4)
FIXED_EFFECTS = ["machine_id", "mold_id", "product_variant", "operator_shift"]

PASS_FAIL_THRESHOLD = 3.2   # % scrap rate (paper §1)

# ── DAG-informed adjustment sets (paper Table 2) ───────────────────────────────
# For each lever T, Z is the set that (i) blocks all backdoor paths from T to Y
# and (ii) contains no descendant of T (backdoor criterion, paper §2.3).
# FE = machine/mold/variant/shift fixed effects (always added on top in models).
ADJUSTMENT_SETS = {
    "cooling_time_s": [
        "mold_temperature_c", "part_weight_g", "shot_size_g",
        "ambient_humidity_pct", "ambient_temperature_c", "maintenance_days_since_last",
    ],
    "mold_temperature_c": [
        "cooling_time_s", "barrel_temperature_c", "part_weight_g",
        "ambient_humidity_pct", "ambient_temperature_c",
    ],
    "barrel_temperature_c": [
        "injection_pressure_bar", "mold_temperature_c",
        "resin_batch_quality_index",
        "ambient_humidity_pct", "ambient_temperature_c",
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


# ── Loading ────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load with basic preprocessing: clip sensor-noise clamp values, set pass/fail."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["clamp_force_kn"] = df["clamp_force_kn"].clip(upper=4400)
    df["pass_fail_flag"] = (df["scrap_rate_pct"] > PASS_FAIL_THRESHOLD).astype(int)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────
def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def zscore_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = zscore(df[c])
    return df
