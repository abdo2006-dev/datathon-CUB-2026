# Data

## Dataset

**File:** `synthetic_injection_molding_demo.csv`  
**Rows:** 5,000 | **Columns:** 33  
**Granularity:** One 30-minute production interval per (machine × mold × product variant × resin lot)  
**Time span:** 2026-01-05 to 2026-03-20 (75 days)  
**Plants:** VN_QUANGNAM, DE_WINNENDEN, RO_CURTEA, DE_OBERSONT  
**Machines:** 12  

This is a synthetic but realistic injection molding dataset designed for causal analysis. It contains realistic confounders, mediators, interaction effects, and machine-level heterogeneity.

---

## Variable Glossary

### Identifiers
| Column | Description |
|---|---|
| `timestamp` | Interval end time (UTC) |
| `plant_id` | Plant identifier (4 plants) |
| `machine_id` | Machine identifier (12 machines) |
| `mold_id` | Mold identifier |
| `resin_lot_id` | Resin batch identifier |

### Context / Operating Conditions
| Column | Units | Description |
|---|---|---|
| `product_variant` | — | Part type being produced |
| `operator_shift` | — | Shift assignment (A_Morning, B_Afternoon, C_Night) |
| `operator_experience_level` | 1–5 | Operator experience rating |
| `cavity_count` | count | Number of mold cavities |
| `part_weight_g` | g | Target part weight |

### Environmental Confounders
| Column | Units | Description |
|---|---|---|
| `ambient_temperature_c` | °C | Ambient temperature at machine |
| `ambient_humidity_pct` | % RH | Relative humidity at machine |

### Process Levers (Controllable Setpoints)
| Column | Units | Description |
|---|---|---|
| `dryer_dewpoint_c` | °C | Dryer dewpoint setting |
| `barrel_temperature_c` | °C | Barrel/melt temperature |
| `mold_temperature_c` | °C | Mold surface temperature |
| `injection_pressure_bar` | bar | Peak injection pressure |
| `hold_pressure_bar` | bar | Hold/pack pressure |
| `screw_speed_rpm` | rpm | Screw rotation speed |
| `cooling_time_s` | s | Cooling phase duration |
| `shot_size_g` | g | Material shot weight |
| `clamp_force_kn` | kN | Mold clamp force |

### Planning Lever
| Column | Units | Description |
|---|---|---|
| `maintenance_days_since_last` | days | Days since last scheduled maintenance |

### Mediators
| Column | Description |
|---|---|
| `resin_moisture_pct` | Resin moisture content at injection (mediates humidity → splay pathway) |
| `calibration_drift_index` | Sensor/actuator calibration deviation (mediates maintenance → instability pathway) |
| `tool_wear_index` | Tooling wear state (mediates maintenance → flash/dimensional pathway) |
| `resin_batch_quality_index` | Incoming resin quality score |

### Outcome Variables
| Column | Description |
|---|---|
| `scrap_rate_pct` | **Primary KPI.** Scrap rate as a percentage of parts produced |
| `scrap_count` | Raw count of scrapped parts in interval |
| `defect_type` | Dominant defect mode in interval |
| `pass_fail_flag` | 1 = scrap_rate_pct > 3.2% threshold |
| `parts_produced` | Total parts produced in interval |
| `energy_kwh_interval` | Energy consumption for interval |
| `cycle_time_s` | Total cycle time (mechanically subsumes cooling_time_s) |

---

## Key Statistics

| Metric | Value |
|---|---|
| Mean scrap rate | 4.44% |
| Median scrap rate | 4.24% |
| Std dev scrap rate | 1.52 p.p. |
| % intervals failing 3.2% threshold | 78% |
| Dominant defect | Warpage (33%) |
| Missing values | 0 |

---

## Important Notes

1. **Do not use `cycle_time_s` as a predictor.** It mechanically subsumes `cooling_time_s` and will cause data leakage in any model estimating cooling-time effects.
2. **Mediators should not enter adjustment sets** when estimating total effects of upstream levers. Conditioning on `resin_moisture_pct`, `calibration_drift_index`, or `tool_wear_index` blocks causal pathways and produces direct (not total) effect estimates.
3. **`scrap_rate_pct` is the authoritative KPI.** It correlates at ρ = 0.80 with the independently computed `scrap_count / parts_produced`; max |Δ| = 7.4 p.p.
4. Thirteen `clamp_force_kn` values marginally above the 4,400 kN ceiling were clipped (sensor noise).
