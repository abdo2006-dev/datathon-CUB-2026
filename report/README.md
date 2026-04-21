# Report

## Full Paper

**"Reducing Scrap in Injection Molding: A DAG-Informed Causal Decision Analysis"**  
Constructor University Datathon 2026 · Insytes Manufacturing Challenge  
Team: Limit Breakers

The full paper (`Injection_Molding_Paper_v2.pdf`) contains:
- Abstract and introduction
- Causal setup and DAG (Section 2)
- Estimation methods with equations (Section 3)
- Full results including all tables and figures (Section 4)
- Recommendations and trade-offs (Section 5)
- Limitations (Section 6)
- Conclusions and references (Section 7)

## Supporting Materials

| File | Contents |
|---|---|
| `Injection_Molding_DAG_Notes.pdf` | Narrative interpretation of the causal graph and its five layers |
| `Datathon_Student_Guide_Injection_Molding.pdf` | Dataset description, ontology guide, and suggested analytical workflow |

## Key Tables from the Paper

### Table 2 — DAG-Adjusted Effect Estimates

| Lever | β (std) | 95% CI | Per op. step |
|---|---|---|---|
| `cooling_time_s` | **−1.75** | [−1.85, −1.65] | −0.61 p.p. / +1.5 s |
| `mold_temperature_c` | +0.88 | [+0.80, +0.96] | +0.40 p.p. / +5 °C |
| `injection_pressure_bar` | +0.31 | [+0.22, +0.41] | +0.11 p.p. / +50 bar |
| `dryer_dewpoint_c` | +0.09 | [+0.05, +0.13] | +0.15 p.p. / +5 °C |
| `maintenance_days_since_last` | +0.10 | [+0.07, +0.14] | +0.07 p.p. / +7 days |

### Table 6 — Combined Package Impact

| Metric | Baseline | With package |
|---|---|---|
| Mean scrap rate | 4.44% | ≈ 3.85% (−13% relative) |
| Cycle time | 53.7 s | ≈ 54.3 s (+1.0%) |
| Energy per interval | 19.01 kWh | ≈ 19.02 kWh (+0.07%) |
