# Report & Reference Files

## Final Paper

**[Injection_Molding_Paper_v2.pdf](Injection_Molding_Paper_v2.pdf)**  
*Reducing Scrap in Injection Molding: A DAG-Informed Causal Decision Analysis*  
Constructor University Datathon 2026 · Insytes Manufacturing Challenge  
Team: Limit Breakers

This is the authoritative source of truth for all numbers in this repository. Where
repository outputs differ from paper values, the discrepancy is documented in the
README.md at the root of this repo.

## DAG Documentation

**[Injection_Molding_DAG_Notes.pdf](Injection_Molding_DAG_Notes.pdf)**  
Narrative interpretation of the causal graph. Covers the five causal chains, why
certain variables are confounders vs. mediators, and the identification assumptions
underlying the adjustment sets used in `src/utils.py`.

## Challenge Reference

**[Datathon_Student_Guide.pdf](Datathon_Student_Guide.pdf)**  
Dataset description, variable ontology, and the challenge's suggested analytical
framework. The variable classification in `src/utils.py` is derived directly from
the ontology defined in this guide.

---

## Key Tables from the Paper

### Table 2 — DAG-Adjusted Effect Estimates (paper values)

| Lever | β (std) | 95% CI | Per op. step |
|---|---|---|---|
| `cooling_time_s` | **−1.75** | [−1.85, −1.65] | −0.61 p.p. / +1.5 s |
| `mold_temperature_c` | +0.88 | [+0.80, +0.96] | +0.40 p.p. / +5 °C |
| `injection_pressure_bar` | +0.31 | [+0.22, +0.41] | +0.11 p.p. / +50 bar |
| `dryer_dewpoint_c` | +0.09 | [+0.05, +0.13] | +0.15 p.p. / +5 °C |
| `maintenance_days_since_last` | +0.10 | [+0.07, +0.14] | +0.07 p.p. / +7 days |

> Repository reproduction: β (std) values match within 1%. Natural-unit (p.p./s) value
> for cooling matches at −0.41 p.p./s. See root README for full comparison table.

### Table 6 — Combined Package Impact (paper values)

| Metric | Baseline | With package |
|---|---|---|
| Mean scrap rate | 4.44% | ≈ 3.85% (−13% relative) |
| Cycle time | 53.7 s | ≈ 54.3 s (+1.0%) |
| Energy per interval | 19.01 kWh | ≈ 19.02 kWh (+0.07%) |

> Repository reproduction: 4.44% → ≈3.88% (−12.6%). Within 5% of paper.
> Difference attributed to GBR stochasticity and demo dataset version.
