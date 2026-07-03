# ISDMAAS Technical Documentation — Index

Each doc is self-contained. Cross-reference by filename, don't duplicate content across docs.

| # | Doc | Owns | Status | Last updated |
|---|---|---|---|---|
| 01 | [Architecture](01-architecture.md) | System-level module map, data flow | Draft — populate from `ISDMAAS_Technical_Reference.md` §1 | 2026-06 |
| 02 | [AI Models](02-ai-models.md) | Transformer spec, RTN frame, training data, architecture-selection history | Draft — from §2 | 2026-06 |
| 03 | [Orbital Mechanics](03-orbital-mechanics.md) | SGP4 baseline, RTN derivation, TCA search physics | Draft — extract from §2.3, §4.1 | 2026-06 |
| 04 | [Collision Risk Engine](04-collision-risk-engine.md) | Covariance model + calibration bug/fix, Pc computation | Draft — from §4 | 2026-06 |
| 05 | [Maneuver Planning](05-maneuver-planning.md) | Δv search, safety re-screen. **Not RL.** | Draft — from §5 | 2026-06 |
| 06 | [API & Software](06-api-software.md) | Endpoint reference, library rationale | Draft — from §6 + phase11_api.py routes | 2026-06 |
| 07 | [Validation Results](07-validation-results.md) | All numbers, dated, append-only | Draft — from §3 | 2026-06 |
| 08 | [Math Appendix](08-math-appendix.md) | Every equation, single source of truth | Draft — from §9 | 2026-06 |

## Update discipline
- Change model/covariance/data → update the matching doc **in the same commit**.
- New validation run → append a new dated section to `07`, never overwrite prior results.
- New equation used anywhere → goes in `08` first; other docs reference it (`see Eq. A.3`), don't restate it.

## Source
This split was generated from `ISDMAAS_Technical_Reference.md` (single-file version, superseded by this structure) and `ISDMAAS_IEEE_Paper.docx` (formal writeup).
