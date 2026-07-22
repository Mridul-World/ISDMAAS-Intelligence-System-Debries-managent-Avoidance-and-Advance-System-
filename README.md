# ISDMAAS
### Intelligent Satellite Debris Management & Autonomous Avoidance System

**Status:** Validated prototype (TRL 4–5) · **Owner:** Mridul Gupta  · **Last updated:** 2026-06

A decision-support system for satellite collision avoidance. Improves orbit prediction over the SGP4 baseline using a residual-correction Transformer, computes collision probability via a two-tier engine, and recommends minimum-fuel avoidance maneuvers — all running on public tracking data, with an architecture ready to ingest real operator CDM data.

---

## 1. Quick Start

```powershell
# clone
git clone https://github.com/<your-username>/isdmaas.git
cd isdmaas

# environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the backend
cd phase11
uvicorn phase11_api:app --port 8000

# run the console (new terminal)
python -m http.server 8080
# open http://localhost:8080/console.html
```

**Pre-flight check:** top bar shows `LINK ACTIVE` (green) and `DATA HH:MMZ` (green). If red, check the API terminal for startup errors.

---

## 2. What This Is (and Is Not)

| It IS | It is NOT |
|---|---|
| Decision-support software | Autonomous satellite control |
| Validated on real ESA precise-orbit data | Validated across all orbital regimes |
| Architected to ingest real CDM covariance | Validated with real operator CDMs (needs a pilot) |
| A deterministic minimum-Δv maneuver search | A reinforcement-learning policy (RL exists in early dev branches, not in the validated pipeline) |
| TRL 4–5 | Production / operationally deployed |

Read this table before writing any external-facing material about the system. Overstating any of the right column has already been an active failure mode in this project's history — see `docs/technical/07-validation-results.md` for what is and isn't proven.

---

## 3. Repository Structure

```
isdmaas/
├── phase7_collision.py         Collision risk engine — TCA, covariance, Pc (Foster + Chan)
├── phase11_api.py              FastAPI backend — all HTTP endpoints
├── phase11/console.html        Mission console — Three.js + satellite.js frontend
├── phase12/event_library.py    Historical collision/ASAT event database
├── cdm_ingest.py                CCSDS CDM parser (KVN + XML) — real-covariance bridge
├── multi_regime_validation.py  Validation harness — runs on any satellite/regime given truth data
├── docs/technical/              Full technical documentation (see §5)
├── models/                      Trained model checkpoints (see Repo Hygiene, §7)
├── data/samples/                Small format-reference files only
└── outputs/                     Generated deliverables (paper, pitch, pilot proposal)
```

---

## 4. System Architecture

```
DATA LAYER
  CelesTrak (TLE/GP catalog, SOCRATES feed) · Space-Track · ESA POD (training truth only)
  · CDM parser (real covariance, when operator-supplied)
        │
        ▼
PREDICTION LAYER            (phase55, the core IP — see docs/technical/02-ai-models.md)
  SGP4 baseline → Residual-Correction Transformer → corrected trajectory
  (RTN-frame residual learned, added back to SGP4, rotated to ECI)
        │
        ▼
RISK LAYER                  (phase7_collision.py — see docs/technical/04-collision-risk-engine.md)
  TCA search → covariance propagation/combination → 2D encounter-plane projection
  → Pc (Foster quadrature + Chan cross-check)
        │
        ▼
DECISION LAYER               (see docs/technical/05-maneuver-planning.md)
  Δv candidate search → re-screen against full catalog → safety validation
  → minimum-fuel recommendation
        │
        ▼
PRESENTATION LAYER            (phase11_api.py + console.html)
  FastAPI backend · Three.js/satellite.js live mission console
```

Full detail per layer: `docs/technical/01-architecture.md`.

---

## 5. Documentation Map

| File | Contents |
|---|---|
| [`00-INDEX.md`](docs/technical/00-INDEX.md) | Navigation, doc status, last-updated per section |
| [`01-architecture.md`](docs/technical/01-architecture.md) | Full module map, data flow, design rationale |
| [`02-ai-models.md`](docs/technical/02-ai-models.md) | Residual Transformer spec, RTN frame, training data, architecture-selection history (Ridge → LSTM → Transformer) |
| [`03-orbital-mechanics.md`](docs/technical/03-orbital-mechanics.md) | SGP4 baseline behavior, RTN derivation, TCA search |
| [`04-collision-risk-engine.md`](docs/technical/04-collision-risk-engine.md) | Covariance model, the calibration bug + fix (with real numbers), Pc computation |
| [`05-maneuver-planning.md`](docs/technical/05-maneuver-planning.md) | Δv search, safety re-screening — explicitly not RL |
| [`06-api-software.md`](docs/technical/06-api-software.md) | Endpoint reference, library choices with rationale |
| [`07-validation-results.md`](docs/technical/07-validation-results.md) | All validation numbers, dated per run — append-only as new regimes/pilots are added |
| [`08-math-appendix.md`](docs/technical/08-math-appendix.md) | Every equation, pulled out of prose docs for single-source-of-truth |

**Update rule:** when you change model parameters, covariance constants, or add validation data — update the relevant numbered doc in the same commit as the code change. Do not let docs drift from `phase7_collision.py` / `phase55` — this has already caused one real bug (see §6).

---

## 6. Key Technical Facts (for fast recall — full derivations in `docs/technical/`)

**Model:** Residual-correction Transformer. 2 layers, 4 attention heads, d_model=64. Input: 30-step sequence, 10-minute grid, 23 features. Predicts the RTN-frame residual between SGP4 and true position; not an end-to-end trajectory predictor.

**Training data:** ESA Copernicus POD (~5cm truth), 5 Sentinel satellites (39634, 40697, 42063, 41335, 43437), 1 year. Inference uses only public TLEs.

**Validated results:**
- Multi-horizon RMSE vs SGP4: +9.9% (6h) → +33.1% (7d), improvement grows with horizon.
- LOSO generalization: +7.4% mean RMSE on unseen satellites; median-error transfer is bounded — only strong when a platform "sibling" exists in training (see `07-validation-results.md` for the full explanation — this is the most important nuance in the project's results).
- Physics-consistency loss term: **null result**, reported honestly. Model is "residual-correction," not "physics-informed" — do not use the latter term.

**Collision risk:** Foster 2D quadrature (primary) + Chan's method (cross-check). Threshold Pc = 1e-4. Covariance base_sigma was found mis-calibrated for TLE objects (too tight, causing Pc ≈ 1e-19 instead of realistic ≈1e-4 for a ~1km miss) — fixed by raising radial/cross-track base and growth rate. Details + before/after numbers: `04-collision-risk-engine.md`.

**Maneuver planning:** deterministic minimum-Δv search with full-catalog safety re-screen. **Not RL** — RL exists in exploratory code, not the validated pipeline.

**CDM ingestion:** real CCSDS 508.0-B-1 parser (`cdm_ingest.py`), KVN + XML. Architecturally ready; unvalidated against real operator CDMs (requires a pilot).

---

## 7. Repo Hygiene — read before your next commit

1. **Never commit the 8GB ESA POD dataset or trained weights directly.** Use Git LFS (`git lfs track "*.pt"`) or keep them off-repo with a documented external location. `.gitignore` already blocks `data/esa_pod/` and model checkpoint extensions.
2. **Never commit credentials.** Space-Track login, API keys — check `.gitignore` covers your actual filenames before first commit.
3. **One numbered doc per topic, per §5.** Don't let architecture notes leak into `07-validation-results.md` or vice versa — that document specifically should stay numbers-only, dated per validation run, so it's diffable and doesn't require prose editing to add a new result.
4. **Every `*.bak` from a patch script (`fix_covariance.py`, `fix_celestrak.py`, etc.) is gitignored** — don't force-add them.
5. **Backup discipline:** this repo (code) → private GitHub. Full tree (code + 8GB data + models) → external drive, updated after every significant session. GitHub alone is not your backup for the data/model assets.

---

## 8. Roadmap (honest, TRL-anchored)

| Stage | Status |
|---|---|
| Validated prototype (prediction, risk engine, console, CDM parser, validation harness) | ✅ Done — TRL 4–5 |
| Provisional patent filing (prediction-residual method) | ⏳ Do before any public disclosure |
| Pilot with a design-partner operator (real CDMs + multi-regime POD) | ⏳ Next milestone — closes the multi-regime and real-covariance validation gaps |
| Operational shadow-mode validation | Blocked on pilot |
| TRL 6–7 | Blocked on pilot |

See `outputs/ISDMAAS_Pilot_Proposal.docx` for the concrete ask, and `docs/technical/07-validation-results.md` §"Open Validation Gaps" for exactly what data would close each gap.

---

## 9. Citation

If referencing this work academically, see `outputs/ISDMAAS_Paper.docx` for the formal citation and methodology writeup.

---

## 10. Contact

Mridul Gupta · [mridul1735@gmail.com]
