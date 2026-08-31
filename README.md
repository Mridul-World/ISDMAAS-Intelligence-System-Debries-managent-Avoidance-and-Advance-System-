# ISDMAAS
### Intelligent Satellite Debris Management & Autonomous Avoidance System

**Status:** Validated prototype (TRL 4–5) · **Owner:** Mridul Gupta (Zyton) · **Last updated:** 2026-08-31

> **Open finding (2026-08-31):** the prediction-layer validation numbers do not
> reproduce from the artifacts in this repository. Read
> [`docs/technical/09-model-artifact-audit.md`](docs/technical/09-model-artifact-audit.md)
> before citing any prediction result. The risk, planning and safety layers are
> unaffected and are covered by the test suite in `phase11/tests/`.

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
pip install -r phase11/requirements-dev.txt

# run the backend
cd phase11
uvicorn phase11_api:app --port 8000

# run the console (new terminal)
python -m http.server 8080
# open http://localhost:8080/dashboard.html
```

**Pre-flight check:** the top bar shows **Link active** with a green indicator, and a data badge with the catalog object count. If it shows *Service offline*, check the API terminal for startup errors.

Deploying anywhere other than a laptop: read [`phase11/DEPLOY.md`](phase11/DEPLOY.md) first. The service refuses to start in production mode without a secret key and an explicit CORS origin list.

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
├── phase11/dashboard.html      Mission console — markup (console.css / console.js alongside)
├── phase11/isdmaas_core/       Service layer — config, security, astrodynamics, screening
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
PRESENTATION LAYER            (phase11_api.py + dashboard.html)
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
| [`09-model-artifact-audit.md`](docs/technical/09-model-artifact-audit.md) | **Open finding** — why the prediction-layer numbers don't reproduce, and what closes it |
| [`phase11/DEPLOY.md`](phase11/DEPLOY.md) | Running the service outside a laptop: configuration, TLS, health probes, honest security posture |

**Update rule:** when you change model parameters, covariance constants, or add validation data — update the relevant numbered doc in the same commit as the code change. Do not let docs drift from `phase7_collision.py` / `phase55` — this has already caused one real bug (see §6).

---

## 6. Key Technical Facts (for fast recall — full derivations in `docs/technical/`)

**Model:** Residual-correction Transformer. 2 layers, 4 attention heads, d_model=64. Input: 30-step sequence, 10-minute grid, 23 features. Predicts the RTN-frame residual between SGP4 and true position; not an end-to-end trajectory predictor.

**Training data:** ESA Copernicus POD (~5cm truth), 5 Sentinel satellites (39634, 40697, 42063, 41335, 43437), 1 year. Inference uses only public TLEs.

**Validated results — ⚠ see [`09-model-artifact-audit.md`](docs/technical/09-model-artifact-audit.md) before citing any of these.**

As of 2026-08-31 the checkpoint in `phase55/model/` does not reproduce the
committed single-horizon validation report: re-scoring it against the committed
validation tensors gives **+2.95% RMSE and −2.57% median** where the report
claims +7.62% and +36.6%, and its predictions are anti-correlated with the target
on the radial and cross-track axes. The model file on disk is byte-identical to
a two-week-older backup that appears to have been restored over the trained one.
The audit document has the full evidence and the remediation steps.

- Multi-horizon RMSE vs SGP4: +9.9% (6h) → +33.1% (7d). **Not re-verified** —
  from a separate run whose checkpoint is not in the repository.
- LOSO generalization: +7.4% mean RMSE on unseen satellites; median-error
  transfer is bounded and only strong when a platform "sibling" exists in
  training. **Not re-verified**, same reason.
- Physics-consistency loss term: **null result**, reported honestly. Model is
  "residual-correction," not "physics-informed" — do not use the latter term.
  This one is *strengthened* by the audit: the two checkpoints differ by at most
  6×10⁻⁴ km in their predictions.

The risk, planning and safety layers are unaffected — they are deterministic
astrodynamics with no learned component, and they are covered by 115 tests in
`phase11/tests/` that check against independent ground truth.

**Collision risk:** Foster 2D quadrature (primary) + Chan's method (cross-check). Threshold Pc = 1e-4. Covariance base_sigma was found mis-calibrated for TLE objects (too tight, causing Pc ≈ 1e-19 instead of realistic ≈1e-4 for a ~1km miss) — fixed by raising radial/cross-track base and growth rate. Details + before/after numbers: `04-collision-risk-engine.md`.

**Maneuver planning:** deterministic minimum-Δv search with full-catalog safety re-screen. **Not RL** — RL exists in exploratory code, not the validated pipeline.

**CDM ingestion:** real CCSDS 508.0-B-1 parser (`cdm_ingest.py`), KVN + XML. Architecturally ready; unvalidated against real operator CDMs (requires a pilot).

---

## 7. Repo Hygiene — read before your next commit

1. **Never commit the 8GB ESA POD dataset or trained weights directly.** Use Git LFS (`git lfs track "*.pt"`) or keep them off-repo with a documented external location. `.gitignore` already blocks `data/esa_pod/` and model checkpoint extensions.
2. **Never commit credentials.** Space-Track login, API keys — check `.gitignore` covers your actual filenames before first commit.
3. **One numbered doc per topic, per §5.** Don't let architecture notes leak into `07-validation-results.md` or vice versa — that document specifically should stay numbers-only, dated per validation run, so it's diffable and doesn't require prose editing to add a new result.
4. **The `fix_*.py` / `add_*.py` patch scripts are archived and disabled.** They rewrote source files in place by string substitution, and one of them is why the entire authentication surface silently disappeared from `phase11_api.py`: a script appended `install_auth(app)`, a later script rewrote the file from a stale `.bak`, and the wiring was lost while the console still showed a sign-in dialog. Each now refuses to run. **Change code by editing code and adding a test.** Their `*.bak` output is gitignored — don't force-add it.

5. **⚠ The collision engine is duplicated six times and five copies are stale.**
   `phase7/`, `phase8/`, `phase9/`, `phase10/`, `phase12/` and `phase13/` each
   contain a byte-identical `phase7_collision.py`, and all six still carry the
   one-sided TCA clamp and the Chan-series overflow that were fixed in
   `phase11/phase7_collision.py`. Anything those historical tools report about a
   miss distance is unreliable.

   The corrected engine was **not** copied over them, because several of their
   callers do `pre["pc"] > PC_THRESHOLD` and the corrected `assess_conjunction`
   returns `pc: None` for a conjunction below the short-encounter speed limit —
   that would turn a wrong number into a crash in tools that cannot be exercised
   from this repository. Before propagating it, guard those comparisons for
   `None`, then replace all six with a single import of the phase11 module rather
   than another copy.

6. **Backup discipline:** this repo (code) → private GitHub. Full tree (code + 8GB data + models) → external drive, updated after every significant session. GitHub alone is not your backup for the data/model assets.

7. **⚠ Credentials were committed and are still in git history.** `phase11/auth_store.json` (password hashes and live session tokens), `phase11/isdmaas_ops.db` and the catalog caches have been removed from tracking and gitignored, but **removing them from HEAD does not remove them from history.** Before this repository is shared or made public: rotate every affected password, and purge the files from history with `git filter-repo` (or accept the repo as compromised and start a fresh one). Every session token in that file must be treated as public — the new store discards them on import for exactly this reason.

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

If referencing this work academically, see `outputs/ISDMAAS_IEEE_Paper.docx` for the formal citation and methodology writeup.

---

## 10. Contact

Mridul Gupta · Zyton · [email]
