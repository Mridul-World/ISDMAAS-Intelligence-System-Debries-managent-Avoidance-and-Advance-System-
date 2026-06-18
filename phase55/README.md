# Phase 5.5 — Real Truth Validation (INDUSTRY BUILD, multi-satellite)

ONE command produces, permanently:
- the real-truth dataset (`data/phase55_dataset.csv`) — scraping done forever
- training tensors + scaler
- the trusted physics-informed model (`model/rtn_physics_ml_phase55.pt`)
- the pure-ML ablation model (5.2 comparison)
- the publishable validation report (`reports/phase55_validation.json`)
- an OPERATIONAL future-orbit predictor (`infer_future.py`) needing only TLEs

## Run
```
pip install requests astropy sgp4 torch scikit-learn pandas numpy joblib

set SPACETRACK_USER=...   set SPACETRACK_PASS=...     (space-track.org, free)
set CDSE_USER=...         set CDSE_PASS=...           (dataspace.copernicus.eu, free)

python run_all.py
```
Stages checkpoint automatically; rerun resumes. Force from a stage:
`python run_all.py --from build`

## After completion
```
python infer_future.py 39634        # Sentinel-1A state, now + 24 h, AI-corrected
python validate_phase55.py          # reprint the numbers
```

## Fleet
config.py SATELLITES: S1A, S2A, S2B, S3A, S3B (ESA POD AUX_POEORB truth,
~5 cm). Satellites without POEORB products in CDSE are skipped gracefully.
Add satellites by adding registry entries.

## Scientific design (what makes the numbers defensible)
1. Truth = real ESA POD ephemerides — independent of SGP4. No RK4 circularity.
2. EOF is ITRF; SGP4 is TEME. Rigorous astropy ITRS->TEME incl. velocity terms
   (round-trip verified to <1e-9 km).
3. Baseline TLE rule: epoch <= target - horizon. No information leakage.
4. INPUT features are TLE/SGP4-derived (deployable: inference needs no
   precise ephemeris — matches training distribution exactly).
5. Chronological 80/20 split per satellite.
6. Physics-informed loss (5.1): energy + angular momentum + SMA + period
   penalties (frozen Phase-5 trainer). 5.2: validate_phase55.py reports
   Kepler-law compliance and SGP4 vs pure-ML vs physics-ML.

## Sanity anchors (real data)
- SGP4 vs POE truth @ ~1 day TLE age: ~0.5–3 km (along-track dominated).
  Thousands of km => frame bug. ~0 km => truth==SGP4 bug. Stop either way.
- Honest expectation: 30–60% RMSE reduction. >85% on real truth: investigate
  before celebrating.
- Acceptance: worse_than_sgp4_fraction < 0.25; improvement holds at P90/P99.

## Phase 6 hook
New trainer (do not modify this one): head Linear(d//2,3) -> Linear(d//2,6) =
(mu_RTN, log_sigma_RTN), Gaussian NLL on RTN residual, MC-dropout ->
covariance -> Phase 7 collision probability.
