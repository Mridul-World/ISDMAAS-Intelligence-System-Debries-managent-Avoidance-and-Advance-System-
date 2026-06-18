# Phase 11 — ISDMAAS API + Dashboard

Unifies the full pipeline (predict → uncertainty → risk → maneuver → safety)
behind a REST API, with a browser dashboard.

## Files
- phase11_api.py    FastAPI service (the product)
- dashboard.html    browser UI (open in any browser)
- Dockerfile        containerized deploy
- requirements.txt  deps
- phase7/8/10 .py    pipeline modules (copy in)

## Run locally
```
pip install fastapi "uvicorn[standard]" numpy requests sgp4
# copy pipeline modules into this folder:
copy ..\phase7\phase7_collision.py .
copy ..\phase8\phase8_maneuver.py .
copy ..\phase10\phase10_safety.py .
# point at your phase55 outputs (has state_<norad>.json from Phase 6):
set PHASE55_DIR=..\phase55          (Windows CMD)   or leave default ../phase55

uvicorn phase11_api:app --reload --port 8000
```
Then open **dashboard.html** in your browser, and the interactive API docs at
**http://localhost:8000/docs**.

## Endpoints
- GET  /health
- GET  /satellites
- GET  /screen/{norad}?coarse_km=200
- POST /assess              {primary_norad, secondary_norad}
- POST /maneuver-plan       {primary_norad, synthetic_miss_km|secondary_norad, tca_hours, sat_mass_kg, fuel_available_kg}

## Docker
```
docker build -t isdmaas .
docker run -p 8000:8000 -v /path/to/isdmaas/phase55:/data/phase55 isdmaas
```

## Phase-9 hook
The maneuver proposer is isolated in `propose_maneuver()`. Swap the rule-based
Phase-8 planner for an RL policy there; Phase-10 safety validation still gates
every output unchanged.
