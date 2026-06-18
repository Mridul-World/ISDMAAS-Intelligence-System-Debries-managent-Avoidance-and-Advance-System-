"""
Phase 5.5 — STEP 5.2 VALIDATION against REAL truth.

Compares three systems on the validation split:
    1. SGP4 baseline             (the Phase-2-equivalent reference)
    2. SGP4 + pure-ML residual   (ablation, no physics loss)
    3. SGP4 + physics-ML residual (the core innovation)

Reports:
    - RMSE / median / P90 / P99 position error (km), overall + per satellite
    - Kepler-law compliance of predictions vs truth:
        |Δ specific energy|, |Δ angular momentum|, |Δ orbital period|
    - fraction of windows worse than SGP4
Output: reports/phase55_validation.json
"""
import os, json, math
import numpy as np
import torch
from train_phase55_rtn import RTNTransformer, rtn_to_eci, CFG as TCFG

MU = 398600.4418


def metrics(err):
    return {"rmse": float(np.sqrt((err ** 2).mean())), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "p99": float(np.percentile(err, 99)),
            "mean": float(err.mean())}


def kepler_panel(pred, true):
    """Physics consistency of predicted states vs truth (5.2)."""
    def invariants(st):
        r, v = st[:, :3], st[:, 3:]
        rn = np.linalg.norm(r, axis=1)
        E = 0.5 * (v ** 2).sum(1) - MU / rn
        h = np.linalg.norm(np.cross(r, v), axis=1)
        a = -MU / (2 * np.minimum(E, -1e-6))
        T = 2 * math.pi * np.sqrt(a ** 3 / MU)
        return E, h, T
    Ep, hp, Tp = invariants(pred); Et, ht, Tt = invariants(true)
    return {"d_energy_km2s2_mean": float(np.abs(Ep - Et).mean()),
            "d_ang_momentum_pct_mean": float((np.abs(hp - ht) / ht).mean() * 100),
            "d_period_s_mean": float(np.abs(Tp - Tt).mean())}


def predict(model_path, Xva, bva, rm, rs):
    net = RTNTransformer(Xva.shape[2], TCFG["D_MODEL"], TCFG["NHEAD"],
                         TCFG["NUM_LAYERS"], TCFG["DIM_FF"], TCFG["DROPOUT"])
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        pred_rtn = torch.cat([net(Xva[j:j + 1024]) * rs + rm
                              for j in range(0, len(Xva), 1024)], 0)
        final = bva.clone()
        final[:, :3] = bva[:, :3] - rtn_to_eci(pred_rtn, bva)
    return final


def main():
    ld = lambda f: torch.load(f, weights_only=False).float()
    Xva, bva, yva = ld("X_val.pt"), ld("baseline_val.pt"), ld("y_val.pt")
    sat = ld("sat_val.pt").numpy().astype(int)
    norm = torch.load(os.path.join(TCFG["MODEL_DIR"], "residual_norm_phase55.pt"),
                      weights_only=False)
    rm, rs = norm["rm"], norm["rs"]

    systems = {"sgp4": bva}
    for tag, fname in [("pure_ml", "rtn_pure_ml_phase55.pt"),
                       ("physics_ml", "rtn_physics_ml_phase55.pt")]:
        p = os.path.join(TCFG["MODEL_DIR"], fname)
        if os.path.exists(p):
            systems[tag] = predict(p, Xva, bva, rm, rs)
        else:
            print(f"NOTE: {fname} not found — skipping '{tag}'")

    yv = yva.numpy()
    rep = {"truth_source": "REAL ESA POD precise ephemeris (AUX_POEORB), TEME",
           "n_val": int(len(Xva)), "systems": {}}
    sgp4_err = np.linalg.norm(bva[:, :3].numpy() - yv[:, :3], axis=1)

    for tag, pred in systems.items():
        pn = pred.numpy()
        err = np.linalg.norm(pn[:, :3] - yv[:, :3], axis=1)
        entry = {"position_error_km": metrics(err),
                 "kepler_compliance_vs_truth": kepler_panel(pn, yv),
                 "per_satellite_rmse_km": {
                     str(s): float(np.sqrt((err[sat == s] ** 2).mean()))
                     for s in np.unique(sat)}}
        if tag != "sgp4":
            entry["improvement_vs_sgp4_pct"] = {
                k: 100 * (1 - metrics(err)[k] / metrics(sgp4_err)[k])
                for k in ("rmse", "median", "p90", "p99")}
            entry["worse_than_sgp4_fraction"] = float((err > sgp4_err).mean())
        rep["systems"][tag] = entry

    os.makedirs(TCFG["REPORT_DIR"], exist_ok=True)
    json.dump(rep, open(os.path.join(TCFG["REPORT_DIR"], "phase55_validation.json"),
                        "w"), indent=2)

    print("=" * 70)
    print(f"{'system':12}{'RMSE':>9}{'median':>9}{'P90':>9}{'P99':>9}"
          f"{'|ΔE|':>10}{'|ΔT|s':>8}")
    for tag, e in rep["systems"].items():
        m = e["position_error_km"]; k = e["kepler_compliance_vs_truth"]
        print(f"{tag:12}{m['rmse']:9.3f}{m['median']:9.3f}{m['p90']:9.3f}"
              f"{m['p99']:9.3f}{k['d_energy_km2s2_mean']:10.4f}"
              f"{k['d_period_s_mean']:8.3f}")
    print("Saved -> reports/phase55_validation.json")


if __name__ == "__main__":
    main()
