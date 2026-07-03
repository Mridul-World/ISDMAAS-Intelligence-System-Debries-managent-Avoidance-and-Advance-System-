"""
multi_regime_validation.py  —  ISDMAAS validation across ANY satellites / orbital
regimes you provide.
================================================================================
WHAT THIS IS (honest scope):
A validation harness that measures ISDMAAS's orbit-prediction accuracy (vs the
SGP4 baseline) for ANY set of satellites you have precise-orbit truth for —
grouped by orbital regime (LEO-low, LEO-high, SSO, MEO, GEO, HEO). It produces
the same RMSE-improvement metrics as your existing Sentinel validation, but
regime-by-regime, so you can show an operator "here is our accuracy in YOUR
regime", not just "in LEO on 5 Sentinels".

WHY IT MATTERS (Gap 1):
Your validation currently rests on 5 Sentinels in one regime. This harness lets
you EXTEND that validation the moment you have truth data for more satellites —
e.g. when a pilot operator provides their POD, or when you download more public
precise ephemerides. The capability is real and ready; it runs on whatever data
exists. It does NOT fabricate satellites — with no new data it reports only what
you have.

WHAT IT NEEDS:
Per satellite: a precise-orbit truth file (the same kind you use for the Sentinels
— sampled ECI/ECEF position states over time) and a TLE/GP element for the SGP4
baseline. Point the harness at a folder of these and it does the rest.

INPUT LAYOUT (default):
  data_regimes/
    <SATNAME>/
      truth.csv        # columns: epoch_utc, x_km, y_km, z_km  (precise truth)
      tle.txt          # two-line element (or 3-line with name) for SGP4 baseline
      meta.json        # {"norad": 12345, "regime": "SSO"}  (regime optional; auto-detected)

OUTPUT:
  - per-satellite RMSE (SGP4 vs ISDMAAS-corrected), % improvement
  - per-regime aggregate
  - regime_validation_results.csv  +  regime_validation.png
================================================================================
"""
import os, json, glob
import numpy as np
from datetime import datetime, timezone

# ---- optional plotting (degrades gracefully) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ----------------------------------------------------------------------------
# Regime classification (from semi-major axis / inclination)
# ----------------------------------------------------------------------------
def classify_regime(alt_km, inc_deg, ecc=0.0):
    """Coarse orbital-regime label from mean altitude, inclination, eccentricity."""
    if ecc > 0.25:
        return "HEO"
    if alt_km < 2000:
        if 96 <= inc_deg <= 102:
            return "SSO"          # sun-synchronous (most EO sats)
        return "LEO-low" if alt_km < 700 else "LEO-high"
    if alt_km < 30000:
        return "MEO"
    if 35000 <= alt_km <= 36500 and inc_deg < 15:
        return "GEO"
    return "MEO"


# ----------------------------------------------------------------------------
# Truth + baseline loading
# ----------------------------------------------------------------------------
def load_truth_csv(path):
    """Load precise-truth states: epoch_utc,x_km,y_km,z_km. Returns (times, pos)."""
    import csv
    times, pos = [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            # accept a few common column spellings
            ep = row.get("epoch_utc") or row.get("epoch") or row.get("time")
            x = row.get("x_km") or row.get("x")
            y = row.get("y_km") or row.get("y")
            z = row.get("z_km") or row.get("z")
            if None in (ep, x, y, z):
                continue
            times.append(ep)
            pos.append([float(x), float(y), float(z)])
    return times, np.array(pos)


def sgp4_predict(tle_text, times_utc):
    """Propagate the TLE with SGP4 at the given UTC timestamps. Returns Nx3 km."""
    from sgp4.api import Satrec, jday
    lines = [ln for ln in tle_text.strip().splitlines() if ln.strip()]
    l1, l2 = (lines[-2], lines[-1])  # last two lines (handles 3-line w/ name)
    sat = Satrec.twoline2rv(l1, l2)
    out = []
    for t in times_utc:
        dt = _parse_dt(t)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second + dt.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)
        out.append(r if e == 0 else [np.nan, np.nan, np.nan])
    return np.array(out)


def _parse_dt(s):
    s = s.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unparseable epoch: {s}")


# ----------------------------------------------------------------------------
# ISDMAAS-corrected prediction hook
# ----------------------------------------------------------------------------
def isdmaas_predict(tle_text, times_utc, model=None):
    """
    Produce the ISDMAAS-corrected prediction.

    PLUG-IN POINT: this is where your trained residual-correction model is applied
    on top of the SGP4 baseline. If you pass `model=None`, the harness uses the
    SGP4 baseline as a placeholder so the pipeline runs end-to-end; swap in your
    real model inference here to get the corrected track.

    To wire your model: load it once in main(), pass it in, and replace the body
    below with: base = sgp4_predict(...); residual = model.infer(features(...));
    return base + residual.
    """
    base = sgp4_predict(tle_text, times_utc)
    if model is None:
        return base  # placeholder: no correction (RMSE improvement will read ~0)
    try:
        return model.predict_corrected(tle_text, times_utc)  # your model's API
    except Exception:
        return base


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def rmse_3d(pred, truth):
    """3D position RMSE (km), ignoring NaNs."""
    d = pred - truth
    err = np.sqrt(np.sum(d * d, axis=1))
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def median_err(pred, truth):
    d = pred - truth
    err = np.sqrt(np.sum(d * d, axis=1))
    err = err[np.isfinite(err)]
    return float(np.median(err)) if len(err) else float("nan")


# ----------------------------------------------------------------------------
# Main validation loop
# ----------------------------------------------------------------------------
def validate_folder(root="data_regimes", model=None):
    sat_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not sat_dirs:
        print(f"No satellite folders found under '{root}/'.")
        print("Create data_regimes/<SATNAME>/ with truth.csv + tle.txt + meta.json")
        return []

    results = []
    for d in sorted(sat_dirs):
        name = os.path.basename(d)
        truth_p = os.path.join(d, "truth.csv")
        tle_p = os.path.join(d, "tle.txt")
        meta_p = os.path.join(d, "meta.json")
        if not (os.path.exists(truth_p) and os.path.exists(tle_p)):
            print(f"  [skip] {name}: missing truth.csv or tle.txt")
            continue
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        times, truth = load_truth_csv(truth_p)
        if len(truth) < 10:
            print(f"  [skip] {name}: too few truth points ({len(truth)})")
            continue
        tle = open(tle_p).read()

        sgp4_track = sgp4_predict(tle, times)
        isd_track = isdmaas_predict(tle, times, model=model)

        sgp4_rmse = rmse_3d(sgp4_track, truth)
        isd_rmse = rmse_3d(isd_track, truth)
        improve = (1 - isd_rmse / sgp4_rmse) * 100 if sgp4_rmse > 0 else 0.0

        # regime: from meta, else auto-detect from mean altitude/inclination
        regime = meta.get("regime")
        if not regime:
            alt = np.mean(np.linalg.norm(truth, axis=1)) - 6371.0
            regime = classify_regime(alt, meta.get("inclination", 98.0))

        results.append({
            "satellite": name, "norad": meta.get("norad"),
            "regime": regime, "n_points": len(truth),
            "sgp4_rmse_km": round(sgp4_rmse, 3),
            "isdmaas_rmse_km": round(isd_rmse, 3),
            "improvement_pct": round(improve, 1),
            "sgp4_median_km": round(median_err(sgp4_track, truth), 3),
            "isdmaas_median_km": round(median_err(isd_track, truth), 3),
        })
        print(f"  {name:16s} [{regime:8s}] SGP4 {sgp4_rmse:6.3f} -> ISDMAAS "
              f"{isd_rmse:6.3f} km  ({improve:+.1f}%)")
    return results


def summarize_by_regime(results):
    from collections import defaultdict
    byr = defaultdict(list)
    for r in results:
        byr[r["regime"]].append(r)
    print("\n" + "=" * 56)
    print("PER-REGIME SUMMARY")
    print("=" * 56)
    summary = []
    for regime, rows in sorted(byr.items()):
        mean_imp = np.mean([r["improvement_pct"] for r in rows])
        mean_sgp4 = np.mean([r["sgp4_rmse_km"] for r in rows])
        mean_isd = np.mean([r["isdmaas_rmse_km"] for r in rows])
        summary.append({"regime": regime, "n_sats": len(rows),
                        "mean_sgp4_rmse_km": round(mean_sgp4, 3),
                        "mean_isdmaas_rmse_km": round(mean_isd, 3),
                        "mean_improvement_pct": round(mean_imp, 1)})
        print(f"  {regime:10s}: {len(rows)} sat(s), "
              f"SGP4 {mean_sgp4:.3f} -> ISDMAAS {mean_isd:.3f} km "
              f"({mean_imp:+.1f}% mean)")
    return summary


def write_outputs(results, summary):
    import csv
    with open("regime_validation_results.csv", "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
    print("\nWrote regime_validation_results.csv")

    if HAVE_PLT and summary:
        regimes = [s["regime"] for s in summary]
        imps = [s["mean_improvement_pct"] for s in summary]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(regimes, imps, color="#3d8bff", edgecolor="#1a3a5c")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("Mean RMSE improvement vs SGP4 (%)")
        ax.set_title("ISDMAAS orbit-prediction accuracy by orbital regime")
        for b, v in zip(bars, imps):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        plt.tight_layout()
        plt.savefig("regime_validation.png", dpi=140)
        print("Wrote regime_validation.png")
    elif not HAVE_PLT:
        print("(matplotlib not available — skipped the figure)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ISDMAAS multi-regime validation")
    ap.add_argument("--data", default="data_regimes",
                    help="folder of <SATNAME>/ dirs with truth.csv + tle.txt")
    args = ap.parse_args()

    print("=" * 56)
    print("ISDMAAS MULTI-REGIME VALIDATION HARNESS")
    print("=" * 56)
    print(f"Scanning '{args.data}/' for satellites...\n")

    # ---- WIRE YOUR MODEL HERE ----
    # from your_model_module import load_model
    # model = load_model("path/to/checkpoint")
    model = None   # placeholder: runs the pipeline with SGP4 baseline only
    if model is None:
        print("NOTE: no model wired in — running with SGP4 baseline as placeholder.")
        print("      Improvement will read ~0 until you plug in your trained model")
        print("      in isdmaas_predict() / main(). The PIPELINE is what's being")
        print("      validated here; results become real once data + model are set.\n")

    results = validate_folder(args.data, model=model)
    if not results:
        return
    summary = summarize_by_regime(results)
    write_outputs(results, summary)
    print("\nDone. Add more satellites under the data folder to extend coverage.")


if __name__ == "__main__":
    main()
