"""
================================================================================
MULTI-HORIZON EVALUATION  —  6 h / 12 h / 1 d / 3 d / 7 d
================================================================================
Shows how prediction accuracy AND the advantage over SGP4 change with forecast
horizon. This is the methodologically correct study: HORIZON_S is a dataset-
construction parameter (the baseline is SGP4 propagated to t+HORIZON), so for
each horizon we REBUILD the dataset, RETRAIN, and VALIDATE through the existing
pipeline. (Running the 1-day model at 7 days would be extrapolation — not done.)

    python multi_horizon_eval.py                  # all 5 horizons (long: trains 2 models x5)
    python multi_horizon_eval.py 21600 86400      # just 6 h and 1 d
    python multi_horizon_eval.py --report-only     # rebuild table/plot from saved runs
    python multi_horizon_eval.py --inspect         # print your validation-JSON structure & exit

PREREQUISITE: one successful normal run (so TLE/POE data exist on disk).
The harness sets ISDMAAS_HORIZON_S per horizon and runs the pipeline stages.
It backs up your 1-day production models and restores them at the end.

ROBUSTNESS: the summary auto-discovers RMSE / P90 for sgp4 / pure-ML / physics-ML
no matter how your validation JSON nests or names them (run --inspect first if
unsure). If discovery misses a field, edit METRIC_HINTS below.
================================================================================
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

HORIZON_LABELS = {21600: "6 h", 43200: "12 h", 86400: "1 day",
                  259200: "3 days", 604800: "7 days"}
DEFAULT_HORIZONS = [21600, 43200, 86400, 259200, 604800]
HZ_DIR = "reports/horizons"
VALIDATION_JSON = "reports/phase55_validation.json"

# substrings used to recognise each system + metric in your JSON (case-insensitive)
SYS_HINTS = {
    "sgp4":       ["sgp4", "baseline"],
    "pure_ml":    ["pure", "pure_ml", "pure_transformer", "ml_only"],
    "physics_ml": ["physics", "physics_ml", "physics_transformer", "residual"],
}
RMSE_HINTS = ["rmse", "rmse_km", "pos_rmse", "position_rmse", "rmse_pos_km"]
P90_HINTS  = ["p90", "p90_km", "pct90", "percentile_90", "p90_pos_km"]

# stages to run per horizon (edit names here if your files differ — see --inspect note)
def stages_for(env):
    pure_flag = _detect_pure_flag()
    return [
        "build_truth_dataset.py",
        "export_phase55_tensors.py",
        "train_phase55_rtn.py",
        f"train_phase55_rtn.py {pure_flag}".strip(),
        "validate_phase55.py",
    ]

PRODUCTION_MODELS = ["model/rtn_physics_ml_phase55.pt", "model/rtn_pure_ml_phase55.pt"]
HORIZON_SPECIFIC = PRODUCTION_MODELS + [
    "X_train.pt", "X_val.pt", "y_train.pt", "y_val.pt",
    "baseline_train.pt", "baseline_val.pt", "residual_rtn_train.pt",
    "residual_rtn_val.pt", "sat_train.pt", "sat_val.pt",
    "data/phase55_dataset.csv", VALIDATION_JSON,
]


def _detect_pure_flag():
    """Find how the trainer selects the pure-ML ablation (--pure-ml / --pure / --no-physics)."""
    try:
        src = open("train_phase55_rtn.py").read()
    except Exception:
        return "--pure-ml"
    for flag in ["--pure-ml", "--pure_ml", "--pure", "--no-physics", "--ablation"]:
        if flag.replace("--", "").replace("-", "_") in src or flag in src:
            return flag
    return "--pure-ml"


# ---------------------------------------------------------------- json discovery
def _walk(obj, path=""):
    """Yield (path, key, value) for every scalar in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                yield from _walk(v, p)
            else:
                yield p, k, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _find_metric(d, sys_hints, metric_hints):
    """Find a numeric value whose path mentions a system hint AND a metric hint.
       Also handles list-of-records where the system name is a sibling label
       field (e.g. {"model":"SGP4 baseline","rmse":6.67})."""
    best = None
    # pass 1: path-based (handles nested dicts keyed by system name)
    for path, key, val in _walk(d):
        if not isinstance(val, (int, float)):
            continue
        pl = path.lower()
        if any(s in pl for s in sys_hints) and any(m in pl for m in metric_hints):
            score = (("overall" in pl) + ("all" in pl) + ("mean" in pl) + ("total" in pl))
            if best is None or score > best[0]:
                best = (score, float(val))
    if best:
        return best[1]
    # pass 2: record-based (system name in a sibling label field)
    def scan_records(obj):
        recs = []
        if isinstance(obj, dict):
            # is this dict itself a record with a label + metrics?
            recs.append(obj)
            for v in obj.values():
                recs += scan_records(v)
        elif isinstance(obj, list):
            for v in obj:
                recs += scan_records(v)
        return recs
    for rec in scan_records(d):
        if not isinstance(rec, dict):
            continue
        label = " ".join(str(v).lower() for v in rec.values() if isinstance(v, str))
        if not any(s in label for s in sys_hints):
            continue
        for k, v in rec.items():
            if isinstance(v, (int, float)) and any(m in k.lower() for m in metric_hints):
                return float(v)
    return None


def build_summary(horizons):
    rows = []
    for H in horizons:
        p = os.path.join(HZ_DIR, f"val_{H}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rec = {"horizon_s": H, "horizon": HORIZON_LABELS.get(H, f"{H}s")}
        for sysname, sh in SYS_HINTS.items():
            rec[f"{sysname}_rmse_km"] = _find_metric(d, sh, RMSE_HINTS)
            rec[f"{sysname}_p90_km"] = _find_metric(d, sh, P90_HINTS)
        s, pm = rec.get("sgp4_rmse_km"), rec.get("physics_ml_rmse_km")
        rec["improvement_pct"] = (100 * (s - pm) / s) if (s and pm) else None
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- run control
def run_stage(script, env):
    print(f"    -> {script}")
    r = subprocess.run([sys.executable] + script.split(), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {script}")


def backup_production():
    os.makedirs("model/_production_backup", exist_ok=True)
    for m in PRODUCTION_MODELS:
        if os.path.exists(m):
            shutil.copy2(m, os.path.join("model/_production_backup", os.path.basename(m)))
    print("backed up 1-day production models -> model/_production_backup/")


def restore_production():
    for m in PRODUCTION_MODELS:
        b = os.path.join("model/_production_backup", os.path.basename(m))
        if os.path.exists(b):
            shutil.copy2(b, m)
    print("restored 1-day production models.")


def eval_horizon(H, env_base):
    env = dict(env_base); env["ISDMAAS_HORIZON_S"] = str(H)
    label = HORIZON_LABELS.get(H, f"{H}s")
    print("-" * 64); print(f"HORIZON {label}  ({H} s)")
    for f in HORIZON_SPECIFIC:
        if os.path.exists(f):
            os.remove(f)
    for stage in stages_for(env):
        run_stage(stage, env)
    os.makedirs(HZ_DIR, exist_ok=True)
    dst = os.path.join(HZ_DIR, f"val_{H}.json")
    shutil.copy2(VALIDATION_JSON, dst)
    print(f"    saved {dst}")


def write_outputs(rows):
    os.makedirs("reports", exist_ok=True)
    json.dump({"generated": datetime.utcnow().isoformat(), "rows": rows},
              open("reports/multi_horizon_summary.json", "w"), indent=2)
    cols = ["horizon", "horizon_s", "sgp4_rmse_km", "pure_ml_rmse_km",
            "physics_ml_rmse_km", "sgp4_p90_km", "physics_ml_p90_km", "improvement_pct"]
    with open("reports/multi_horizon.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("\n" + "=" * 74)
    print("MULTI-HORIZON RESULTS  (RMSE position error, km)")
    print("=" * 74)
    print(f"{'horizon':>8} | {'SGP4':>9} | {'pure-ML':>9} | {'physics-ML':>11} | {'improve':>8}")
    print("-" * 74)
    for r in rows:
        def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "   n/a"
        imp = f"{r['improvement_pct']:+.1f}%" if r.get("improvement_pct") is not None else "   n/a"
        print(f"{r['horizon']:>8} | {fmt(r.get('sgp4_rmse_km')):>9} | "
              f"{fmt(r.get('pure_ml_rmse_km')):>9} | {fmt(r.get('physics_ml_rmse_km')):>11} | {imp:>8}")
    print("=" * 74)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        hrs = [r["horizon_s"]/3600 for r in rows]
        plt.figure(figsize=(8, 5))
        sg = [r.get("sgp4_rmse_km") for r in rows]
        pm = [r.get("physics_ml_rmse_km") for r in rows]
        if any(sg): plt.plot(hrs, sg, "o-", label="SGP4 baseline", color="#888")
        if any(pm): plt.plot(hrs, pm, "s-", label="physics-ML", color="#3d8bff")
        plt.xlabel("forecast horizon (hours)"); plt.ylabel("RMSE position error (km)")
        plt.title("ISDMAAS orbit prediction vs forecast horizon")
        plt.legend(); plt.grid(True, alpha=0.3); plt.xscale("log")
        plt.tight_layout(); plt.savefig("reports/multi_horizon.png", dpi=140)
        print("saved plot: reports/multi_horizon.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print("saved: reports/multi_horizon_summary.json + reports/multi_horizon.csv")


def main():
    if "--inspect" in sys.argv:
        if not os.path.exists(VALIDATION_JSON):
            sys.exit(f"{VALIDATION_JSON} not found — run validate_phase55.py first.")
        d = json.load(open(VALIDATION_JSON))
        print("=== your validation JSON ===")
        print(json.dumps(d, indent=2)[:2000])
        print("\n=== auto-discovered metrics ===")
        for s, sh in SYS_HINTS.items():
            print(f"  {s:12s} RMSE={_find_metric(d, sh, RMSE_HINTS)}  "
                  f"P90={_find_metric(d, sh, P90_HINTS)}")
        print("\nIf any are None, add the right substring to SYS/RMSE/P90 hints at top.")
        return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    horizons = [int(a) for a in args] if args else DEFAULT_HORIZONS

    if "--report-only" in sys.argv:
        write_outputs(build_summary(horizons)); return

    if not (os.path.exists("data/tle") or os.path.exists("data/precise_orbit")):
        print("WARNING: expected data/tle and data/precise_orbit. If your data is "
              "elsewhere, the pipeline stages may re-download. Continuing in 3s...")
    env = dict(os.environ)
    backup_production()
    try:
        for H in horizons:
            eval_horizon(H, env)
    finally:
        restore_production()
    write_outputs(build_summary(horizons))
    print("\nDone. Production 1-day models restored; per-horizon reports in reports/horizons/.")


if __name__ == "__main__":
    main()
