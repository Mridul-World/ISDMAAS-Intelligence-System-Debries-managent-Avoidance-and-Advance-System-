"""
================================================================================
LEAVE-ONE-SATELLITE-OUT (LOSO) EVALUATION
================================================================================
Proves the model GENERALIZES to a satellite it has never seen — not just
memorizes the 5 it trained on. For each satellite, train on the OTHER 4 and test
only on the held-out one.

REQUIRES the one-line holdout hook in export_phase55_tensors.py (see loso_patch.txt):
the env var ISDMAAS_HOLDOUT_NORAD makes that satellite contribute zero training
windows and become the entire validation set.

    python loso_eval.py                 # all 5 satellites held out in turn
    python loso_eval.py 39634 40697     # only these holdouts
    python loso_eval.py --report-only   # rebuild table from saved runs
    python loso_eval.py --inspect       # show what one val JSON looks like

For each holdout it runs: build -> export(with holdout) -> train -> validate, and
saves reports/loso/val_<norad>.json. Production models are backed up/restored.

HONEST READING
  - If held-out RMSE stays close to the all-satellites RMSE (~2.4 km at 1 day),
    the model GENERALIZES (learned transferable orbital-error structure).
  - If held-out RMSE degrades badly, the model was partly MEMORIZING per-satellite
    quirks — report that honestly; it bounds the generalization claim.
  - Either outcome is a legitimate, reportable result.
================================================================================
"""
import os, sys, json, shutil, subprocess
from datetime import datetime

SATELLITES = {39634: "SENTINEL-1A", 40697: "SENTINEL-2A", 42063: "SENTINEL-2B",
              41335: "SENTINEL-3A", 43437: "SENTINEL-3B"}
LOSO_DIR = "reports/loso"
VALIDATION_JSON = "reports/phase55_validation.json"
PRODUCTION_MODELS = ["model/rtn_physics_ml_phase55.pt", "model/rtn_pure_ml_phase55.pt"]
HORIZON_SPECIFIC = PRODUCTION_MODELS + [
    "X_train.pt", "X_val.pt", "y_train.pt", "y_val.pt",
    "baseline_train.pt", "baseline_val.pt", "residual_rtn_train.pt",
    "residual_rtn_val.pt", "sat_train.pt", "sat_val.pt", VALIDATION_JSON,
]
# we keep the prebuilt dataset CSV (holdout is applied at the tensor-split stage,
# so the dataset itself is identical — no need to rebuild it per holdout).
STAGES = ["export_phase55_tensors.py", "train_phase55_rtn.py", "validate_phase55.py"]

SYS_HINTS = {"sgp4": ["sgp4", "baseline"],
             "physics_ml": ["physics", "physics_ml", "residual"]}
RMSE_HINTS = ["rmse", "rmse_km", "pos_rmse", "position_rmse"]
MED_HINTS = ["median", "med_km", "p50"]


def _walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            np_ = f"{p}.{k}" if p else k
            if isinstance(v, (dict, list)):
                yield from _walk(v, np_)
            else:
                yield np_, k, v
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{p}[{i}]")


def _find(d, sh, mh):
    best = None
    for path, key, val in _walk(d):
        if isinstance(val, (int, float)):
            pl = path.lower()
            if any(s in pl for s in sh) and any(m in pl for m in mh):
                sc = ("overall" in pl) + ("all" in pl) + ("mean" not in pl)
                if best is None or sc > best[0]:
                    best = (sc, float(val))
    return best[1] if best else None


def _find_holdout_rmse(d, holdout):
    """Per-satellite RMSE for the held-out NORAD, from the MODEL (physics_ml)
       block specifically. In LOSO the held-out sat is the whole validation set,
       so the overall model RMSE equals this; we prefer the explicit per-sat
       value when present, scoped to a physics/model path."""
    key = str(holdout)
    for path, k, v in _walk(d):
        if isinstance(v, (int, float)) and k == key and "per_satellite" in path.lower():
            pl = path.lower()
            if any(s in pl for s in SYS_HINTS["physics_ml"]):
                return float(v)
    # fall back to overall model RMSE (correct in true LOSO)
    return _find(d, SYS_HINTS["physics_ml"], RMSE_HINTS)


def _find_holdout_sgp4(d, holdout):
    """SGP4 per-satellite RMSE for the held-out NORAD (scoped to sgp4 path)."""
    key = str(holdout)
    for path, k, v in _walk(d):
        if isinstance(v, (int, float)) and k == key and "per_satellite" in path.lower():
            pl = path.lower()
            if any(s in pl for s in SYS_HINTS["sgp4"]):
                return float(v)
    return _find(d, SYS_HINTS["sgp4"], RMSE_HINTS)


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
    print("backed up production models -> model/_production_backup/")


def restore_production():
    for m in PRODUCTION_MODELS:
        b = os.path.join("model/_production_backup", os.path.basename(m))
        if os.path.exists(b):
            shutil.copy2(b, m)
    print("restored production models.")


def eval_holdout(norad, env_base):
    env = dict(env_base); env["ISDMAAS_HOLDOUT_NORAD"] = str(norad)
    print("-" * 64); print(f"HOLD OUT {SATELLITES.get(norad, norad)}  (NORAD {norad})")
    for f in HORIZON_SPECIFIC:
        if os.path.exists(f):
            os.remove(f)
    for stage in STAGES:
        run_stage(stage, env)
    os.makedirs(LOSO_DIR, exist_ok=True)
    dst = os.path.join(LOSO_DIR, f"val_{norad}.json")
    shutil.copy2(VALIDATION_JSON, dst)
    print(f"    saved {dst}")


def build_summary(norads):
    rows = []
    for n in norads:
        p = os.path.join(LOSO_DIR, f"val_{n}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        sgp4 = _find_holdout_sgp4(d, n)
        held = _find_holdout_rmse(d, n)
        rows.append({"norad": n, "name": SATELLITES.get(n, str(n)),
                     "sgp4_rmse_km": sgp4, "model_rmse_km": held,
                     "improvement_pct": (100*(sgp4-held)/sgp4) if (sgp4 and held) else None})
    return rows


def write_outputs(rows):
    os.makedirs("reports", exist_ok=True)
    json.dump({"generated": datetime.utcnow().isoformat(), "rows": rows},
              open("reports/loso_summary.json", "w"), indent=2)
    with open("reports/loso.csv", "w") as f:
        f.write("norad,name,sgp4_rmse_km,model_rmse_km,improvement_pct\n")
        for r in rows:
            f.write(f"{r['norad']},{r['name']},{r['sgp4_rmse_km']},"
                    f"{r['model_rmse_km']},{r['improvement_pct']}\n")
    print("\n" + "=" * 72)
    print("LEAVE-ONE-SATELLITE-OUT RESULTS  (RMSE on the HELD-OUT satellite, km)")
    print("=" * 72)
    print(f"{'held-out satellite':<22} | {'SGP4':>8} | {'model':>8} | {'improve':>8}")
    print("-" * 72)
    vals = []
    for r in rows:
        def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "   n/a"
        imp = f"{r['improvement_pct']:+.1f}%" if r.get("improvement_pct") is not None else "   n/a"
        print(f"{r['name']:<22} | {fmt(r.get('sgp4_rmse_km')):>8} | "
              f"{fmt(r.get('model_rmse_km')):>8} | {imp:>8}")
        if isinstance(r.get("improvement_pct"), (int, float)):
            vals.append(r["improvement_pct"])
    print("-" * 72)
    if vals:
        print(f"{'MEAN (generalization)':<22} | {'':>8} | {'':>8} | {sum(vals)/len(vals):+7.1f}%")
    print("=" * 72)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = [r["name"].replace("SENTINEL-", "S") for r in rows]
        sg = [r.get("sgp4_rmse_km") for r in rows]
        md = [r.get("model_rmse_km") for r in rows]
        x = range(len(rows)); w = 0.38
        plt.figure(figsize=(8, 5))
        plt.bar([i-w/2 for i in x], sg, w, label="SGP4 (held-out sat)", color="#888")
        plt.bar([i+w/2 for i in x], md, w, label="Model trained on other 4", color="#2563eb")
        plt.xticks(list(x), names); plt.ylabel("RMSE position error (km)")
        plt.title("Leave-one-satellite-out: generalization to an unseen satellite")
        plt.legend(frameon=False); plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig("reports/loso.png", dpi=150)
        print("saved plot: reports/loso.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print("saved: reports/loso_summary.json + reports/loso.csv")


def main():
    if "--inspect" in sys.argv:
        import glob
        files = glob.glob(os.path.join(LOSO_DIR, "val_*.json"))
        if not files:
            sys.exit("no LOSO runs yet.")
        d = json.load(open(files[0]))
        print(json.dumps(d, indent=2)[:2000]); return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    norads = [int(a) for a in args] if args else list(SATELLITES.keys())

    if "--report-only" in sys.argv:
        write_outputs(build_summary(norads)); return

    if not os.path.exists("data/phase55_dataset.csv"):
        sys.exit("data/phase55_dataset.csv not found — run a normal build first.")
    # safety: confirm the holdout hook is present
    src = open("export_phase55_tensors.py").read()
    if "ISDMAAS_HOLDOUT_NORAD" not in src:
        sys.exit("Holdout hook missing in export_phase55_tensors.py. "
                 "Apply loso_patch.txt first (one-line change), then re-run.")

    env = dict(os.environ)
    backup_production()
    try:
        for n in norads:
            eval_holdout(n, env)
    finally:
        restore_production()
    write_outputs(build_summary(norads))
    print("\nDone. Production models restored; per-holdout reports in reports/loso/.")


if __name__ == "__main__":
    main()
