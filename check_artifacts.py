"""
ISDMAAS — artifact locator / verifier.
Run from the project root:  python check_artifacts.py
Prints exactly which models, datasets, and reports exist on THIS machine,
with sizes, and flags anything missing.
"""
import os, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def show(label, relpath, kind="file"):
    p = os.path.join(ROOT, relpath)
    if kind == "glob":
        hits = glob.glob(p)
        if hits:
            total = sum(os.path.getsize(h) for h in hits if os.path.isfile(h))
            print(f"  [OK]  {label:42s} {len(hits):>4} file(s)  {human(total)}")
            print(f"        {relpath}")
        else:
            print(f"  [--]  {label:42s} NONE FOUND")
            print(f"        {relpath}")
    else:
        if os.path.isfile(p):
            print(f"  [OK]  {label:42s} {human(os.path.getsize(p)):>10}")
            print(f"        {relpath}")
        else:
            print(f"  [--]  {label:42s} MISSING")
            print(f"        {relpath}")

print("=" * 72)
print(f"ISDMAAS artifact check    root = {ROOT}")
print("=" * 72)

print("\nMODELS")
show("Main model (physics-informed Transformer)", "phase55/model/rtn_physics_ml_phase55.pt")
show("Ablation model (pure-ML, no physics)",      "phase55/model/rtn_pure_ml_phase55.pt")
show("Residual normalization",                    "phase55/model/residual_norm_phase55.pt")
show("Feature scaler",                            "phase55/feature_scaler.pkl")
show("RL PPO policy (framework only)",            "phase9/model/phase9_ppo.zip")

print("\nFINAL DATASET + SOURCES")
show("FINAL dataset (all 5 satellites)",          "phase55/data/phase55_dataset.csv")
show("ESA POD precise-orbit truth (.EOF)",        "phase55/data/precise_orbit/**/*.EOF", "glob")
show("Downloaded TLEs",                           "phase55/data/tle/*.csv", "glob")
show("Space-weather drivers",                     "phase55/data/space_weather_daily.csv")
show("Active catalog for screening",              "phase11/data/catalog_active.tle")

print("\nTRAINING TENSORS")
for t in ["X_train.pt", "X_val.pt", "y_train.pt", "y_val.pt",
          "baseline_train.pt", "baseline_val.pt",
          "residual_rtn_train.pt", "residual_rtn_val.pt",
          "sat_train.pt", "sat_val.pt"]:
    show(t, f"phase55/{t}")

print("\nREPORTS (JSON results)")
show("HEADLINE orbit-prediction numbers",         "phase55/reports/phase55_validation.json")
show("Uncertainty calibration table",             "phase55/reports/phase6_validation.json")
show("Phase-6 calibrated states",                 "phase55/reports/state_*.json", "glob")
show("Conjunction CDM reports",                   "phase7/reports/conjunctions/*.json", "glob")
show("Validated-maneuver verdict",                "phase10/reports/validated_maneuver.json")
show("Historical replay results",                 "phase12/reports/replay_*.json", "glob")

print("\n" + "=" * 72)
print("Tip: [OK] = present on this machine, [--] = not generated yet.")
print("=" * 72)