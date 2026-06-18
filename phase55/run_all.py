"""
Phase 5.5 — ONE-COMMAND PIPELINE with stage checkpointing.

    python run_all.py            # runs every stage, skipping completed ones
    python run_all.py --from train     # force re-run from a stage onward
                                       # stages: tle, poe, parse, build,
                                       #         export, train, validate

After a successful full run you have, permanently:
    data/phase55_dataset.csv     the real-truth dataset (never rebuild)
    *.pt + feature_scaler.pkl    training tensors
    model/rtn_physics_ml_phase55.pt   the trusted physics-informed model
    model/rtn_pure_ml_phase55.pt      ablation model (5.2 comparison)
    reports/phase55_validation.json   the publishable numbers
Then: python infer_future.py <NORAD> for operational predictions.
"""
import os, sys, subprocess

import glob

STAGES = [
    ("tle",      "download_tles.py",            ["data/tle/tle_*.csv"]),
    ("poe",      "download_precise_orbits.py",  ["data/precise_orbit/*/*.EOF"]),
    ("parse",    "parse_precise_orbits.py",     ["data/ephemeris_*.csv"]),
    ("build",    "build_truth_dataset.py",      ["data/phase55_dataset.csv"]),
    ("export",   "export_phase55_tensors.py",   ["X_train.pt", "X_val.pt"]),
    ("train",    "train_phase55_rtn.py",        ["model/rtn_physics_ml_phase55.pt"]),
    ("train_ablation", "train_phase55_rtn.py --pure-ml",
                                                ["model/rtn_pure_ml_phase55.pt"]),
    ("validate", "validate_phase55.py",         ["reports/phase55_validation.json"]),
]


def done(artifacts):
    return artifacts and all(glob.glob(a) for a in artifacts)

def main():
    start = None
    if "--from" in sys.argv:
        start = sys.argv[sys.argv.index("--from") + 1]
    forcing = start is None

    for name, cmd, artifacts in STAGES:
        if start and name == start:
            forcing = False
        skip = (start is not None and forcing) or (start is None and done(artifacts))
        if skip:
            print(f"[skip] {name}")
            continue
        print(f"\n{'='*64}\n[run ] {name}: python {cmd}\n{'='*64}")
        r = subprocess.run([sys.executable] + cmd.split())
        if r.returncode != 0:
            sys.exit(f"\nSTAGE FAILED: {name} — fix and rerun "
                     f"`python run_all.py --from {name}`")
    print("\nPHASE 5.5 COMPLETE. Next: python infer_future.py 39634")


if __name__ == "__main__":
    main()
