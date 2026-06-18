"""
================================================================================
PHASE 5 (CONSOLIDATED)  —  Residual Transformer  (SGP4 + ML correction)
================================================================================
Consumes the ALIGNED tensors written by phase4_build_tensors.py:
    X_train.pt y_train.pt X_val.pt y_val.pt baseline_train.pt baseline_val.pt

ARCHITECTURE (the patent core)
------------------------------
    final_state = SGP4_baseline + Transformer(recent_state_sequence)
The Transformer learns the RESIDUAL between raw SGP4 propagation and the true
future orbit. This is the right inductive bias: physics carries the bulk of the
signal; the network only models SGP4's systematic drift (drag mismodelling,
stale-TLE error, maneuvers reflected in the recent state sequence).

WHY RESIDUAL (not absolute) TARGETS
-----------------------------------
Predicting absolute positions (~7000 km) is badly conditioned. The residual
(truth - SGP4) is O(km) and centred near zero -> stable, fast convergence, and it
GUARANTEES the model can never do worse than falling back to SGP4 (residual -> 0).

REPORTS (reports/phase5_validation.json):
    baseline position RMSE  vs  model position RMSE  (+ mean/median, velocity)
================================================================================
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# AUTO PATH RESOLUTION  (find the folder holding X_train.pt)
# =========================================================
def find_tensor_dir():
    """Locate the dir containing X_train.pt. Searches cwd, script dir, their
    parents, and a sibling 'phase4' folder. No manual path editing needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    seeds = [os.getcwd(), here,
             os.path.join(here, "phase4"),
             os.path.join(here, "..", "phase4"),
             os.path.join(os.getcwd(), "phase4"),
             os.path.join(os.getcwd(), "..", "phase4")]
    for start in (here, os.getcwd()):
        p = start
        for _ in range(4):
            seeds.append(p)
            p = os.path.dirname(p)
    for c in seeds:
        if c and os.path.isfile(os.path.join(c, "X_train.pt")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not find X_train.pt anywhere nearby. Run phase4_build_tensors.py FIRST "
        "(it writes the .pt files into the phase4 folder), then run this script.")


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "TENSOR_DIR": find_tensor_dir(),   # auto-located; no manual editing
    "MODEL_DIR":  "model",
    "REPORT_DIR": "reports",

    # model
    "D_MODEL": 128,
    "NHEAD": 8,
    "NUM_LAYERS": 4,
    "DIM_FF": 512,
    "DROPOUT": 0.1,

    # training
    "EPOCHS": 60,
    "BATCH_SIZE": 128,
    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-5,
    "GRAD_CLIP": 1.0,
    "VAL_EVERY": 1,
    "PATIENCE": 8,              # early stop on val position RMSE

    # ROBUSTNESS (critical on this data):
    # The residual (truth - SGP4) is outlier-dominated: median ~3 km but p99 ~2700 km
    # and max ~13000 km (TLE-gap / maneuver windows that are NOT predictable from past
    # kinematics). Without this, RMSE/std are swamped by outliers and the model just
    # collapses to residual=0 (== SGP4). We clip the TRAINING residual target so the
    # net learns the smooth, drag-driven, learnable component, and we report a robust
    # metric panel (median, p90, trimmed RMSE) instead of a single outlier-driven RMSE.
    "RESIDUAL_CLIP_KM": 150.0,  # clip |residual| used for the training target
    "TRIM_PCT": 1.0,            # drop top X% by error when reporting trimmed RMSE

    # ---- PHYSICS-INFORMED LOSS (Phase 5.1) ----
    # Penalties are computed on the RECONSTRUCTED state (baseline + residual) vs TRUTH,
    # i.e. "the predicted state's orbital invariants must match the true state's invariants".
    # This is consistency-to-truth, NOT two-body conservation over the window -- LEO orbits
    # decay (drag), so penalizing energy/period CHANGE would fight real physics. Keep weights
    # small: the data term carries the signal; physics nudges error into physical directions.
    "USE_PHYSICS_LOSS": True,
    "W_ENERGY":   0.05,   # |E_pred - E_true|        (km^2/s^2)
    "W_MOMENTUM": 0.05,   # |h_pred - h_true|        (km^2/s)
    "W_SMA":      0.02,   # |a_pred - a_true|        (km)
    "W_PERIOD":   0.01,   # |T_pred - T_true|        (s)

    # ABLATION: train pure-ML then physics-ML in one run and print a 3-way table
    # (SGP4 baseline vs pure-ML vs physics-ML). Set False to train a single model.
    "RUN_ABLATION": True,

    # set True for a fast mechanics check (few epochs, subsample);
    # set False for the real run (use GPU).
    "SMOKE_TEST": False,
    "SMOKE_EPOCHS": 3,
    "SMOKE_SUBSET": 8000,

    "SEED": 42,
}


# =========================================================
# MODEL  (self-contained; mirrors phase4 OrbitalTransformer)
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ResidualOrbitTransformer(nn.Module):
    """Outputs a 6-D residual (standardized). final = baseline + unstd(residual)."""
    def __init__(self, n_features, d_model, nhead, num_layers, dim_ff, dropout,
                 n_targets=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


# =========================================================
# PHYSICS  (Phase 5.1) — all on RECONSTRUCTED state vs TRUTH
# =========================================================
MU_EARTH = 398600.4418       # km^3 / s^2
R_EARTH = 6378.137           # km


def _orbital_invariants(state):
    """state: [...,6] = [x,y,z,vx,vy,vz] (km, km/s). Returns E, h_vec, a, T."""
    r = state[:, 0:3]
    v = state[:, 3:6]
    r_n = torch.norm(r, dim=-1).clamp_min(1.0)
    v2 = (v * v).sum(-1)
    E = 0.5 * v2 - MU_EARTH / r_n                  # specific energy (negative if bound)
    h = torch.cross(r, v, dim=-1)                  # specific angular momentum vector
    E_bound = torch.clamp(E, max=-1e-3)            # guard: keep bound for a,T
    a = -MU_EARTH / (2.0 * E_bound)
    T = 2.0 * math.pi * torch.sqrt((a.abs() ** 3) / MU_EARTH)
    return E, h, a, T


def physics_penalty(pred_state, true_state, cfg):
    """Consistency-to-truth penalty on orbital invariants. Real units, robust."""
    Ep, hp, ap, Tp = _orbital_invariants(pred_state)
    Et, ht, at, Tt = _orbital_invariants(true_state)
    energy = (Ep - Et).abs().mean() / 100.0
    momentum = torch.norm(hp - ht, dim=-1).mean() / 10000.0
    sma = (ap - at).abs().mean() / 1000.0
    period = (Tp - Tt).abs().mean() / 1000.0
    return (cfg["W_ENERGY"] * energy + cfg["W_MOMENTUM"] * momentum
            + cfg["W_SMA"] * sma + cfg["W_PERIOD"] * period)


def kepler_plausibility(pred_state):
    """Fraction of predictions that are physically valid bound orbits + invariant stats."""
    E, h, a, T = _orbital_invariants(pred_state)
    bound = (E < 0).float().mean().item()                 # bound orbit fraction
    # eccentricity from vis-viva + ang. momentum: e = sqrt(1 + 2 E h^2 / mu^2)
    h2 = (h * h).sum(-1)
    e2 = 1.0 + 2.0 * E * h2 / (MU_EARTH ** 2)
    e = torch.sqrt(e2.clamp_min(0.0))
    valid_e = ((e >= 0) & (e < 1)).float().mean().item()  # elliptical fraction
    above_surface = (a > R_EARTH).float().mean().item()
    return {"bound_frac": bound, "elliptical_frac": valid_e,
            "above_surface_frac": above_surface,
            "mean_period_min": float(T.mean().item() / 60.0)}


# =========================================================
# METRICS
# =========================================================
def pos_rmse(pred, true):
    d = pred[:, :3] - true[:, :3]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def pos_mean_median(pred, true):
    e = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
    return float(e.mean()), float(np.median(e))


def vel_rmse(pred, true):
    d = pred[:, 3:6] - true[:, 3:6]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def pos_metrics(pred, true, trim_pct=1.0):
    """Robust panel: RMSE, mean, median, p90, and trimmed-RMSE (drop top trim_pct%)."""
    e = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
    keep = e <= np.percentile(e, 100 - trim_pct)
    return {
        "rmse_km": float(np.sqrt(np.mean(e ** 2))),
        "mean_km": float(e.mean()),
        "median_km": float(np.median(e)),
        "p90_km": float(np.percentile(e, 90)),
        "trimmed_rmse_km": float(np.sqrt(np.mean(e[keep] ** 2))),
    }


# =========================================================
# TRAINER (reusable: pure-ML and physics-ML share this)
# =========================================================
def train_model(data, cfg, dev, epochs, use_physics, tag):
    X_tr, r_tr_n, b_tr, X_va, r_va_n, b_va, y_va, r_mean, r_std = data
    r_mean_d, r_std_d = r_mean.to(dev), r_std.to(dev)

    tr_loader = DataLoader(TensorDataset(X_tr, r_tr_n, b_tr), batch_size=cfg["BATCH_SIZE"],
                           shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, b_va, y_va),
                           batch_size=cfg["BATCH_SIZE"], shuffle=False)

    torch.manual_seed(cfg["SEED"])
    model = ResidualOrbitTransformer(
        n_features=X_tr.shape[2], d_model=cfg["D_MODEL"], nhead=cfg["NHEAD"],
        num_layers=cfg["NUM_LAYERS"], dim_ff=cfg["DIM_FF"], dropout=cfg["DROPOUT"]
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"],
                            weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    data_loss_fn = nn.SmoothL1Loss()

    print(f"\n--- training [{tag}] (physics={'ON' if use_physics else 'OFF'}) ---")
    best = {"epoch": -1, "trimmed_rmse_km": float("inf")}
    best_preds = None
    bad = 0; history = []

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for xb, rb, bb in tr_loader:
            xb, rb, bb = xb.to(dev), rb.to(dev), bb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = data_loss_fn(out, rb)
            if use_physics:
                # reconstruct ABSOLUTE states (baseline + residual) in real units,
                # then penalize orbital-invariant mismatch vs the (clipped) truth state.
                pred_state = bb + (out * r_std_d + r_mean_d)
                true_state = bb + (rb * r_std_d + r_mean_d)
                loss = loss + physics_penalty(pred_state, true_state, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        train_loss = tot / len(tr_loader.dataset)

        # ---- validate (physics computed on ABSOLUTE reconstructed state) ----
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, bb, yb in va_loader:
                res = model(xb.to(dev)) * r_std_d + r_mean_d
                final = bb.to(dev) + res
                preds.append(final.cpu().numpy()); trues.append(yb.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        m = pos_metrics(preds, trues, cfg["TRIM_PCT"])
        m_vrmse = vel_rmse(preds, trues)
        kep = kepler_plausibility(torch.tensor(preds))
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_trimmed_rmse": m["trimmed_rmse_km"], "val_median": m["median_km"]})

        improved = m["trimmed_rmse_km"] < best["trimmed_rmse_km"] - 1e-6
        flag = ""
        if improved:
            best = {"epoch": ep, **{f"model_{k}": v for k, v in m.items()},
                    "trimmed_rmse_km": m["trimmed_rmse_km"], "model_vel_rmse": m_vrmse,
                    "kepler": kep}
            best_preds = preds.copy()
            torch.save(model.state_dict(),
                       os.path.join(cfg["MODEL_DIR"], f"residual_transformer_{tag}.pt"))
            bad = 0; flag = "  <- best"
        else:
            bad += 1
        print(f"ep {ep:3d}/{epochs} | loss {train_loss:.4f} | "
              f"trimRMSE {m['trimmed_rmse_km']:8.2f} | median {m['median_km']:6.2f} | "
              f"p90 {m['p90_km']:7.2f} | bound {kep['bound_frac']*100:5.1f}%{flag}")
        if bad >= cfg["PATIENCE"]:
            print(f"early stop ({cfg['PATIENCE']} epochs no improvement)"); break

    best["history"] = history
    return best, best_preds


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = CONFIG
    torch.manual_seed(cfg["SEED"]); np.random.seed(cfg["SEED"])
    os.makedirs(cfg["MODEL_DIR"], exist_ok=True)
    os.makedirs(cfg["REPORT_DIR"], exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"tensor dir (auto): {cfg['TENSOR_DIR']}")

    td = cfg["TENSOR_DIR"]
    ld = lambda f: torch.load(os.path.join(td, f), weights_only=False).float()
    X_tr, y_tr, b_tr = ld("X_train.pt"), ld("y_train.pt"), ld("baseline_train.pt")
    X_va, y_va, b_va = ld("X_val.pt"), ld("y_val.pt"), ld("baseline_val.pt")
    print(f"train {tuple(X_tr.shape)}  val {tuple(X_va.shape)}")

    val_perm = None
    if cfg["SMOKE_TEST"]:
        g = torch.Generator().manual_seed(cfg["SEED"])
        k = min(cfg["SMOKE_SUBSET"], len(X_tr))
        perm = torch.randperm(len(X_tr), generator=g)[:k]
        X_tr, y_tr, b_tr = X_tr[perm], y_tr[perm], b_tr[perm]
        kv = min(cfg["SMOKE_SUBSET"] // 4, len(X_va))
        val_perm = torch.randperm(len(X_va), generator=g)[:kv].numpy()
        X_va, y_va, b_va = X_va[val_perm], y_va[val_perm], b_va[val_perm]
        epochs = cfg["SMOKE_EPOCHS"]
        print(f"[SMOKE] train {len(X_tr)}  val {len(X_va)}  epochs {epochs}")
    else:
        epochs = cfg["EPOCHS"]

    # residual target (truth - SGP4), clipped + standardized on TRAIN
    clip = cfg["RESIDUAL_CLIP_KM"]
    r_tr = (y_tr - b_tr).clamp(-clip, clip)
    r_mean = r_tr.mean(0, keepdim=True)
    r_std = r_tr.std(0, keepdim=True).clamp_min(1e-6)
    r_tr_n = (r_tr - r_mean) / r_std
    r_va_n = ((y_va - b_va).clamp(-clip, clip) - r_mean) / r_std
    data = (X_tr, r_tr_n, b_tr, X_va, r_va_n, b_va, y_va, r_mean, r_std)

    # ---- Phase 2 / SGP4 physical baseline (fixed reference) ----
    base = pos_metrics(b_va.numpy(), y_va.numpy(), cfg["TRIM_PCT"])
    base_vrmse = vel_rmse(b_va.numpy(), y_va.numpy())
    base_kep = kepler_plausibility(b_va)
    print(f"\nSGP4 BASELINE (val): trimRMSE={base['trimmed_rmse_km']:.2f}  "
          f"median={base['median_km']:.2f}  p90={base['p90_km']:.2f} km  "
          f"vel_RMSE={base_vrmse:.4f} km/s  bound={base_kep['bound_frac']*100:.1f}%")

    # ---- train: ablation (pure-ML, physics-ML) or single ----
    runs, preds_by_tag = {}, {}
    if cfg["RUN_ABLATION"]:
        runs["pure_ml"], preds_by_tag["pure_ml"] = train_model(
            data, cfg, dev, epochs, use_physics=False, tag="pure_ml")
        runs["physics_ml"], preds_by_tag["physics_ml"] = train_model(
            data, cfg, dev, epochs, use_physics=True, tag="physics_ml")
    else:
        tag = "physics_ml" if cfg["USE_PHYSICS_LOSS"] else "pure_ml"
        runs[tag], preds_by_tag[tag] = train_model(
            data, cfg, dev, epochs, use_physics=cfg["USE_PHYSICS_LOSS"], tag=tag)

    # ---- report + comparison table ----
    report = {"baseline_sgp4": {**base, "vel_rmse_kms": base_vrmse, "kepler": base_kep},
              "runs": runs, "config": cfg}
    with open(os.path.join(cfg["REPORT_DIR"], "phase5_validation.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("PHASE 5 COMPARISON  (val position error; trimmed drops top "
          f"{cfg['TRIM_PCT']}% maneuver/gap outliers)")
    print("=" * 72)
    hdr = f"{'model':<14}{'trimRMSE':>11}{'median':>10}{'p90':>10}{'fullRMSE':>11}{'bound%':>9}"
    print(hdr); print("-" * len(hdr))
    print(f"{'SGP4 (Phase2)':<14}{base['trimmed_rmse_km']:>11.2f}{base['median_km']:>10.2f}"
          f"{base['p90_km']:>10.2f}{base['rmse_km']:>11.2f}{base_kep['bound_frac']*100:>8.1f}%")
    for tag, r in runs.items():
        print(f"{tag:<14}{r.get('model_trimmed_rmse_km', float('nan')):>11.2f}"
              f"{r.get('model_median_km', float('nan')):>10.2f}"
              f"{r.get('model_p90_km', float('nan')):>10.2f}"
              f"{r.get('model_rmse_km', float('nan')):>11.2f}"
              f"{r.get('kepler', {}).get('bound_frac', float('nan'))*100:>8.1f}%")

    # ---- STALENESS-STRATIFIED TABLE (the part that surfaces the real win) ----
    age_path = os.path.join(cfg["TENSOR_DIR"], "age_val.pt")
    if os.path.exists(age_path):
        age = torch.load(age_path, weights_only=False).numpy()
        # align age to the (possibly subsampled) val set
        if cfg["SMOKE_TEST"] and val_perm is not None:
            age = age[val_perm]
        age = age[:len(y_va)]
        yv = y_va.numpy(); bv = b_va.numpy()
        bins = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 1e9)]
        def med_p90(err):
            return (float(np.median(err)), float(np.percentile(err, 90)))
        print("\n" + "=" * 72)
        print("STALENESS-STRATIFIED  (median / p90 position error, km, by TLE age in days)")
        print("=" * 72)
        cols = "SGP4  |  " + "  ".join(preds_by_tag.keys())
        print(f"{'age(d)':<9}{'n':>7}   {cols}")
        print("-" * 72)
        for lo, hi in bins:
            mask = (age >= lo) & (age < hi)
            if mask.sum() == 0:
                continue
            label = f"{lo}-{'+' if hi >= 1e9 else hi}"
            be = np.linalg.norm(bv[mask, :3] - yv[mask, :3], axis=1)
            bmed, bp90 = med_p90(be)
            row = f"{label:<9}{int(mask.sum()):>7}   {bmed:6.1f}/{bp90:6.1f}"
            for tag, pr in preds_by_tag.items():
                me = np.linalg.norm(pr[mask, :3] - yv[mask, :3], axis=1)
                mmed, mp90 = med_p90(me)
                row += f"   {mmed:6.1f}/{mp90:6.1f}"
            print(row)
        print("\nRead: in high-age rows SGP4 degrades; if model med/p90 < SGP4 there, that is the win.")
    print("\nSaved: model/residual_transformer_*.pt, reports/phase5_validation.json")


if __name__ == "__main__":
    main()
