"""
================================================================================
PHASE 6 — UNCERTAINTY-AWARE RESIDUAL MODEL + CONFIDENCE GATING
================================================================================
Encoder identical to Phase 5.5 (warm-started from rtn_physics_ml_phase55.pt).
Head: 3 -> 6 = (mu_R, mu_T, mu_N, log_sigma_R, log_sigma_T, log_sigma_N).
Loss: Gaussian NLL on NORMALIZED RTN residuals (same normalization as 5.5).

Fixes the two Phase-5.5 defects:
  (A) median regression / worse_than_sgp4 -> per-channel CONFIDENCE GATING.
  (B) produces the per-prediction RTN covariance Phase 7 needs.

Post-hoc TEMPERATURE SCALING calibrates sigma so 1-sigma coverage ~= 0.68 per
channel (NLL training alone tends to over-inflate radial/cross-track sigma).
The per-channel scale factors are saved for inference/Phase 7.

Outputs:
  model/rtn_uncertainty_phase6.pt        trained weights
  model/residual_norm_phase6.pt          residual mean/std
  model/sigma_temperature_phase6.pt      per-channel sigma calibration factors
  reports/phase6_validation.json         gating table + calibration
  reports/phase6_calibration.png         reliability plot (if matplotlib present)

Run:
  python train_phase6_uncertainty.py            train + evaluate (aleatoric sigma)
  python train_phase6_uncertainty.py --mc 20    add MC-dropout epistemic sigma
  python train_phase6_uncertainty.py --eval-only --mc 20   skip training
================================================================================
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from train_phase55_rtn import PosEnc, rtn_to_eci
import warnings; warnings.filterwarnings("ignore")

CFG = {"D_MODEL": 64, "NHEAD": 4, "NUM_LAYERS": 2, "DIM_FF": 128, "DROPOUT": 0.1,
       "EPOCHS": 40, "BATCH": 512, "LR": 2e-4, "WD": 1e-4, "PATIENCE": 8,
       "SEED": 42, "WARM_START": "model/rtn_physics_ml_phase55.pt"}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # signal-to-noise gating thresholds


class UncertaintyRTN(nn.Module):
    """Same trunk as Phase 5.5; 6-output head (mu + log_sigma per RTN channel)."""
    def __init__(s, nf, d, nh, nl, ff, dr):
        super().__init__()
        s.ip = nn.Sequential(nn.Linear(nf, d), nn.LayerNorm(d), nn.GELU())
        s.pe = PosEnc(d, dr=dr)
        e = nn.TransformerEncoderLayer(d, nh, ff, dr, batch_first=True,
                                       norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(e, nl, enable_nested_tensor=False)
        s.hd = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
                             nn.Linear(d // 2, 6))
        [nn.init.xavier_uniform_(p) for p in s.parameters() if p.dim() > 1]

    def forward(s, x):
        out = s.hd(s.enc(s.pe(s.ip(x)))[:, -1, :])
        mu = out[:, :3]
        log_sigma = out[:, 3:].clamp(-6.0, 4.0)
        return mu, log_sigma


def nll(mu, log_sigma, target):
    inv_var = torch.exp(-2 * log_sigma)
    return (0.5 * inv_var * (target - mu) ** 2 + log_sigma).mean()


def metrics(err):
    return {"rmse": float(np.sqrt((err ** 2).mean())), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "p99": float(np.percentile(err, 99)),
            "mean": float(err.mean())}


def save_calibration_plot(z_abs_by_channel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    levels = np.linspace(0.1, 3.0, 30)
    plt.figure(figsize=(5, 5))
    for ch, z in z_abs_by_channel.items():
        emp = [float((z < L).mean()) for L in levels]
        from math import erf, sqrt
        theo = [erf(L / sqrt(2)) for L in levels]
        plt.plot(theo, emp, marker=".", label=f"{ch}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.xlabel("expected coverage"); plt.ylabel("empirical coverage")
    plt.title("Phase 6 calibration (RTN, after scaling)"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/phase6_calibration.png", dpi=120)
    print("saved: reports/phase6_calibration.png")


def main():
    c = CFG
    torch.manual_seed(c["SEED"]); np.random.seed(c["SEED"])
    mc = int(sys.argv[sys.argv.index("--mc") + 1]) if "--mc" in sys.argv else 0
    eval_only = "--eval-only" in sys.argv

    ld = lambda f: torch.load(f, weights_only=False).float()
    Xtr, rtr = ld("X_train.pt"), ld("residual_rtn_train.pt")
    Xva, bva, yva, rva = (ld("X_val.pt"), ld("baseline_val.pt"),
                          ld("y_val.pt"), ld("residual_rtn_val.pt"))

    rm = rtr.mean(0, keepdim=True); rs = rtr.std(0, keepdim=True).clamp_min(1e-6)
    rtr_n, rva_n = (rtr - rm) / rs, (rva - rm) / rs
    os.makedirs("model", exist_ok=True); os.makedirs("reports", exist_ok=True)
    torch.save({"rm": rm, "rs": rs}, "model/residual_norm_phase6.pt")

    net = UncertaintyRTN(Xtr.shape[2], c["D_MODEL"], c["NHEAD"], c["NUM_LAYERS"],
                         c["DIM_FF"], c["DROPOUT"])

    if not eval_only:
        if os.path.exists(c["WARM_START"]):
            sd = torch.load(c["WARM_START"], map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith("hd.")}
            net.load_state_dict(sd, strict=False)
            print(f"warm-started encoder from {c['WARM_START']}")
        else:
            print("no warm-start file; training from scratch")

        opt = torch.optim.AdamW(net.parameters(), lr=c["LR"], weight_decay=c["WD"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["EPOCHS"])
        loader = DataLoader(TensorDataset(Xtr, rtr_n), batch_size=c["BATCH"], shuffle=True)
        best = {"nll": 1e9}; bad = 0

        def val_nll():
            net.eval()
            with torch.no_grad():
                tot, n = 0.0, 0
                for j in range(0, len(Xva), 1024):
                    mu, lsg = net(Xva[j:j + 1024])
                    tot += float(nll(mu, lsg, rva_n[j:j + 1024])) * len(mu); n += len(mu)
            return tot / n

        for ep in range(1, c["EPOCHS"] + 1):
            net.train()
            for xb, rb in loader:
                opt.zero_grad(); mu, lsg = net(xb)
                loss = nll(mu, lsg, rb)
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            sch.step(); v = val_nll(); fl = ""
            if v < best["nll"] - 1e-5:
                best = {"nll": v, "epoch": ep}; bad = 0; fl = "  <-best"
                torch.save(net.state_dict(), "model/rtn_uncertainty_phase6.pt")
            else:
                bad += 1
            print(f"ep{ep:3d}  val_NLL {v:9.4f}{fl}")
            if bad >= c["PATIENCE"]:
                print("early stop"); break

    net.load_state_dict(torch.load("model/rtn_uncertainty_phase6.pt", map_location="cpu"))

    # ---------------- predict mu, sigma on val ----------------
    net.eval()
    with torch.no_grad():
        mus, lsgs = [], []
        for j in range(0, len(Xva), 1024):
            mu, lsg = net(Xva[j:j + 1024]); mus.append(mu); lsgs.append(lsg)
        mu_n = torch.cat(mus); sig_n = torch.exp(torch.cat(lsgs))    # aleatoric (normalized)

    if mc:                                                  # MC-dropout epistemic variance
        net.train()
        with torch.no_grad():
            samples = []
            for _ in range(mc):
                ms = [net(Xva[j:j + 1024])[0] for j in range(0, len(Xva), 1024)]
                samples.append(torch.cat(ms))
            ep_var = torch.stack(samples).var(0)
        sig_n = torch.sqrt(sig_n ** 2 + ep_var)
        net.eval()
        print(f"MC-dropout: {mc} passes, epistemic variance folded into sigma")

    mu_km = mu_n * rs + rm                                   # de-normalize to km
    sig_km = sig_n * rs

    # ---------------- raw calibration (before scaling) ----------------
    z_raw = ((rva - mu_km) / sig_km).numpy()
    raw_1s = {ch: float((np.abs(z_raw[:, i]) < 1).mean())
              for i, ch in enumerate(["R", "T", "N"])}

    # ---------------- post-hoc temperature scaling ----------------
    # Find per-channel scale s such that fraction(|z|/s < 1) ~= 0.6827, i.e.
    # s = 0.6827-quantile of |z|. Dividing sigma by nothing / multiplying by s
    # makes 1-sigma coverage hit the target. No retraining.
    temp = {}
    for i, ch in enumerate(["R", "T", "N"]):
        s = float(np.quantile(np.abs(z_raw[:, i]), 0.6827))
        temp[ch] = max(s, 1e-3)
    temp_vec = torch.tensor([[temp["R"], temp["T"], temp["N"]]], dtype=sig_km.dtype)
    sig_km = sig_km * temp_vec
    sig_n = sig_n * temp_vec
    torch.save({"temp": temp_vec}, "model/sigma_temperature_phase6.pt")
    print("temperature scaling factors:", {k: round(v, 3) for k, v in temp.items()})

    # ---------------- calibration after scaling ----------------
    z = ((rva - mu_km) / sig_km).numpy()
    z_abs = {ch: np.abs(z[:, i]) for i, ch in enumerate(["R", "T", "N"])}
    calib_1s = {ch: float((z_abs[ch] < 1).mean()) for ch in z_abs}     # target ~0.68
    calib_2s = {ch: float((z_abs[ch] < 2).mean()) for ch in z_abs}     # target ~0.95
    save_calibration_plot(z_abs)

    # ---------------- gating sweep ----------------
    # Per-channel SNR gate: a correction channel is trusted only when its
    # predicted magnitude exceeds alpha * its (calibrated) sigma. Channels that
    # fail are zeroed, so the gate is graded, not all-or-nothing.
    sgp4_err = np.linalg.norm(bva[:, :3].numpy() - yva[:, :3].numpy(), axis=1)
    base = metrics(sgp4_err)
    snr = (mu_km.abs() / sig_km.clamp_min(1e-9))           # (N,3) per-channel SNR

    rep = {"truth_source": "REAL ESA POD precise ephemeris (AUX_POEORB), TEME",
           "n_val": int(len(Xva)), "mc_dropout_passes": mc,
           "sgp4": base,
           "temperature_factors": {k: temp[k] for k in temp},
           "calibration_within_1sigma_raw": raw_1s,
           "calibration_within_1sigma": calib_1s,
           "calibration_within_2sigma": calib_2s,
           "gating": {}}

    print("\nSGP4 baseline:", {k: round(v, 3) for k, v in base.items()})
    print(f"calibration 1σ RAW   : {{R:{raw_1s['R']:.2f} T:{raw_1s['T']:.2f} N:{raw_1s['N']:.2f}}}")
    print(f"calibration 1σ SCALED: {{R:{calib_1s['R']:.2f} T:{calib_1s['T']:.2f} N:{calib_1s['N']:.2f}}}"
          f"   2σ: {{R:{calib_2s['R']:.2f} T:{calib_2s['T']:.2f} N:{calib_2s['N']:.2f}}}")
    print(f"\n{'alpha':>6}{'applied%':>10}{'RMSE':>9}{'median':>9}{'P90':>9}{'P99':>9}{'worse%':>9}")
    for a in ALPHAS:
        ch_gate = (snr > a).float()                        # (N,3) keep trusted channels
        mu_gated = mu_km * ch_gate
        delta = rtn_to_eci(mu_gated, bva)                  # ECI correction (gated channels)
        final = bva.clone()
        final[:, :3] = bva[:, :3] - delta
        err = np.linalg.norm(final[:, :3].numpy() - yva[:, :3].numpy(), axis=1)
        m = metrics(err); worse = float((err > sgp4_err).mean())
        applied = float((ch_gate.sum(1) > 0).float().mean())
        rep["gating"][f"{a}"] = {**m, "worse_than_sgp4_fraction": worse,
                                 "applied_fraction": applied,
                                 "mean_channels_applied": float(ch_gate.sum(1).mean())}
        print(f"{a:6.1f}{100*applied:10.1f}{m['rmse']:9.3f}"
              f"{m['median']:9.3f}{m['p90']:9.3f}{m['p99']:9.3f}{100*worse:9.1f}")

    # recommend alpha: keep median <= SGP4, minimize RMSE; fall back to min worse%
    ok = [(a, d) for a, d in rep["gating"].items()
          if d["median"] <= base["median"] + 1e-6]
    rec = min(ok, key=lambda kv: kv[1]["rmse"]) if ok else \
          min(rep["gating"].items(), key=lambda kv: kv[1]["worse_than_sgp4_fraction"])
    rep["recommended_alpha"] = rec[0]
    print(f"\nrecommended alpha = {rec[0]}  -> {{rmse:{rec[1]['rmse']:.3f}, "
          f"median:{rec[1]['median']:.3f}, p90:{rec[1]['p90']:.3f}, "
          f"worse%:{100*rec[1]['worse_than_sgp4_fraction']:.1f}}}")

    json.dump(rep, open("reports/phase6_validation.json", "w"), indent=2)
    print("\nSaved: model/rtn_uncertainty_phase6.pt, model/residual_norm_phase6.pt, "
          "model/sigma_temperature_phase6.pt, reports/phase6_validation.json")
    print("PASS: a row with median<=SGP4, P90<<SGP4, worse%<15, calibration 1σ in ~0.55-0.80")


if __name__ == "__main__":
    main()