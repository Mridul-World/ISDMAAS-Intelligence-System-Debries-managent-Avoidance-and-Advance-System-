"""
evaluate_ensemble.py — honest re-evaluation of the trained residual-correction
models, plus two post-hoc accuracy improvements that need no retraining.

WHY THIS EXISTS
---------------
The trainer selects its checkpoint by lowest validation RMSE and then reports
that same validation RMSE as the result. That number is an optimistic estimate:
it is the minimum of ~40 noisy draws, so it is biased low by construction. This
script splits the validation set CHRONOLOGICALLY in half and never lets anything
fitted on the first half be evaluated on itself:

    calibration half   fit the shrinkage factor, choose the ensemble weight
    test half          the only numbers reported

TWO IMPROVEMENTS, BOTH VERIFIABLE HERE
--------------------------------------
1. ENSEMBLE. Two checkpoints already exist — the physics-loss variant and the
   pure-ML ablation. They were trained from the same seed and data but different
   objectives, so their errors are not perfectly correlated. Averaging two
   unbiased predictors with correlation rho reduces error variance by
   (1 + rho) / 2, which is a real reduction for any rho < 1 and costs one extra
   forward pass. The blend weight is fitted on the calibration half.

2. SHRINKAGE. A regressor trained on a noisy target over-states the magnitude of
   what it predicts: the MSE-optimal rescaling of any prediction p towards the
   truth t is alpha = <p,t> / <p,p>, and for a noisy predictor that is less than
   one. Applying the alpha fitted on the calibration half is the single cheapest
   accuracy gain available on an already-trained model.

Both are reported against the SGP4 baseline on the test half so the comparison is
like for like. If either fails to help, this script says so — the point is to
measure, not to advertise.

Run:  python evaluate_ensemble.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
REPORT_DIR = HERE / "reports"

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT = 64, 4, 2, 128, 0.1


# --------------------------------------------------------------- model shell
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class RTNTransformer(nn.Module):
    """The architecture the checkpoints were trained with. Do not change it here."""

    def __init__(self, n_features, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        self.ip = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.hd = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 3))

    def forward(self, x):
        return self.hd(self.enc(self.pe(self.ip(x)))[:, -1, :])


# ----------------------------------------------------------------- geometry
def rtn_basis(state):
    r, v = state[:, :3], state[:, 3:]
    R = r / torch.norm(r, dim=-1, keepdim=True)
    W = torch.cross(r, v, dim=-1)
    W = W / torch.norm(W, dim=-1, keepdim=True)
    S = torch.cross(W, R, dim=-1)
    return R, S, W


def rtn_to_eci(rtn, reference_state):
    R, S, W = rtn_basis(reference_state)
    return rtn[:, 0:1] * R + rtn[:, 1:2] * S + rtn[:, 2:3] * W


def reconstruct(baseline, residual_rtn):
    """
    truth_pos = baseline_pos - RTN_to_ECI(residual_rtn, baseline)

    The sign is the project's frozen convention: the learned residual is
    RTN(baseline - truth), so it is SUBTRACTED from the baseline.
    """
    out = baseline.clone()
    out[:, :3] = baseline[:, :3] - rtn_to_eci(residual_rtn, baseline)
    return out


def error_panel(predicted, truth):
    e = np.linalg.norm(predicted[:, :3] - truth[:, :3], axis=1)
    return {
        "rmse_km": float(np.sqrt((e ** 2).mean())),
        "median_km": float(np.median(e)),
        "p90_km": float(np.percentile(e, 90)),
        "p99_km": float(np.percentile(e, 99)),
        "mean_km": float(e.mean()),
    }


def improvement(baseline_panel, model_panel):
    return {
        key.replace("_km", "_pct"): 100.0 * (1.0 - model_panel[key] / baseline_panel[key])
        for key in ("rmse_km", "median_km", "p90_km")
    }


# ------------------------------------------------------------------ predict
@torch.no_grad()
def predict_residual(model, features, mean, scale, batch=1024):
    model.eval()
    chunks = [model(features[i:i + batch]) * scale + mean
              for i in range(0, len(features), batch)]
    return torch.cat(chunks, 0)


def load_checkpoint(path, n_features):
    model = RTNTransformer(n_features)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model


# ------------------------------------------------------------ calibration
def fit_shrinkage(predicted_rtn, target_rtn):
    """
    MSE-optimal scalar rescaling: alpha = <p, t> / <p, p>.

    Fitted per RTN axis, because the along-track residual is an order of
    magnitude larger than radial or cross-track and a single shared factor would
    be dominated by it.
    """
    numerator = (predicted_rtn * target_rtn).sum(axis=0)
    denominator = (predicted_rtn * predicted_rtn).sum(axis=0)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 1.0)


def fit_blend_weight(a, b, target, grid=np.linspace(0.0, 1.0, 101)):
    """Weight w minimizing ||w*a + (1-w)*b - target||^2 by a dense scan."""
    errors = [float(((w * a + (1 - w) * b - target) ** 2).sum()) for w in grid]
    return float(grid[int(np.argmin(errors))])


def main():
    torch.manual_seed(0)
    load = lambda name: torch.load(HERE / name, weights_only=False).float()

    try:
        X = load("X_val.pt")
        y = load("y_val.pt")
        baseline = load("baseline_val.pt")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Validation tensors are missing ({exc.filename}). Rebuild them with "
            "export_phase55_tensors.py before running this evaluation."
        )

    norm = torch.load(MODEL_DIR / "residual_norm_phase55.pt",
                      map_location="cpu", weights_only=False)
    mean, scale = norm["rm"].float(), norm["rs"].float()

    # The tensors are stored in chronological order per satellite, so a straight
    # split in half keeps calibration strictly before test within each satellite's
    # block and never fits on data it is later scored against.
    n = len(X)
    half = n // 2
    calibration = slice(0, half)
    test = slice(half, n)

    checkpoints = {
        "physics_ml": MODEL_DIR / "rtn_physics_ml_phase55.pt",
        "pure_ml": MODEL_DIR / "rtn_pure_ml_phase55.pt",
    }
    available = {name: path for name, path in checkpoints.items() if path.exists()}
    if not available:
        raise SystemExit(f"No trained checkpoints found in {MODEL_DIR}")

    residuals = {}
    for name, path in available.items():
        model = load_checkpoint(path, X.shape[2])
        residuals[name] = predict_residual(model, X, mean, scale).numpy()

    truth = y.numpy()
    baseline_np = baseline.numpy()
    target_rtn = None
    try:
        target_rtn = load("residual_rtn_val.pt").numpy()
    except FileNotFoundError:
        pass

    def score(residual_rtn, index):
        predicted = reconstruct(
            baseline[index], torch.from_numpy(residual_rtn[index]).float()).numpy()
        return error_panel(predicted, truth[index])

    baseline_test = error_panel(baseline_np[test], truth[test])
    report = {
        "note": (
            "Every figure below is measured on the LAST half of the validation "
            "set. The blend weight and shrinkage factors were fitted on the first "
            "half only, so no reported number was tuned on the data it scores."
        ),
        "n_calibration": half,
        "n_test": n - half,
        "sgp4_baseline_test": baseline_test,
        "models": {},
    }

    print("=" * 74)
    print("ISDMAAS residual-correction model — held-out evaluation")
    print("=" * 74)
    print(f"validation windows: {n}  (calibration {half} / test {n - half})")
    print(f"\nSGP4 baseline on the test half: "
          f"rmse {baseline_test['rmse_km']:.4f}  median {baseline_test['median_km']:.4f} km")
    print(f"\n{'variant':<28}{'rmse km':>10}{'median km':>12}{'p90 km':>10}{'vs SGP4':>10}")
    print("-" * 74)

    def emit(label, residual_rtn):
        panel = score(residual_rtn, test)
        gain = improvement(baseline_test, panel)
        report["models"][label] = {**panel, **gain}
        print(f"{label:<28}{panel['rmse_km']:>10.4f}{panel['median_km']:>12.4f}"
              f"{panel['p90_km']:>10.4f}{gain['rmse_pct']:>9.2f}%")
        return panel

    for name, residual_rtn in residuals.items():
        emit(name, residual_rtn)

    # ---------------------------------------------------------- ensemble
    if len(residuals) >= 2 and target_rtn is not None:
        names = list(residuals)
        a, b = residuals[names[0]], residuals[names[1]]
        weight = fit_blend_weight(a[calibration], b[calibration], target_rtn[calibration])
        blended = weight * a + (1 - weight) * b
        emit(f"ensemble (w={weight:.2f})", blended)
        report["ensemble_weight"] = weight

        # ------------------------------------------------------- shrinkage
        alpha = fit_shrinkage(blended[calibration], target_rtn[calibration])
        emit("ensemble + shrinkage", blended * alpha)
        report["shrinkage_alpha_rtn"] = [float(x) for x in alpha]
        print(f"\nshrinkage factors fitted on the calibration half "
              f"(R, T, N): {np.round(alpha, 4).tolist()}")
        if np.all(np.abs(alpha - 1.0) < 0.02):
            print("  -> within 2% of unity: the model is already well calibrated in "
                  "magnitude, so shrinkage has little to offer here.")
    elif target_rtn is None:
        print("\nresidual_rtn_val.pt is missing, so the blend weight and shrinkage "
              "factors cannot be fitted. Reporting the individual models only.")

    best = min(report["models"].items(), key=lambda item: item[1]["rmse_km"])
    print("-" * 74)
    print(f"best on the held-out half: {best[0]} — rmse {best[1]['rmse_km']:.4f} km, "
          f"{best[1]['rmse_pct']:.2f}% better than SGP4")
    report["best_variant"] = best[0]

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "holdout_ensemble_evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
