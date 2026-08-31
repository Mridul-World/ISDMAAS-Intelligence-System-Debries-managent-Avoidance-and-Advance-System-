"""
================================================================================
PHASE 5.5 — RTN RESIDUAL TRAINER, v2
================================================================================
Trains the residual-correction Transformer that predicts the RTN-frame residual
between SGP4 and precise-orbit truth.

Target convention (frozen, unchanged from v1):

    residual_rtn = RTN_baseline(baseline_pos - truth_pos)
    truth_pos    = baseline_pos - RTN_to_ECI(residual_rtn, baseline_r, baseline_v)

WHAT IS DIFFERENT FROM v1, AND WHY EACH CHANGE EARNS ITS PLACE
--------------------------------------------------------------

1. THREE-WAY SPLIT, NOT TWO.
   v1 picked its checkpoint by lowest validation RMSE and then reported that
   same RMSE. That number is the minimum of forty noisy draws, so it is biased
   low by construction — it is a selection statistic, not a generalization
   estimate. v2 keeps a `dev` slice for early stopping and checkpoint choice and
   an untouched `test` slice that is scored exactly once, at the end.

2. AN ALIGNMENT GUARD BEFORE TRAINING, AND BEFORE SAVING.
   This repository currently ships a checkpoint whose predictions are
   ANTI-CORRELATED with the target on the radial and cross-track axes
   (r = -0.47 on both). That happens when a checkpoint is paired with tensors
   built under a different residual convention — here, a model file restored
   from a backup over a newer training run. Nothing in the pipeline noticed,
   and the committed validation report still claims a 7.6% improvement that the
   artifacts on disk do not reproduce.
   v2 refuses to save a checkpoint whose per-axis correlation with the target is
   negative, and refuses to start if the supplied tensors fail a reconstruction
   identity check. A silent frame mismatch is the most expensive failure mode
   this pipeline has, and it is cheap to detect.

3. EXPONENTIAL MOVING AVERAGE OF THE WEIGHTS.
   Averaging weights along the SGD trajectory lands in a flatter minimum than
   any single iterate. For a regression head on a noisy target this is one of
   the most dependable improvements available, and it costs one extra copy of
   the parameters.

4. WARMUP THEN COSINE DECAY.
   A pre-LayerNorm Transformer trained from scratch at full learning rate spends
   its first epochs undoing a bad initialization. A short linear warmup fixes
   that; cosine decay afterwards is the same schedule v1 used.

5. PER-AXIS LOSS WEIGHTING.
   The along-track residual has roughly ten times the standard deviation of the
   radial and cross-track ones, so after per-axis normalization the three axes
   contribute equally to the loss — even though along-track error is what
   actually dominates the position error the operator cares about. The loss is
   weighted by each axis's share of the unnormalized variance so that optimizing
   it optimizes the reported metric.

6. SEED ENSEMBLING.
   Averaging N independently seeded models reduces the variance component of
   the error. It is the most reliable accuracy gain here because the models
   differ only by initialization and batch order, so their errors decorrelate
   without any of them being worse.

7. POST-HOC PER-AXIS AFFINE CALIBRATION.
   Fitted on `dev`, applied to `test`. A regressor trained on a noisy target
   systematically under-uses its own signal; the MSE-optimal per-axis affine map
   recovers part of it for free.

Run:
    python train_phase55_rtn_v2.py                 # full tensors, 3 seeds
    python train_phase55_rtn_v2.py --seeds 1       # single model, faster
    python train_phase55_rtn_v2.py --self-test     # verify on synthetic data
    python train_phase55_rtn_v2.py --from-val      # demo run on the val tensors
                                                   # when X_train.pt is absent
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent

CONFIG = {
    "d_model": 96,
    "nhead": 6,
    "num_layers": 3,
    "dim_ff": 192,
    "dropout": 0.1,
    "epochs": 60,
    "batch": 256,
    "lr": 6e-4,
    "weight_decay": 1e-4,
    "warmup_frac": 0.05,
    "patience": 12,
    "ema_decay": 0.999,
    "grad_clip": 1.0,
    "seeds": 3,
    "dev_frac": 0.5,      # of the held-out portion; the rest is test
}

MU = 398600.4418


# ============================================================== model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
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
    def __init__(self, n_features: int, cfg: Dict):
        super().__init__()
        d = cfg["d_model"]
        self.input = nn.Sequential(
            nn.Linear(n_features, d), nn.LayerNorm(d), nn.GELU())
        self.positional = PositionalEncoding(d, dropout=cfg["dropout"])
        layer = nn.TransformerEncoderLayer(
            d, cfg["nhead"], cfg["dim_ff"], cfg["dropout"],
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(
            layer, cfg["num_layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(d // 2, 3))
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        return self.head(self.encoder(self.positional(self.input(x)))[:, -1, :])


class ExponentialMovingAverage:
    """
    Shadow copy of the weights, updated multiplicatively after each step.

    The effective decay is warmed up as `min(decay, (1 + step) / (10 + step))`.
    Without that warmup a decay of 0.999 has a time constant of a thousand
    steps, so on any short run the shadow is still mostly the random
    initialization when it is evaluated — which makes the averaged model look
    catastrophically worse than the raw one and silently sinks checkpoint
    selection. The self-test in this file exists to catch exactly that.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.shadow = {name: p.detach().clone()
                       for name, p in model.state_dict().items()
                       if p.dtype.is_floating_point}
        self.buffers = {name: p.detach().clone()
                        for name, p in model.state_dict().items()
                        if not p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    value.detach(), alpha=1.0 - decay)
            else:
                self.buffers[name] = value.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {**self.shadow, **self.buffers}


# ============================================================== geometry
def rtn_basis(state: torch.Tensor):
    r, v = state[:, :3], state[:, 3:]
    R = r / torch.norm(r, dim=-1, keepdim=True)
    W = torch.cross(r, v, dim=-1)
    W = W / torch.norm(W, dim=-1, keepdim=True)
    S = torch.cross(W, R, dim=-1)
    return R, S, W


def rtn_to_eci(rtn: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    R, S, W = rtn_basis(reference)
    return rtn[:, 0:1] * R + rtn[:, 1:2] * S + rtn[:, 2:3] * W


def reconstruct(baseline: torch.Tensor, residual_rtn: torch.Tensor) -> torch.Tensor:
    out = baseline.clone()
    out[:, :3] = baseline[:, :3] - rtn_to_eci(residual_rtn, baseline)
    return out


def error_panel(predicted: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    e = np.linalg.norm(predicted[:, :3] - truth[:, :3], axis=1)
    return {
        "rmse_km": float(np.sqrt((e ** 2).mean())),
        "median_km": float(np.median(e)),
        "p90_km": float(np.percentile(e, 90)),
        "p99_km": float(np.percentile(e, 99)),
    }


# ============================================================== guards
class AlignmentError(RuntimeError):
    """The tensors or the checkpoint do not agree with the frame convention."""


def check_reconstruction_identity(
    baseline: torch.Tensor, residual: torch.Tensor, truth: torch.Tensor,
    tolerance_km: float = 1e-3,
) -> float:
    """
    The stored residual must reconstruct the stored truth exactly.

    If this fails, the tensors were built with a different sign or frame
    convention from the one this trainer reconstructs with, and every metric
    downstream would be measuring the wrong thing. Returns the median error.
    """
    rebuilt = reconstruct(baseline, residual)
    error = torch.norm(rebuilt[:, :3] - truth[:, :3], dim=1)
    median = float(error.median())
    if median > tolerance_km:
        raise AlignmentError(
            f"Reconstruction identity failed: the stored RTN residual rebuilds "
            f"the truth to a median of {median:.4f} km, not ~0. The tensors were "
            f"built under a different residual convention from the one this "
            f"trainer assumes (truth = baseline - RTN_to_ECI(residual)). "
            f"Re-export the tensors before training."
        )
    return median


def check_prediction_alignment(
    predicted: np.ndarray, target: np.ndarray, min_correlation: float = 0.0
) -> Dict[str, float]:
    """
    Per-axis correlation between what the model predicts and the target.

    A negative correlation on any axis means the model is systematically wrong
    in SIGN there — subtracting its correction makes the orbit worse, not
    better. The shipped v1 checkpoint scores -0.47 on both the radial and
    cross-track axes, and nothing in the pipeline flagged it.
    """
    correlations = {}
    for index, axis in enumerate("RTN"):
        p, t = predicted[:, index], target[:, index]
        if p.std() < 1e-12 or t.std() < 1e-12:
            correlations[axis] = 0.0
        else:
            correlations[axis] = float(np.corrcoef(p, t)[0, 1])
    worst_axis = min(correlations, key=correlations.get)
    if correlations[worst_axis] < min_correlation:
        raise AlignmentError(
            f"Prediction alignment failed: the model's {worst_axis}-axis "
            f"predictions correlate {correlations[worst_axis]:+.3f} with the "
            f"target. A negative correlation means applying this correction "
            f"moves the orbit away from truth on that axis. Per-axis "
            f"correlations: {correlations}"
        )
    return correlations


# ============================================================== calibration
def fit_affine(predicted: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-axis least-squares fit of target ≈ scale * predicted + offset."""
    scale = np.ones(3)
    offset = np.zeros(3)
    for index in range(3):
        p, t = predicted[:, index], target[:, index]
        variance = float(((p - p.mean()) ** 2).sum())
        if variance > 1e-12:
            scale[index] = float(((p - p.mean()) * (t - t.mean())).sum() / variance)
        offset[index] = float(t.mean() - scale[index] * p.mean())
    return scale, offset


# ============================================================== training
def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one(
    seed: int,
    tensors: Dict[str, torch.Tensor],
    splits: Dict[str, np.ndarray],
    cfg: Dict,
    axis_weights: torch.Tensor,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Train a single model. Checkpoint selection uses `dev` only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, baseline, truth, residual = (
        tensors["X"], tensors["baseline"], tensors["y"], tensors["residual"])
    train_idx, dev_idx = splits["train"], splits["dev"]

    # Normalization statistics come from the TRAINING slice only; computing them
    # over everything would leak the held-out distribution into the model.
    mean = residual[train_idx].mean(0, keepdim=True)
    scale = residual[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    normalized = (residual - mean) / scale

    model = RTNTransformer(X.shape[2], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(
        TensorDataset(X[train_idx], normalized[train_idx]),
        batch_size=cfg["batch"], shuffle=True, drop_last=False)
    total_steps = cfg["epochs"] * max(len(loader), 1)
    warmup_steps = int(cfg["warmup_frac"] * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, total_steps, warmup_steps))
    ema = ExponentialMovingAverage(model, cfg["ema_decay"])
    huber = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def evaluate(state_dict, index) -> Dict[str, float]:
        probe = RTNTransformer(X.shape[2], cfg)
        probe.load_state_dict(state_dict)
        probe.eval()
        features = X[index]
        chunks = [probe(features[i:i + 1024]) * scale + mean
                  for i in range(0, len(features), 1024)]
        predicted_rtn = torch.cat(chunks, 0)
        rebuilt = reconstruct(baseline[index], predicted_rtn)
        return error_panel(rebuilt.numpy(), truth[index].numpy())

    best = {"rmse_km": float("inf")}
    best_state = deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for features, target in loader:
            optimizer.zero_grad()
            prediction = model(features)
            loss = (huber(prediction, target) * axis_weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            ema.update(model)

        metrics = evaluate(ema.state_dict(), dev_idx)
        improved = metrics["rmse_km"] < best["rmse_km"] - 1e-7
        if improved:
            best = {**metrics, "epoch": epoch}
            best_state = deepcopy(ema.state_dict())
            stale = 0
        else:
            stale += 1
        if verbose:
            print(f"  seed {seed} epoch {epoch:3d}  dev rmse {metrics['rmse_km']:8.4f}"
                  f"  median {metrics['median_km']:7.4f}{'  <- best' if improved else ''}")
        if stale >= cfg["patience"]:
            if verbose:
                print(f"  seed {seed}: early stop at epoch {epoch}")
            break

    final = RTNTransformer(X.shape[2], cfg)
    final.load_state_dict(best_state)
    final.eval()
    return final, {"dev": best, "residual_mean": mean, "residual_scale": scale}


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, mean, scale, batch: int = 1024):
    model.eval()
    return torch.cat([model(X[i:i + batch]) * scale + mean
                      for i in range(0, len(X), batch)], 0)


# ============================================================== data
def chronological_splits(n: int, dev_frac: float, train_frac: float = 0.6):
    """
    Split by position, which is chronological order in these tensors.

    Time-ordered data must never be split randomly: a random split lets the model
    interpolate between neighbouring ten-minute samples and reports a score it
    could not achieve on a genuinely future epoch.
    """
    train_end = int(train_frac * n)
    remaining = n - train_end
    dev_end = train_end + int(dev_frac * remaining)
    return {
        "train": np.arange(0, train_end),
        "dev": np.arange(train_end, dev_end),
        "test": np.arange(dev_end, n),
    }


def load_tensors(from_val: bool) -> Dict[str, torch.Tensor]:
    load = lambda name: torch.load(HERE / name, weights_only=False).float()
    suffix = "val" if from_val else "train"
    names = {
        "X": f"X_{suffix}.pt", "y": f"y_{suffix}.pt",
        "baseline": f"baseline_{suffix}.pt", "residual": f"residual_rtn_{suffix}.pt",
    }
    missing = [f for f in names.values() if not (HERE / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing tensors: {', '.join(missing)}.\n"
            f"Rebuild them with export_phase55_tensors.py, or pass --from-val to "
            f"train on the validation tensors alone (a smaller demonstration run)."
        )
    return {key: load(name) for key, name in names.items()}


def synthetic_tensors(n: int = 900, seq: int = 30, features: int = 23):
    """
    A small dataset with a known linear relationship, for --self-test.

    It exists so the training loop, the EMA, the reconstruction identity and the
    ensembling can be verified without the eight-gigabyte precise-orbit archive.
    """
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(n, seq, features, generator=generator)
    weights = torch.randn(features, 3, generator=generator) * 0.3
    residual = X[:, -1, :] @ weights + 0.05 * torch.randn(n, 3, generator=generator)
    radius = 7000.0 + 50.0 * torch.randn(n, 1, generator=generator)
    angle = torch.rand(n, 1, generator=generator) * 2 * math.pi
    position = torch.cat([radius * torch.cos(angle), radius * torch.sin(angle),
                          torch.zeros(n, 1)], dim=1)
    speed = math.sqrt(MU) / torch.sqrt(radius)
    velocity = torch.cat([-speed * torch.sin(angle), speed * torch.cos(angle),
                          torch.zeros(n, 1)], dim=1)
    baseline = torch.cat([position, velocity], dim=1)
    truth = reconstruct(baseline, residual)
    return {"X": X, "y": truth, "baseline": baseline, "residual": residual}


# ============================================================== entry point
def run(cfg: Dict, tensors: Dict[str, torch.Tensor], label: str,
        verbose: bool = True) -> Dict:
    n = len(tensors["X"])
    splits = chronological_splits(n, cfg["dev_frac"])
    print(f"windows: {n}  (train {len(splits['train'])} / "
          f"dev {len(splits['dev'])} / test {len(splits['test'])})")

    identity = check_reconstruction_identity(
        tensors["baseline"], tensors["residual"], tensors["y"])
    print(f"reconstruction identity: median {identity:.3e} km — tensors are "
          f"consistent with the frame convention")

    # Weight each axis by its share of the unnormalized residual variance, so
    # that minimizing the loss minimizes the position error that gets reported.
    variance = tensors["residual"][splits["train"]].var(0)
    axis_weights = (variance / variance.sum() * 3.0).clamp(0.05, 10.0)
    print(f"per-axis loss weights (R, T, N): "
          f"{[round(float(w), 4) for w in axis_weights]}")

    baseline_test = error_panel(
        tensors["baseline"][splits["test"]].numpy(),
        tensors["y"][splits["test"]].numpy())
    print(f"\nSGP4 baseline on the untouched test slice: "
          f"rmse {baseline_test['rmse_km']:.4f}  median {baseline_test['median_km']:.4f} km\n")

    models, metadata = [], []
    started = time.time()
    for seed in range(cfg["seeds"]):
        model, info = train_one(
            42 + seed, tensors, splits, cfg, axis_weights, verbose)
        models.append((model, info))
        metadata.append(info["dev"])

    mean = models[0][1]["residual_mean"]
    scale = models[0][1]["residual_scale"]
    test_idx = splits["test"]
    dev_idx = splits["dev"]

    dev_predictions = [predict(m, tensors["X"][dev_idx], i["residual_mean"], i["residual_scale"])
                       for m, i in models]
    test_predictions = [predict(m, tensors["X"][test_idx], i["residual_mean"], i["residual_scale"])
                        for m, i in models]

    truth_test = tensors["y"][test_idx].numpy()
    baseline_tensor = tensors["baseline"][test_idx]

    def score(residual_rtn: torch.Tensor) -> Dict[str, float]:
        return error_panel(reconstruct(baseline_tensor, residual_rtn).numpy(), truth_test)

    results = {"sgp4_baseline_test": baseline_test, "variants": {}}

    baseline_dev = error_panel(
        tensors["baseline"][dev_idx].numpy(), tensors["y"][dev_idx].numpy())
    baseline_dev_tensor = tensors["baseline"][dev_idx]
    truth_dev = tensors["y"][dev_idx].numpy()

    def score_dev(residual_rtn: torch.Tensor) -> Dict[str, float]:
        return error_panel(
            reconstruct(baseline_dev_tensor, residual_rtn).numpy(), truth_dev)

    def emit(name: str, residual_rtn: torch.Tensor, dev_residual: torch.Tensor):
        panel = score(residual_rtn)
        panel["rmse_improvement_pct"] = 100.0 * (
            1.0 - panel["rmse_km"] / baseline_test["rmse_km"])
        panel["median_improvement_pct"] = 100.0 * (
            1.0 - panel["median_km"] / baseline_test["median_km"])
        # Recorded so the winner can be chosen without looking at the test slice.
        dev_panel = score_dev(dev_residual)
        panel["dev_rmse_km"] = dev_panel["rmse_km"]
        panel["dev_median_km"] = dev_panel["median_km"]
        results["variants"][name] = panel
        print(f"{name:<32}{panel['dev_rmse_km']:>10.4f}{panel['rmse_km']:>10.4f}"
              f"{panel['median_km']:>11.4f}{panel['rmse_improvement_pct']:>10.2f}%")
        return panel

    print(f"\n{'variant':<32}{'dev rmse':>10}{'rmse km':>10}"
          f"{'median km':>11}{'vs SGP4':>11}")
    print("-" * 74)
    for index, (prediction, dev_prediction) in enumerate(
            zip(test_predictions, dev_predictions)):
        emit(f"single model (seed {42 + index})", prediction, dev_prediction)

    ensemble_test = torch.stack(test_predictions).mean(0)
    ensemble_dev = torch.stack(dev_predictions).mean(0)
    if len(models) > 1:
        emit(f"ensemble of {len(models)}", ensemble_test, ensemble_dev)

    # Calibration is fitted on dev and applied to test — never on test itself.
    target_dev = tensors["residual"][dev_idx].numpy()
    correlations = check_prediction_alignment(ensemble_dev.numpy(), target_dev)
    print(f"\nper-axis correlation with the target on dev "
          f"(R, T, N): {[round(correlations[a], 4) for a in 'RTN']}")

    affine_scale, affine_offset = fit_affine(ensemble_dev.numpy(), target_dev)
    scale_t = torch.from_numpy(affine_scale).float()
    offset_t = torch.from_numpy(affine_offset).float()
    emit("ensemble + affine calibration",
         ensemble_test * scale_t + offset_t, ensemble_dev * scale_t + offset_t)

    # THE REPORTED MODEL IS THE ENSEMBLE. There is no selection step.
    #
    # The obvious thing to do here is pick whichever variant scored best, and
    # every version of that idea is wrong. Picking on the test slice is the same
    # selection bias this trainer exists to avoid, moved one level up from epochs
    # to variants. Picking on dev is not much better when dev rankings do not
    # transfer — on this data the affine calibration beats SGP4's median on dev
    # and loses to it by a third on test, and the best single seed on dev is not
    # the best on test either. A selector tuned until it picks the variant you
    # already liked is just a slower way of fitting the test set.
    #
    # The ensemble needs no selection: averaging every seed is decided before any
    # score is seen, so its number is an estimate rather than a maximum. Every
    # other variant is reported for information, with both RMSE and median,
    # because a correction that shrinks the error tail while pushing the typical
    # case up is not an improvement for an operator screening thousands of
    # conjunctions a day — and only reporting RMSE would hide that.
    best_name = (f"ensemble of {len(models)}" if len(models) > 1
                 else "single model (seed 42)")
    results["reported_variant"] = best_name
    results["reported_variant_rationale"] = (
        "The ensemble is reported because it requires no selection: averaging "
        "all seeds is fixed in advance, so its score is an estimate and not the "
        "maximum of several draws."
    )

    seed_scores = [v["rmse_improvement_pct"] for k, v in results["variants"].items()
                   if k.startswith("single model")]
    warnings: List[str] = []
    if len(seed_scores) > 1 and max(seed_scores) - min(seed_scores) > 5.0:
        warnings.append(
            f"Single-seed improvement spans {min(seed_scores):.1f}% to "
            f"{max(seed_scores):.1f}%. On a spread that wide, any single-seed "
            f"figure is a draw from a broad distribution, not a result."
        )
    dev_best = min(results["variants"],
                   key=lambda k: results["variants"][k]["dev_rmse_km"])
    test_best = min(results["variants"],
                    key=lambda k: results["variants"][k]["rmse_km"])
    if dev_best != test_best:
        warnings.append(
            f"Dev rankings do not transfer: '{dev_best}' is best on dev but "
            f"'{test_best}' is best on test. Treat every ranking here as noise "
            f"and the dataset as too small to separate these variants."
        )
    regressed = [k for k, v in results["variants"].items()
                 if v["median_improvement_pct"] < 0]
    if regressed:
        warnings.append(
            "These variants improve RMSE while making the MEDIAN error worse, "
            "i.e. they help the tail at the typical case's expense: "
            + ", ".join(regressed)
        )
    results["warnings"] = warnings
    for warning in warnings:
        print(f"\n  WARNING: {warning}")
    results["per_axis_correlation_dev"] = correlations
    results["affine_scale_rtn"] = affine_scale.tolist()
    results["affine_offset_rtn"] = affine_offset.tolist()
    results["dev_metrics_per_seed"] = metadata
    results["config"] = {k: v for k, v in cfg.items()}
    results["label"] = label
    results["train_seconds"] = round(time.time() - started, 1)
    results["note"] = (
        "Every figure in `variants` is measured on a test slice that was never "
        "used for checkpoint selection, early stopping, normalization statistics "
        "or calibration. It is therefore directly comparable to the SGP4 "
        "baseline in `sgp4_baseline_test` and is not a selection statistic."
    )

    print("\n" + "-" * 74)
    chosen = results["variants"][best_name]
    print(f"REPORTED MODEL: {best_name} (no selection — see the note in the report)")
    print(f"  on the untouched test slice: rmse {chosen['rmse_km']:.4f} km "
          f"({chosen['rmse_improvement_pct']:+.2f}% vs SGP4), "
          f"median {chosen['median_km']:.4f} km "
          f"({chosen['median_improvement_pct']:+.2f}%)")

    model_dir = HERE / "model"
    model_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dicts": [m.state_dict() for m, _ in models],
            "residual_mean": mean,
            "residual_scale": scale,
            "affine_scale": torch.from_numpy(affine_scale).float(),
            "affine_offset": torch.from_numpy(affine_offset).float(),
            "config": cfg,
            "n_features": tensors["X"].shape[2],
            "label": label,
        },
        model_dir / f"rtn_residual_v2_{label}.pt",
    )
    report_dir = HERE / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / f"rtn_residual_v2_{label}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"saved: model/rtn_residual_v2_{label}.pt, "
          f"reports/rtn_residual_v2_{label}.json")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=CONFIG["seeds"],
                        help="how many independently seeded models to ensemble")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--self-test", action="store_true",
                        help="verify the pipeline on synthetic data")
    parser.add_argument("--from-val", action="store_true",
                        help="train on the validation tensors (demonstration run "
                             "when the training tensors are unavailable)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["seeds"] = max(1, args.seeds)
    cfg["epochs"] = max(1, args.epochs)

    if args.self_test:
        print("SELF-TEST — synthetic data with a known linear relationship\n")
        cfg.update(epochs=25, seeds=2, patience=25, batch=64, d_model=64,
                   nhead=4, num_layers=2, dim_ff=128)
        results = run(cfg, synthetic_tensors(), "selftest", not args.quiet)
        best = results["variants"][results["best_variant"]]
        if best["rmse_improvement_pct"] <= 50.0:
            raise SystemExit(
                f"SELF-TEST FAILED: the model recovered only "
                f"{best['rmse_improvement_pct']:.1f}% of a relationship that is "
                f"linear and noise-free by construction. The training loop, the "
                f"reconstruction, or the normalization is wrong."
            )
        print(f"\nSELF-TEST PASSED — recovered "
              f"{best['rmse_improvement_pct']:.1f}% of the synthetic signal.")
        return

    tensors = load_tensors(from_val=args.from_val)
    label = "fromval" if args.from_val else "full"
    if args.from_val:
        print("NOTE: training on the validation tensors because the training "
              "tensors are not present. This is a smaller, self-consistent "
              "demonstration run — not the production model.\n")
    run(cfg, tensors, label, not args.quiet)


if __name__ == "__main__":
    main()
