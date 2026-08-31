# 09 — Model Artifact Audit

**Date:** 2026-08-31 · **Status:** open finding · **Owns:** reproducibility of the
prediction-layer validation numbers

This document records a reproducibility failure found while hardening the
service. It is filed here rather than folded into
[`07-validation-results.md`](07-validation-results.md) because that document is
append-only numbers; this is the reason some of those numbers cannot currently
be reproduced.

---

## 1. The finding, in one paragraph

The residual-correction checkpoint shipped in `phase55/model/` is **not the
checkpoint the committed validation report describes**, and it does not
reproduce that report's numbers. Re-running the shipped checkpoint against the
shipped validation tensors yields a **2.95% RMSE improvement over SGP4 and a
2.57% median *regression***, where the committed report claims **7.62% RMSE and
36.6% median improvement**. The shipped checkpoint's predictions are
**anti-correlated with the target** on the radial and cross-track axes.

---

## 2. Evidence

### 2.1 The artifacts disagree in date

| Artifact | Modified | Note |
|---|---|---|
| `phase55/model/rtn_physics_ml_phase55.pt` | 2026-06-12 02:06 | the deployed checkpoint |
| `phase55/model/_production_backup/rtn_physics_ml_phase55.pt` | 2026-06-12 02:06 | **byte-identical**, SHA-256 `5379be96602d…` |
| `phase55/X_val.pt`, `residual_rtn_val.pt`, `phase55_meta.json` | 2026-06-25 15:28 | the tensors |
| `phase55/model/residual_norm_phase55.pt` | 2026-06-25 15:29 | written at the *start* of a training run |
| `phase55/reports/rtn_physics_ml_phase55_validation.json` | 2026-06-25 15:47 | written at the *end* of that run |

The normalization file and the report bracket a training run on 2026-06-25. The
trainer saves its checkpoint whenever validation RMSE improves, so that run must
have written one. The checkpoint on disk is from two weeks earlier and is
byte-identical to the backup directory. The most likely sequence is that
`_production_backup/` was restored over `model/` after the run, replacing the
newly trained weights while leaving the run's report and normalization file in
place.

### 2.2 The report's numbers do not reproduce

Running the deployed checkpoint over the full validation set:

| | SGP4 baseline | deployed checkpoint | committed report |
|---|---|---|---|
| RMSE (km) | 1.9071 | 1.8508 | 1.7618 |
| Median (km) | 0.6090 | **0.6247** | 0.3859 |
| RMSE improvement | — | **+2.95%** | +7.62% |
| Median improvement | — | **−2.57%** | +36.64% |

The SGP4 baseline figure in the committed report (`1.907130479812622`) matches
the current tensors to the last digit, so the report was generated against
*these* tensors — it is the model half of the pair that changed.

### 2.3 The deployed checkpoint is anti-correlated on two axes

Correlation between the predicted RTN residual and the stored target residual,
over the full validation set:

| Axis | Correlation | Predicted σ | True σ |
|---|---|---|---|
| Radial | **−0.470** | 0.151 | 0.168 |
| Along-track | +0.278 | 0.444 | 1.824 |
| Cross-track | **−0.470** | 0.132 | 0.281 |

A negative correlation means the correction is applied in the wrong direction on
that axis: subtracting it moves the predicted position *away* from truth. A value
of −0.47 on two of three axes is far outside anything noise explains. The
along-track axis — which carries most of the position error — is only weakly
correlated at +0.28, and the model's predicted spread there (σ 0.44) is a quarter
of the true spread (σ 1.82), so it barely moves in the direction that matters.

An MSE-optimal per-axis rescaling fitted on a held-out half returns
`α = (−0.54, +1.55, −1.02)`. The negative α on radial and cross-track is the same
finding stated as a correction factor: the fit wants those two axes' signs
flipped.

### 2.4 What this most likely means

A positive-definite normalization (`prediction × σ + μ`) cannot flip a sign, so
the anti-correlation is **not** caused by pairing the checkpoint with the wrong
`residual_norm_*.pt`. The remaining explanation consistent with the evidence is
that the June-12 checkpoint was trained against a dataset whose RTN residual
convention differs from the one the June-25 export produced — a frame or sign
change in `export_phase55_tensors.py` between the two dates.

**This has not been root-caused further**, because the training dataset
(`phase55/data/phase55_dataset.csv`) and `X_train.pt` are not present in the
repository, so the June-12 export cannot be reconstructed for comparison.

---

## 3. What this does and does not invalidate

**Directly affected** — do not cite until re-run:

- The single-horizon (1-day) improvement figure for the phase-5.5 residual model.
- Any statement that the deployed model improves median error. On the artifacts
  present, it makes median error slightly worse.

**Not tested here, and therefore not claimed to be wrong:**

- The multi-horizon sweep (`paper_evidence/multi_horizon*.json`) and the LOSO
  generalization study (`paper_evidence/loso*.csv`). Those came from separate
  runs whose checkpoints are not in `phase55/model/`. They may be sound; they
  have simply not been re-verified, and they should be before publication.
- The physics-consistency-loss null result. If anything this audit *strengthens*
  it: the `physics_ml` and `pure_ml` checkpoints differ by at most 6×10⁻⁴ km in
  their predictions, which is a stronger statement of "no effect" than the
  original report made.

**Unaffected:**

- Everything in the risk, planning and safety layers. Those are deterministic
  astrodynamics with no learned component, and they are covered by the test
  suite in `phase11/tests/`.

---

## 4. Remediation

### 4.1 Done

`phase55/train_phase55_rtn_v2.py` adds two guards that would have caught this
before it shipped:

- **Reconstruction identity check**, before training starts. The stored residual
  must rebuild the stored truth from the stored baseline to within a metre. If
  the tensors were built under a different convention from the one the trainer
  reconstructs with, it refuses to run.
- **Per-axis alignment check**, before a checkpoint is accepted. A model whose
  predictions correlate negatively with the target on any axis is rejected with
  an explicit error rather than saved and reported.

`phase55/evaluate_ensemble.py` re-scores any existing checkpoint against a
strictly held-out half of the validation set, so a claim can be checked without
retraining.

The v2 trainer also fixes a defect its own self-test surfaced: an EMA decay of
0.999 with no warmup leaves the averaged weights dominated by the random
initialization on any short run. With the standard `min(decay, (1+t)/(10+t))`
warmup, the synthetic self-test goes from recovering 5.7% of a known signal to
recovering 60.1%.

### 4.2 Required to close this finding

1. Restore or rebuild `phase55/data/phase55_dataset.csv` (needs the ESA POD
   archive and CDSE credentials).
2. Re-export tensors with `export_phase55_tensors.py`.
3. Retrain with `train_phase55_rtn_v2.py`, which reports on a test slice never
   used for checkpoint selection.
4. Append the new numbers to `07-validation-results.md` **with the SHA-256 of
   the checkpoint they describe**, so a report and a model file can never drift
   apart unnoticed again.
5. Delete `phase55/model/_production_backup/`, or move it outside the model
   search path. A backup directory that shadows the live one is what caused this.

### 4.3 Process change worth making

The pipeline had no step that could have detected a model/data mismatch. Record
the checkpoint hash in every validation report, and make the deployment path
refuse to load a checkpoint whose recorded hash is not the one that was
validated. That is a few lines and it closes the whole class of failure.

---

## 5. How to reproduce this audit

```bash
cd phase55
python evaluate_ensemble.py          # held-out re-scoring of the shipped models
python train_phase55_rtn_v2.py --self-test   # verifies the training pipeline itself
```

Both run on the artifacts already in the repository and need no external data.
