# ML Validation Report

**Date:** 2026-08-31
**Verdict: FAIL — the ML layer is NOT validated and must NOT be put into production.**

The system serves **SGP4 alone**. That is the correct outcome given the evidence
below, not a gap to be closed by wiring the model in.

---

## 1. Summary

| Question | Answer |
|---|---|
| Is an ML model in the serving path? | **No.** No module under `phase11/` imports torch or loads a checkpoint. |
| Does the shipped checkpoint reproduce its validation report? | **No.** Report claims +7.62% RMSE / +36.6% median; measured +2.95% / **−2.57%**. |
| Does it improve the SGP4 baseline on unseen data? | **Not reliably.** See sections 3 and 4. |
| Can it be retrained here? | **No.** The training set is absent from the repository. |
| Is there a safe fallback? | **Yes, and it is the default.** SGP4 is the validated baseline and the only thing serving. |

---

## 2. The serving path contains no model

Verified by search: nothing in `phase11/` imports `torch`, references a `.pt`
file, or calls `load_state_dict`. `phase11_api.py` propagates with SGP4 and
nothing else.

Every assessment record written by the service carries
`model_version: "none (SGP4 baseline)"`. That is the honest value and it makes
the absence visible in every audit record rather than leaving it to be assumed.

**The README and architecture diagram previously implied the model was in the
loop.** That has been corrected in `README.md` and `docs/PRODUCTION_AUDIT.md` §11.

---

## 3. The shipped checkpoint does not reproduce its own report

Full evidence in [`docs/technical/09-model-artifact-audit.md`](technical/09-model-artifact-audit.md).

| Metric | SGP4 baseline | Deployed checkpoint | Committed report claims |
|---|---:|---:|---:|
| RMSE (km) | 1.9071 | 1.8508 | 1.7618 |
| Median (km) | 0.6090 | **0.6247** | 0.3859 |
| RMSE improvement | — | **+2.95%** | +7.62% |
| Median improvement | — | **−2.57%** | +36.64% |

The checkpoint on disk is byte-identical to a backup two weeks older than the
training run that produced the report, and its file modification time predates
the report's. The most likely sequence is that `model/_production_backup/` was
restored over `model/` after the run.

**The deployed checkpoint's predictions are anti-correlated with the target** on
two of three axes:

| Axis | Correlation with target | Predicted σ | True σ |
|---|---:|---:|---:|
| Radial | **−0.470** | 0.151 | 0.168 |
| Along-track | +0.278 | 0.444 | 1.824 |
| Cross-track | **−0.470** | 0.132 | 0.281 |

A negative correlation means the correction is applied in the wrong direction:
subtracting it moves the predicted position *away* from truth. −0.47 on two axes
is far outside anything noise explains.

---

## 4. What a correctly-trained model achieves on this data

The v2 trainer was run on the only tensors present (the validation set,
17 265 windows, split 60/20/20 chronologically). This is a **demonstration on a
small dataset, not the production model**, and it is reported as such.

| Variant | Test RMSE (km) | Test median (km) | vs SGP4 RMSE | vs SGP4 median |
|---|---:|---:|---:|---:|
| SGP4 baseline | 2.2596 | 0.6606 | — | — |
| single seed 42 | 2.2533 | 0.6525 | +0.28% | +1.23% |
| single seed 43 | 2.2112 | 0.6689 | +2.14% | −1.25% |
| single seed 44 | 1.9950 | 0.7762 | +11.71% | −17.50% |
| **ensemble of 3** | **2.1139** | **0.5858** | **+6.45%** | **+11.32%** |
| ensemble + affine calibration | 2.0260 | 0.8978 | +10.33% | −35.89% |

### What this actually shows

- **Single-seed results span 0.28% to 11.71%** — an 11.4-point spread across
  three seeds differing only by initialization. Any single-seed figure from this
  dataset is a draw from a wide distribution, not a result.
- **Dev rankings do not transfer to test.** The affine calibration beats SGP4's
  median on dev and loses to it by 36% on test. The best single seed on dev is
  not the best on test.
- **Only the ensemble improves both metrics**, and it is the only variant chosen
  without looking at any score — averaging all seeds is decided in advance.
- Per-axis correlations with the target are **+0.08 radial, +0.23 along-track,
  +0.03 cross-track**. Positive (unlike the shipped checkpoint) but weak. Only
  along-track carries usable signal, which is physically sensible: along-track is
  where SGP4's drag and period error is predictable from element-set features.

**Conclusion: this dataset is too small to establish a reliable improvement.**
The honest number is the ensemble's +6.45% RMSE / +11.32% median on a held-out
slice, with the caveat that it comes from 10 359 training windows.

---

## 5. Metrics the request asked for, and which are unavailable

| Metric | Status |
|---|---|
| MAE, RMSE, median, p95, p99, max | RMSE / median / p90 / p99 computed; **only on the demonstration split** |
| Per-axis radial / along-track / cross-track error | Correlations reported above; full per-axis error distributions require the training set |
| Error vs prediction horizon | **Unavailable** — needs the multi-horizon dataset, absent |
| Error vs orbit regime | **Unavailable** — training data covers 5 sun-synchronous Sentinels only; no regime diversity exists to measure |
| Inference latency | **Not measured** — no model in the serving path to measure |
| Model size | 426 KB (v1 checkpoints), 3.4 MB (v2 3-seed ensemble) |

These are marked unavailable rather than estimated. Producing numbers for them
without the data would be fabrication.

---

## 6. Leakage prevention

The v2 trainer splits **chronologically, by position**, never randomly:

```
train 60%  |  dev 20%  |  test 20%
```

Time-ordered data split randomly lets the model interpolate between neighbouring
ten-minute samples and report a score it could not achieve on a genuinely future
epoch. Leave-one-satellite-out is supported by `export_phase55_tensors.py` via
`ISDMAAS_HOLDOUT_NORAD` but could not be exercised here without the training set.

Normalization statistics are computed from the **training slice only**.
Calibration is fitted on **dev** and applied to **test**.

---

## 7. Guards now in place (Phase 7)

`phase55/train_phase55_rtn_v2.py` refuses to produce a model that repeats the
failure in section 3:

| Guard | What it catches |
|---|---|
| **Reconstruction identity check**, before training | Tensors built under a different residual sign or frame convention from the one the trainer reconstructs with. Fails if the stored residual does not rebuild the stored truth to within a metre. |
| **Per-axis alignment check**, before saving | A model whose predictions correlate *negatively* with the target on any axis — exactly the shipped checkpoint's condition — is rejected with an explicit error rather than saved. |
| **Three-way split** | Checkpoint selection and early stopping use `dev`; `test` is scored once, at the end. |
| **No variant selection** | The ensemble is reported because it requires no selection. Choosing "whichever scored best" is the same selection bias one level up. |
| **Instability warnings** | Fires when single-seed spread exceeds 5 points, when dev rankings do not transfer to test, or when a variant improves RMSE while worsening median. |

A self-test on synthetic data with a known linear relationship
(`--self-test`) verifies the training loop, EMA, reconstruction and ensembling
end to end. **It found a real defect**: an EMA decay of 0.999 with no warmup left
the averaged weights dominated by the random initialization on short runs.
Recovery of the known signal went from **5.7% to 60.1%** once warmup was added.

---

## 8. What is still missing before a model can ship

1. **Restore or rebuild the training set.** `phase55/data/phase55_dataset.csv`
   and `X_train.pt` are absent; rebuilding needs the ESA POD archive and CDSE
   credentials (which must be rotated first — see `SECURITY_AUDIT.md` §1).
2. **Retrain with `train_phase55_rtn_v2.py`** and record the result against the
   untouched test slice.
3. **Record the checkpoint SHA-256 in the validation report.** A report and a
   model file drifting apart unnoticed is the root cause of section 3, and a
   hash closes the whole class.
4. **Make the deployment path refuse a checkpoint whose hash is not the one that
   was validated.** A few lines, and it is the difference between "we validated a
   model" and "we validated *this* model".
5. **Delete `phase55/model/_production_backup/`** or move it outside the model
   search path. A backup directory shadowing the live one is what caused this.
6. **Only then** wire it in, behind a gate that falls back to SGP4 on any load,
   shape, normalization or version mismatch — and records that it fell back.

---

## 9. Verdict

**FAIL.** The ML layer is not validated, does not reproduce its claimed accuracy,
and is not in production.

The system's prediction layer is SGP4, which is the validated baseline. That is
the correct and safe configuration, and no change should be made to it until
steps 1–6 above are complete.

---

## 10. Reproducing

```bash
cd phase55
python evaluate_ensemble.py                  # re-score the shipped checkpoints
python train_phase55_rtn_v2.py --self-test   # verify the training pipeline itself
python train_phase55_rtn_v2.py --from-val    # the demonstration run in section 4
```

The first two run on artifacts already in the repository. The third takes ~90
minutes on CPU.
