# Reporting addendum and publication figures — v2

16 October 2026. This addendum records reporting corrections and supplemental estimates
specified **after inspection of the fold-10 results**. It preserves the original primary
analysis and locally frozen protocol. The existing test predictions are the analysis inputs.

## 1. Corrections

The corrected [results](27_results_v2.md) describe uncertain quality differences without claiming
equivalence, non-inferiority or a negligible accuracy cost. The current README and study-plan
summary now show that all 120 searches, 150 final models and the locked test evaluation are
complete. Historical experiment queues are labelled accordingly.

The supported central interpretation is:

> Tuning produced modest diagnosis AUROC gains under the prespecified analysis. Gains beyond
> the development-informed rhythm default were not established. P returned models with less
> search time while comparative quality remained uncertain. Increasing the search budget
> produced small changes in observed mean AUROC.

V2 compares complete policies. It has no matched P-without-stopping, round-robin or patience
control and does not attribute a quality/cost advantage to individual controller components.

## 2. Supplemental definition and provenance

The [supplement plan](evidence/v2_reporting_supplement_plan.json) pins the analysis code, source
predictions, run/final metadata, local label/record arrays and the original lock before the
supplemental calculation. Its additions are:

- All six paired 160-minus-80 budget effects, one per task and tuner.
- Absolute test AUROC intervals for displaying means in cost–quality plots.
- Separate accounted search, physical search, cached charges and checkpoint-training totals.

`scripts/reporting_supplement_v2.py` uses the original metric and patient/paired-seed bootstrap
draws (10,000, RNG seed 0), reusing arm-level metric calculations within each draw. It checks
the original locked contrast intervals and bootstrap tail values against the archived report.
A focused test checks draw-for-draw equivalence with the original paired implementation,
including repeated patient records and tied scores.

The budget and absolute-score intervals are pointwise supplemental estimates. No new
confirmatory hypothesis test, NI margin or equivalence decision is introduced. The source
protocol, analysis lock and `scripts/analyze_v2.py` retain their original bytes.

## 3. Paired budget effects

Differences are macro AUROC at **160 minus 80** time units, averaged across ten paired seeds.
The intervals resample patients with all their records and paired seeds. Positive values
favor 160 units. They are not evidence of equivalence when they include zero.

| Task | Tuner | Δ AUROC (160 − 80) | 95% paired patient/seed CI |
|---|---|---:|---|
| Diagnosis | Random | +0.00050 | [-0.00058, +0.00186] |
| Diagnosis | DEHB | +0.00060 | [-0.00061, +0.00198] |
| Diagnosis | P | -0.00014 | [-0.00193, +0.00154] |
| Rhythm | Random | +0.00024 | [-0.00494, +0.00528] |
| Rhythm | DEHB | -0.00097 | [-0.00619, +0.00361] |
| Rhythm | P | +0.00048 | [-0.00491, +0.00595] |

All six intervals include zero. This leaves the direction and size of budget effects uncertain;
it does not establish no effect. All ten pairs are retained, including pairs with identical
predictions. The same replay reproduced all **27 locked primary/secondary contrast intervals
and bootstrap tail values** (10 diagnosis, 17 rhythm).

Source: [supplemental estimates](evidence/v2_reporting_supplement.json).

## 4. Compute accounting and default selection

| Completed work | Diagnosis | Rhythm | Total G4 hours |
|---|---:|---:|---:|
| Accounted search | 20.205 | 23.219 | 43.424 |
| Physical completed search | 19.310 | 22.259 | 41.569 |
| Charged cache reuse | 0.895 | 0.960 | 1.855 |
| Checkpoint training | 0.492 | 0.619 | 1.111 |

Accounted search equals physical search plus charged cached work. Cache charges support method
comparison and must not be added to physical runtime as if they were additional computation.
Documented completed physical search plus checkpoint training is **42.680 hours**. Training
timers end before final prediction, thresholding and model saving. Aborted work, setup, test
inference, idle billing and earlier development work are separate or unavailable; no complete
billed total or monetary cost is claimed.

Rhythm D1 was chosen after inspecting development tuning outcomes, checked locally and fixed
before the main test. Its kernel 15 / dilation 2 settings were not an independently supplied
expert default. D1 has no per-run search after selection; the development cost of choosing it
is disclosed separately. D0 and D1 are both retained in the results. The original planning
budget is historical and does not substitute for a measured final cost inventory.

## 5. Figure preparation

Publication figures and editable/vector exports are in [figures/publication_v2](../figures/publication_v2/README.md).
Each figure has PDF, SVG and 300-dpi PNG versions. CSV tables contain every plotted estimate;
the figure manifest records input/output hashes and the source lock.

| Figure | Purpose | Uncertainty and status |
|---|---|---|
| 1. Cost–quality | Test AUROC against mean accounted search time | Absolute 95% patient/seed intervals, added in this reporting supplement; default mean lines |
| 2. Paired gains | Tuned arm minus task default | Locked primary patient/seed intervals; Holm decisions within task; intervals themselves are pointwise |
| 3. P versus DEHB | Quality and time ratios side by side | Locked secondary quality intervals; paired-seed time-ratio intervals |
| S1. Budget effect | Paired 160-minus-80 differences | Supplemental patient/seed intervals, specified after test inspection |
| S2. Per-class gains | Mean class AUROC differences and positive record counts | Locked secondary descriptive means; no per-class intervals or tests |

The [captions](../figures/publication_v2/CAPTIONS.md) explain uncertainty, D1's selection,
rare-class limitations and the scope of each comparison. The plots avoid equivalence or
component-attribution claims. Absolute-score intervals are not used to infer paired effects.

## 6. Reproduction

```powershell
D:\envs\bamo\python.exe -s scripts/reporting_supplement_v2.py
D:\envs\bamo\python.exe -s scripts/prepare_publication_figures.py
```

The supplement resumes completed task calculations with the identical registered definition.
Figures are generated separately from scientific result production. The original runtime
notebooks and their plots remain execution evidence.

## 7. Remaining publication-package items

- Preserve and hash the 150 original model checkpoints from Drive. They are absent from the
  workspace result exports and cannot be reconstructed by a reporting correction.
- Reconcile actual billed runtime and interrupted work if those records can be obtained.
  Otherwise report the completed-job totals with the exclusions above.
- Complete manuscript text and journal-specific figure formatting after reviewing the figures.
