# 25 — Study plan v2: is tuning worth its compute? (one page)

**Chosen 14 Oct 2026 (route A). This page is the current plan.** Docs 00–24 are the history that led here. The full protocol is [doc 26](26_protocol_v2.md).

**Execution update, 16 Oct:** all 120 searches, 150 final models and the locked fold-10 evaluation
are complete. Current work is corrected reporting, publication figures and manuscript preparation
([results](27_results_v2.md), [reporting addendum](29_reporting_addendum_v2.md)). Doc 26 retains
the original frozen definitions and planned schedule. The stopping question is interpreted as
a comparison of complete policies; v2 does not isolate the effect of stopping within P.

**Working title.** *When is hyperparameter tuning worth its compute? A cost–benefit study of CNN-based ECG classification on PTB-XL.*

## Questions
1. **Gain:** how much does tuning improve held-out accuracy over (a) the original default and (b) a sensible task default?
2. **Cost:** how does the gain grow with search compute (two budget levels)?
3. **Stopping:** how much compute does our tuner's early stopping save, and how much accuracy does it give up?

## What we use
| | |
|---|---|
| Data | PTB-XL v1.0.3, 12-lead, 100 Hz. Official patient-separated folds. |
| Tasks | **Diagnosis**: 5 classes (NORM, MI, STTC, CD, HYP). **Rhythm**: 7 classes, frozen (AFIB, PACE, SARRH, SBRAD, SR, STACH, SVARR). |
| Splits | Search: train folds 1–7, validate on 8. Final models: train 1–8, early-stop on 9. **Test: fold 10, opened once, at the end.** |
| Model | Standard 1-D ResNet with 9 settings to tune. Not a new architecture. |
| Fixed baselines | **D0**: original default (kernel 9, dilation 1). **D1**: development-informed rhythm default, kernel 15 / dilation 2. For diagnosis, the task default is D0 (step 1). D1 has no per-run search once fixed; selecting it involved earlier development experiments. |
| Tuners | **Random search** (simple standard), **DEHB** (strong published method), **P** (our DE tuner with early stopping). |
| Budgets | 80 and 160 time units per search (unit = median training time: 11.2 s for diagnosis, 13.0 s for rhythm on G4). P may stop early. |
| Repeats | 10 per tuner × budget × task. |

## What we measure
- **Main result:** test AUROC gain over D0 and over D1, per task, tuner and budget, with uncertainty over patients and repeats. Shown as a **cost–quality chart**: accuracy against G4 hours.
- **Also reported:** results per class (with the number of positive cases), P's stopping times, and the compute P saved.
- **Not tested:** a "no worse than DEHB" verdict. Non-inferiority was excluded from v2 after the development precision checks (docs 22–24).

## Step 1: two local checks (done 14 Oct; rules registered before training, plan `55252745161e`)
Evidence: [evidence/default_range_check.json](evidence/default_range_check.json). Development fold 7, 5 repeats, local RTX 3050.
1. **The wider rhythm options did not establish a gain in this development check.** The study retained kernel 5/9/15 and dilation 1/2 as its declared search range, with no repeat timing required.

   | Compared with D1 | Difference | 95% CI |
   |---|---|---|
   | Kernel 25 | −0.002 | [−0.018, +0.014] |
   | Dilation 4 | −0.010 | [−0.028, +0.008] |

2. **D1 reduced diagnosis AUROC in this development check, so diagnosis retained D0.** The difference is −0.0097 (CI [−0.0146, −0.0052]), lower in all 5 repeats. This matches the diagnosis searches, which preferred kernel 5–9 and dilation 1. These findings motivated task-specific defaults for the main study.

## Original compute plan (historical estimate)
| Item | Count | G4 hours |
|---|---|---|
| Main searches | 2 tasks × 3 tuners × 2 budgets × 10 = **120** | ≤ 48 (≈ 42 expected, because P stops early) |
| Final models | 120 tuned + 30 baseline (D0 for both tasks, D1 for rhythm) = 150 | ≈ 2 |
| Session setup (about 10 sessions) | | ≈ 3 |
| **Total** | | **≈ 45–53 h ≈ 400–470 compute units (≈ $40–47)** |

Optional add-on: P without stopping, at 160 units (20 runs, about +11 h). It shows directly what continuing would have gained.
The development record at planning time estimated about 70 units for the pilot plus decision run.
Completed main work now totals 43.424 accounted search hours, 41.569 physical completed search
hours and 1.111 checkpoint-training hours. These are separate measures. Full development,
retries/setup/test overhead and billed idle usage remain unreconciled; see doc 29.

## Actual progress

| Date | Work | State |
|---|---|---|
| 14 Oct 2026 | Development checks, v2 implementation, protocol freeze | Complete; `protocol-v2-frozen` |
| 15–16 Oct | Main searches and final models | Complete; 120 searches / 150 models |
| 16 Oct | Analysis lock and fold-10 evaluation | Complete; `analysis-v2-locked` |
| 16 Oct onward | Reporting corrections, supplementary estimates and publication figures | Current reporting work |
| Next | Manuscript, checkpoint archive and complete cost reconciliation | Pending |

## Reporting interpretation

- Diagnosis has modest supported tuning gains over D0 under the primary analysis.
- Rhythm gains beyond development-informed D1 were not established; equivalence was not tested.
- P used less search time than DEHB, with uncertain comparative AUROC. V2 does not establish
  accuracy loss caused specifically by P's stopping component.
- Mean 160-minus-80 budget changes were small. Direct paired intervals are a separate reporting
  supplement specified after test inspection and do not add confirmatory hypotheses.

## Rules that protect the result
- Fold 10 is never opened before the analysis is locked, and then only once.
- The plan is locked (frozen protocol plus git tag) before any main run. Changes after that are logged as amendments.
- Development results (fold 7) are not reported as test results.
