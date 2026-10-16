# 27 — Results of study v2 (fold-10 test, 16 Oct 2026)

**Confirmatory results under protocol v2** ([doc 26](26_protocol_v2.md); frozen 14 Oct, tag `protocol-v2-frozen`; analysis locked 16 Oct 07:54, tag `analysis-v2-locked`, before fold 10 was opened).

**Sources:**
- Primary and prespecified secondary estimates come from `scripts/analyze_v2.py primary`, unchanged since the lock: [evidence/v2_primary.json](evidence/v2_primary.json).
- Direct budget comparisons, absolute-score intervals for figure presentation and compute accounting are separate reporting supplements specified after test inspection: [doc 29](29_reporting_addendum_v2.md).
- Run audit: [evidence/v2_main_audit.json](evidence/v2_main_audit.json).
- 120 of 120 searches and 150 of 150 final models; all 10 seeds complete for every arm, so no seed was excluded.

**Test set (fold 10):**
- Diagnosis: 2,158 ECGs from 1,877 patients.
- Rhythm: 2,089 ECGs from 1,815 patients. Rare classes: SVARR 14 positives, PACE 28.

## 27.1 Diagnosis (5 superclasses); task default D0

| Arm | Test macro AUROC (mean of 10) | Search time per run | Gain over D0 [95% CI] | Holm p | Seeds > D0 |
|---|---|---|---|---|---|
| D0 (no search) | 0.9237 | — | — | — | — |
| Random, 80 units | 0.9265 | 14.9 min | +0.0029 [+0.0001, +0.0058] | 0.025 | 8/10 |
| Random, 160 units | 0.9270 | 29.8 min | +0.0033 [+0.0004, +0.0063] | 0.025 | 8/10 |
| DEHB, 80 units | 0.9271 | 14.9 min | +0.0034 [+0.0012, +0.0058] | 0.005 | 9/10 |
| DEHB, 160 units | 0.9277 | 29.8 min | +0.0040 [+0.0019, +0.0063] | 0.002 | 10/10 |
| P, 80 units | 0.9281 | 12.6 min (85% of cap) | +0.0044 [+0.0021, +0.0068] | 0.001 | 10/10 |
| P, 160 units | 0.9279 | 19.3 min (65% of cap) | +0.0042 [+0.0020, +0.0064] | 0.001 | 10/10 |

- **All six tuned arms beat D0** (primary family, Holm p ≤ 0.025). The gains are small, +0.003 to +0.004 AUROC.
- Increasing the budget changed mean AUROC by +0.00050 (random), +0.00060 (DEHB) and −0.00014 (P). Paired budget-effect intervals are reported in the supplemental analysis below.
- **P against the other tuners** (secondary):
  - versus DEHB: +0.0009 [−0.0007, +0.0027] at 80 units and +0.0002 [−0.0013, +0.0018] at 160 units;
  - search-time ratio P/DEHB: 0.84 [0.76, 0.91] and 0.64 [0.58, 0.70];
  - P used approximately 16% and 36% less accounted search time. The AUROC differences remained uncertain; equivalence and non-inferiority were not established.

## 27.2 Rhythm (7 labels); development-informed default D1 (kernel 15, dilation 2)

D1 was chosen after examining development tuning outcomes, then checked locally and fixed
before the main test (docs 24–26). It has no per-run search once fixed; its selection involved
earlier development work. That development cost and selection history are separate from the
per-run search costs in the table.

| Arm | Test macro AUROC | Search time per run | Gain over D1 [95% CI] | Holm p | Gain over D0 [95% CI] |
|---|---|---|---|---|---|
| D0 (no search) | 0.9486 | — | — | — | — |
| D1 (no search) | 0.9554 | — | — | — | +0.0068 [−0.0019, +0.0157] |
| Random, 80 units | 0.9559 | 17.3 min | +0.0005 [−0.0088, +0.0087] | 1.000 | +0.0073 [−0.0021, +0.0163] |
| Random, 160 units | 0.9562 | 34.5 min | +0.0008 [−0.0092, +0.0099] | 1.000 | +0.0075 [−0.0025, +0.0171] |
| DEHB, 80 units | 0.9601 | 17.3 min | +0.0047 [−0.0008, +0.0103] | 0.268 | **+0.0115 [+0.0043, +0.0189]** |
| DEHB, 160 units | 0.9591 | 34.5 min | +0.0037 [−0.0024, +0.0094] | 0.522 | **+0.0105 [+0.0032, +0.0179]** |
| P, 80 units | 0.9555 | 14.8 min (86% of cap) | +0.0001 [−0.0092, +0.0084] | 1.000 | +0.0069 [−0.0023, +0.0156] |
| P, 160 units | 0.9560 | 20.8 min (60% of cap) | +0.0006 [−0.0084, +0.0086] | 1.000 | +0.0074 [−0.0024, +0.0169] |

- **No tuner established a positive gain over D1** (primary family; every Holm p ≥ 0.27). The intervals allow potentially useful gains, so this does not establish equivalence to D1.
- **Gains over the old default D0** (secondary): DEHB's intervals exclude zero. D1's, random search's and P's do not (point estimates +0.007).
- **P against DEHB** (secondary):
  - −0.0045 [−0.0140, +0.0025] at 80 units and −0.0031 [−0.0130, +0.0048] at 160 units;
  - search-time ratio 0.84 [0.75, 0.94] and 0.58 [0.51, 0.69].
- **P against random search:** the point differences were −0.0004 and −0.0002; the paired intervals include zero. Equivalence was not tested.
- **Per class:** the default change mainly helps sinus arrhythmia (SARRH 0.866 → 0.913; tuned arms 0.913–0.921). The rare classes are noisy: SVARR, with 14 positives, ranges from 0.929 to 0.946 across arms.

## 27.3 Answers to the three questions

1. **Gain.**
   - Diagnosis: tuning gives a small, statistically supported gain of about +0.004 AUROC.
   - Rhythm: no search method established a gain over the development-informed default D1. Against D0, DEHB's secondary gain intervals exclude zero (about +0.011).
2. **Budget.** Going from 80 to 160 units changed mean AUROC by −0.00097 to +0.00060 across the six task–tuner combinations. The direct paired intervals are a reporting supplement defined after test inspection, rather than an additional confirmatory test.
3. **Stopping.**
   - P used **16% less accounted search time at the small budget and 36–42% less at the large budget**, compared with DEHB.
   - Diagnosis: AUROC point differences against DEHB were small and positive, with intervals including zero.
   - Rhythm: AUROC point differences against DEHB were negative (−0.003 to −0.0045), with intervals allowing both losses and gains. Differences against random search were also uncertain.
   - These compare complete policies. V2 contains no matched P-without-stopping, round-robin or patience control, so it does not attribute a quality/cost advantage to individual controller components.

**Central result:** for this CNN on PTB-XL, tuning produced modest diagnosis AUROC gains under the prespecified analysis. Gains beyond a development-informed rhythm default were not established. P returned models with less search time while its comparative quality remained uncertain. Increasing the search budget produced small changes in observed mean AUROC.

## 27.4 Honest notes

- **Development overstated the rhythm default effect.** On development fold 7, D1 gained +0.020 over D0 (doc 24). On fold 10 it is +0.007, with an interval that includes zero. D0 did better on fold 10's SARRH cases (0.866) than on fold 7's (0.751). D1 was chosen on development data, so some regression toward smaller effects was expected (protocol §12).
- **Figures:** the original plots remain historical outputs of the locked script. Publication figures use a separate script, the frozen primary numbers and explicitly supplemental intervals. Captions distinguish patient-and-seed uncertainty, seed-only time uncertainty and descriptive per-class means; see [doc 29](29_reporting_addendum_v2.md).
- **Session interruptions** (Wi-Fi drops on the workstation) restarted searches from scratch. All 6 runtimes passed the timing check (≤ 5.3% from baseline). Every completed search ran in one runtime.
- **Limits** (protocol §12): one dataset; one CNN family; D1 data-informed; ten seeds on one fixed test fold; rare rhythm classes imprecise.

## 27.5 Compute accounting

| Completed work | Diagnosis | Rhythm | Total G4 hours |
|---|---:|---:|---:|
| Accounted search (includes charged cache reuse) | 20.205 | 23.219 | 43.424 |
| Physical completed search | 19.310 | 22.259 | 41.569 |
| Checkpoint training | 0.492 | 0.619 | 1.111 |

Documented physical completed search plus checkpoint training is **42.680 G4 hours**.
Checkpoint-training timers end before final prediction, thresholding and model export.
Aborted attempts, setup, development/default selection, test inference and idle billing are
additional or unavailable. Accounted cache charges support fair method comparison; they are
not extra physical runtime. A complete billed-cost or monetary total is not claimed.

## 27.6 Direct budget-effect estimates

Paired 160-minus-80 AUROC estimates and 95% patient/seed bootstrap intervals are reported
in [doc 29](29_reporting_addendum_v2.md) and the supplemental figure. They use all ten paired
seeds per method and existing predictions. They are labelled as specified after test inspection;
the original primary family and its Holm-adjusted results retain their frozen definition.
All six supplemental intervals include zero, leaving the direction and magnitude of budget
effects uncertain. This does not establish equivalence between the two budgets.
