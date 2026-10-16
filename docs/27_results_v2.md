# 27 — Results of study v2 (fold-10 test, 16 Oct 2026)

**Confirmatory results under protocol v2** ([doc 26](26_protocol_v2.md); frozen 14 Oct, tag `protocol-v2-frozen`; analysis locked 16 Oct 07:54, tag `analysis-v2-locked`, before fold 10 was opened).

**Sources:**
- Every estimate comes from `scripts/analyze_v2.py primary`, unchanged since the lock: [evidence/v2_primary.json](evidence/v2_primary.json).
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
- **Doubling the budget added nothing measurable** for any tuner.
- **P against the other tuners** (secondary):
  - versus DEHB: +0.0009 [−0.0007, +0.0027] at 80 units and +0.0002 [−0.0013, +0.0018] at 160 units;
  - search-time ratio P/DEHB: 0.84 [0.76, 0.91] and 0.64 [0.58, 0.70];
  - P matched DEHB's accuracy with 16% and 36% less search time.

## 27.2 Rhythm (7 labels); task default D1 (kernel 15, dilation 2)

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

- **No tuner showed a supported gain over the sensible task default D1** (primary family; every Holm p ≥ 0.27).
- **Gains over the old default D0** (secondary): DEHB's intervals exclude zero. D1's, random search's and P's do not (point estimates +0.007).
- **P against DEHB** (secondary):
  - −0.0045 [−0.0140, +0.0025] at 80 units and −0.0031 [−0.0130, +0.0048] at 160 units;
  - search-time ratio 0.84 [0.75, 0.94] and 0.58 [0.51, 0.69].
- **P against random search:** about equal (−0.0004 and −0.0002).
- **Per class:** the default change mainly helps sinus arrhythmia (SARRH 0.866 → 0.913; tuned arms 0.913–0.921). The rare classes are noisy: SVARR, with 14 positives, ranges from 0.929 to 0.946 across arms.

## 27.3 Answers to the three questions

1. **Gain.**
   - Diagnosis: tuning gives a small, statistically supported gain of about +0.004 AUROC.
   - Rhythm: no search method gave a supported gain over a sensible task default. Against the old default, only DEHB's gain (+0.011) is clearly supported.
2. **Cost.** Going from 80 to 160 time units (about 15 to 30–35 minutes per search) **did not improve accuracy** for any tuner on either task.
3. **Stopping.**
   - Our tuner's early stopping cut search time by **16% (small budget) and 36–42% (large budget)**.
   - Diagnosis: no detectable accuracy loss.
   - Rhythm: a small, uncertain shortfall against DEHB (−0.003 to −0.0045, intervals include zero), and level with random search.

**One-sentence result:** for this CNN on PTB-XL, hyperparameter search buys at most a few thousandths of AUROC; on rhythm, a task-appropriate default does as well as searching; doubling the search budget does not help; and adaptive early stopping saves a third or more of the search compute at little or no cost.

## 27.4 Honest notes

- **Development overstated the rhythm default effect.** On development fold 7, D1 gained +0.020 over D0 (doc 24). On fold 10 it is +0.007, with an interval that includes zero. D0 did better on fold 10's SARRH cases (0.866) than on fold 7's (0.751). D1 was chosen on development data, so some regression toward smaller effects was expected (protocol §12).
- **The cost–quality figures** written by the locked script have a layout fault: the default labels squeeze the plot. The numbers are correct. Paper figures will be redrawn from `v2_primary.json` without recomputing anything.
- **Session interruptions** (Wi-Fi drops on the workstation) restarted searches from scratch. All 6 runtimes passed the timing check (≤ 5.3% from baseline). Every completed search ran in one runtime.
- **Limits** (protocol §12): one dataset; one CNN family; D1 data-informed; ten seeds on one fixed test fold; rare rhythm classes imprecise.
