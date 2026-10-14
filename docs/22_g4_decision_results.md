# 22 — G4 decision run results (14 Oct 2026)

Development evidence only. No confirmatory search was run and no waveforms or model performance from folds 8–10 were assessed. Source: `runs/colab/decision_2026-10-14/` (227 files; every file matches `export_manifest.json`; code bundle `21dce7d4…`, the one verified in doc 21). The executed notebook is archived next to it.

Runtime: about 4 h 50 min on one G4 (RTX PRO 6000 Blackwell), roughly 43 compute units.

## 22.1 What ran

| Step | Data | Output |
|---|---|---|
| Timing drift vs session 1 | superclass, same 4 configs as the baseline | `timing_drift/*.json` |
| Task calibration (common workload) | superclass and rhythm, train 1–6 / val 7 | `task_calibration/{task}/summary.json` |
| Precision calibration | rhythm, search 1–4 / val 5 | `precision/rhythm/calibration_summary.json` |
| Matched-control smoke | 4 arms × 2 seeds, 80 T_ref | `precision/rhythm/matched_smoke_summary.json` |
| Paired P vs native DEHB | 5 seeds, 160 T_ref, then refit on 1–5/6 and assessment on fold 7 | `precision/rhythm/precision_summary.json` |

## 22.2 Results

**Timing is stable.** On the session-1 configurations, the new session ran at a median 1.017× the baseline duration (limit 1.10×). The largest measured deadline overshoot was 0.1404 s; no late result entered an accepted trajectory. Exact zero-overshoot compliance is not claimed.

**Task calibration (train 1–6 / val 7):**

| Task | T_ref (median of 16 configs) | Seed SD (macro AUROC) | Re-evaluation tolerance (95th pct) |
|---|---|---|---|
| Superclass | 11.17 s | 0.0017 | 0.0044 |
| Rhythm | 12.95 s | 0.0066 | 0.0115 |

The precision workload (rhythm, search 1–4 / val 5) gave T_ref 8.71 s and tolerance 0.0246.

**Rhythm, fold-7 development assessment (5 paired seeds; macro AUROC):**

| Model | Per seed | Mean | Gain over untuned (hierarchical 95% CI) |
|---|---|---|---|
| Untuned reference | 0.9447 0.9431 0.9152 0.9342 0.9259 | 0.9326 | — |
| P (adaptive_utility) | 0.9520 0.9511 0.9367 0.9608 0.9583 | 0.9518 | +0.0192 [0.0058, 0.0332] |
| DEHB (native) | 0.9503 0.9661 0.9550 0.9585 0.9656 | 0.9591 | +0.0265 [0.0104, 0.0410] |

- **P − DEHB:** −0.0073 (hierarchical CI [−0.0212, +0.0068]). Per seed: +0.0017, −0.0150, −0.0183, +0.0022, −0.0073.
- **Search cost:**
  - P stopped by its own rule in 5 of 5 runs, at 47–83% of the cap.
  - DEHB always used the full cap.
  - P/DEHB accounted-time ratio: 0.57 (CI 0.49–0.70).
- **Equal-compute check (validation fold 5, from the trajectories):** at the moment P stopped, DEHB's best validation score was on average the same as P's (difference ≈ 0.0001). DEHB's remaining compute added about +0.002 validation AUROC.

**Matched-control smoke:** all 8 runs completed; six budget-limited runs overshot by 0.082–0.113 s. At 80 T_ref, the utility stop never fired. The patience stop fired in 1 of 2 seeds for both allocation rules. This was a functional check; no efficacy reading is made.

## 22.3 What this means for the design

1. **Tuning pays on rhythm.** Both tuners beat the untuned model, with CIs excluding zero. This replicates the local headroom result (doc 21) on G4, on a different search/assessment split.
2. **P used less time and had a lower point estimate of assessment quality.** The ratios of mean gain and geometric mean search time were about 72% and 57%, respectively; these are descriptive, not a dominance result. Equal-cost validation differed by a mean +0.00011 AUROC. Every DEHB configuration at P's stop differed from its final configuration; those five earlier configurations still require independent refits before comparing equal-cost assessment quality. The current P-DEHB interval includes zero, so superiority or inferiority is not established.
3. **Non-inferiority is not established; it is poorly supported as the primary question.**
   - The paired seed SD of P − DEHB is 0.0094. Training-seed noise alone moves the untuned model by 0.03 AUROC.
   - The planning projection gives a one-sided half-width of about 0.0085 even with 30 seeds. If P and DEHB were truly equal, the chance of showing non-inferiority would be 25% at a 0.005 margin and 13% at 0.0025.
   - The reproduced one-sided 95% lower bound is -0.01870. It fails all proposed margins (0.002, 0.0025, 0.005). The five-pair projection is an unvalidated development diagnostic, not proof that NI is impossible on rhythm or a power guarantee for fold 10.
   - **Assessor consequence:** focus the primary study on final assessment/test gain at declared search costs, supported by validation anytime and censored time-to-target analyses. Do not substitute validation quality for final quality. Retain NI only with a separately justified, frozen secondary margin.
4. **The T_ref definition must be fixed in the protocol.**
   - Superclass T_ref came out at 11.17 s here, against 9.79 s in the 13 Oct pilot.
   - The drift check, which uses the same configurations, shows the GPU speed unchanged.
   - Both the configuration set and the training seed changed from the original pilot. Their differing evaluation durations and early-stopping histories can change the median independently of hardware drift.
   - `task_calibration.yaml` already specified one common 16-configuration design seed and one training seed before this run. The realized configurations and measured task units can now be recorded exactly. No additional timing run is needed merely to freeze that existing definition.
5. **Confirmation is active; its tolerance is a heuristic rather than an accuracy-loss guarantee.**
   - Rhythm's re-evaluation tolerance is 0.0115 on the common workload and 0.0246 on the precision workload.
   - Across the exported runs there were 13 confirmation attempts: seven confirmed stops, three improvement-probe vetoes and three stability rejections. The probe vetoes occurred in the 160-unit P runs; stability rejections occurred in the smoke runs (deviation 0.04819 > tolerance 0.0246).
   - The revised P probes veto any incumbent improvement. Tolerance governs a separate seed-stability check. Its large value does not disable the probe veto. All five final P re-evaluations passed, which does not establish sensitivity to smaller quality losses.
   - Keep the existing rule for the matched follow-up. Changing it now would create another method version and obscure the comparison; do not divide a selected-incumbent threshold by sqrt(n) without a justified new procedure.
6. **Seed budget.** The development projection suggests that patient uncertainty can limit interval narrowing; five pairs cannot establish its magnitude reliably. Select replication for paired frontier estimation. Do not claim that more seeds cannot help or that this projection validates non-inferiority power.

## 22.4 Open items before G2

- [ ] Complete equal-cost held-out development refits and matched stopping comparisons at 160 units.
- [ ] Finalize the primary final-quality/cost estimands and analysis plan before G2.
- [ ] Preserve the seven-label rhythm definition; a fresh >50-count filter on folds 1–7 would add AFLT and BIGU and change the tested task.
- [ ] Replace the oversized draft inventory with a budgeted design using the measured task units.
- [ ] Select replication for paired estimation; do not treat the five-pair projection as validated power.

Independent assessor verification: `docs/evidence/g4_decision_audit.json`. It reproduces all three
main contrasts from the saved predictions and verifies the executed source bundle, registered plan,
227 exported hashes and 15 prediction hashes. The prior literature record is unchanged: ECG-DE was
checked at abstract level; SHCHO's GPU-day comparisons were inspected in the author preprint in
doc 20. No further literature retrieval is required for the current experimental work.

## 22.5 Process note

The 14 Oct notebook ran its scripts through `subprocess.run`, whose output the VS Code Colab kernel does not forward, so no progress was visible while it ran. `scripts/build_colab_notebook.py` now relays child output line by line and also writes it to `logs/` on Drive. The original executed bundle (`21dce7d4…`) remains preserved. New source bundles have different plan hashes and must use separate output roots; the matched follow-up explicitly verifies and reuses the original results.
