# 08 — Decision register

Current status (16 Oct 2026): v2 execution and the locked test analysis are complete.
Older rows preserve the decisions at their recorded dates. Current reporting is in docs 27–29.

Every place where the proposal is silent or ambiguous, and every value that must be fixed before the confirmatory benchmark. **Status:**
* **Adopted:** implemented as default; change only with a reason.
* **Needs your call:** a scientific decision for the researcher, with a recommendation.
* **Pilot:** fixed by a pre-specified rule in doc 06.
* **Frozen:** locked at G2.

After the freeze, changes are allowed only as logged amendments (bottom of this file).

## Needs your call (before G2)

| ID | Decision | Recommendation |
|---|---|---|
| D03 | Class weighting | Unweighted BCE in the primary study (isolates the search method) |
| D05 | Reference (untuned) configuration | lr 1e-3, batch 64, wd 1e-2 (the AdamW default), dropout 0.2, width 1, 2 blocks/group, kernel 9, dilation 1, stem stride 2 |
| D07 | Mixed precision (fp16 autocast) | Yes, for all methods (usually faster on tensor-core GPUs; confirm the speed-up and unchanged AUROC in P1); matched across methods |
| D21 | Re-evaluation stability rule and what happens after an unstable re-evaluation | 95th-percentile rule (doc 06 P3); block termination until the incumbent changes |
| D26 | BOHB implementation | Keep the in-house version only if it matches hpbandster on the synthetic check; otherwise wrap the official one |
| D30 | Non-inferiority margin justification | Keep 0.005 if it is no larger than the P3 pooled seed SD of validation AUROC and smaller than the differences between architectures reported in the PTB-XL benchmark; document both numbers |
| D31 | Cost criterion for the efficiency claim | One-sided 95% upper bound of the P/DEHB time ratio < 1 |
| D41 | Parallel jobs on one G4 and how cost is measured | Decide after session 1's concurrency probe. If the gain is large, use **profiled cost**: each architecture's epoch time measured once, alone, on the G4; job clocks advance by profiled cost per epoch actually trained. Jobs can then run in parallel with noise-free, reproducible budgets. If the gain is small, keep one job at a time with wall-clock cost |

## Register

| ID | Topic | Proposal text / gap | Decision | Status | Where |
|---|---|---|---|---|---|
| D01 | Label mapping | "official benchmark mapping" | Diagnostic statements → `diagnostic_class`, no likelihood threshold; drop records without a superclass | Adopted | `data/ptbxl.py` |
| D02 | Policy-development partitions | "additional patient-separated partitions within folds 1–7" | Pilot = train folds 1–6, validate fold 7 | Adopted | `base.yaml` |
| D03 | Class weighting | "if adopted… from training records" | See above | Needs your call | `base.yaml` |
| D04 | Normalization | "estimated exclusively from the relevant training partition" | Per-lead z-score from the training role of each partition | Adopted | `data/dataset.py` |
| D05 | Reference configuration | Not specified | See above | Needs your call | `search_space.yaml` |
| D06 | Fixed architecture details | "fixed stride rules" | Stem kernel 15; widths rounded to a multiple of 8; dilation on all residual convs; group strides 1/2/2; 1×1 projection shortcuts | Adopted | `resnet1d.py` |
| D07 | Mixed precision | Not specified | See above | Needs your call | `base.yaml` |
| D08 | Training seed policy | Not specified | One candidate-training seed per search seed, shared by all methods and constant within a run | Adopted | `runner.py` |
| D09 | Shared initial design cost | "evaluation costs charged to each method" | Evaluate once per seed; reuse via cache; **charge the recorded cost** to every method's clock | Adopted | `tracker.py` |
| D10 | Use of the shared design | Not specified | Random: first 8; DE/P: initial population; TPE: enqueued start-up trials; HB: incumbent candidates; BOHB: 20-epoch model observations; DEHB: incumbent only (cannot ingest) | Adopted | methods |
| D11 | Duplicates | "cached evaluations… only when identical" | Run-level cache for identical (config, seed, partition, fidelity); DE trials identical to their target are regenerated (≤ 10 tries), then one active coordinate is forced to change | Adopted | `controller.py` |
| D12 | DE bound handling | Not specified | Bounce-back: violated coordinate → midpoint of target value and bound | Adopted | `de.py` |
| D13 | DE target selection | Not specified | Cyclic over the population | Adopted | `de.py` |
| D14 | Warm-up order | "three initial rounds to each of A, B, C, J" | Interleaved A, B, C, J × 3 (reduces time-drift confounding) | Adopted | `controller.py` |
| D15 | Rolling window | "at most eight allocation rounds" | **Per block**: last 8 rounds of that block; t is the global round count | Adopted | `controller.py` |
| D16 | SD estimate | Not specified | Sample SD (ddof = 1); 0 if n < 2; then the σ_floor floor | Adopted | `controller.py` |
| D17 | What earns reward | "best observed fitness after round t" | Improvement of the **global** incumbent only | Adopted | `controller.py` |
| D18 | Round duration | "actual round duration" | From the end of the previous round to the end of this one (includes decision overhead) | Adopted | `controller.py` |
| D19 | Stop streak | "below τ for three decisions" | Counted after post-warm-up decisions only; reset after a failed confirmation | Adopted | `controller.py` |
| D20 | Confirmation probes | "additional round in each block" | Fixed order A, B, C, J; exit on the first rate ≥ τ; probe rewards enter block statistics but not the streak | Adopted | `controller.py` |
| D21 | Re-evaluation stability | "pilot-defined stability criterion" | max \|m_i − M*\| ≤ tolerance; tolerance from P3; after failure, the incumbent is flagged and stopping is blocked until it changes | Needs your call → Pilot | `controller.py`, doc 06 |
| D22 | No time for confirmation | "recorded as budget exhaustion" | Keep searching until the cap; reason = `budget`; time of infeasibility logged | Adopted | `controller.py` |
| D23 | `mu_nocost` ablation | "prioritize improvement alone" | Δc = number of evaluations in the round (keeps τ's scale) | Adopted | `controller.py` |
| D24 | Hypothetical stop in no-stop runs | "identify the hypothetical stopping time" | First time the stop-proposal condition holds (without confirmation); incumbent snapshot saved for final training | Adopted | `controller.py` |
| D25 | Multi-fidelity details | Fidelities 5/10/20 | η = 2; every rung retrains from scratch; patience 5 at every fidelity; incumbent only from 20 epochs | Adopted | `hyperband.py`, `runner.py` |
| D26 | Hyperband/BOHB implementation | Not specified | In-house, inside the shared tracker; BOHB checked against hpbandster | Needs your call | `hyperband.py` |
| D27 | Library versions | — | Pin optuna, dehb, ConfigSpace, torch in `requirements-lock.txt` before P6 | Adopted | — |
| D28 | TPE settings | — | Optuna defaults except n_startup_trials = 8 (= shared design); univariate | Adopted | `methods.yaml` |
| D29 | Anytime curve before the first result | Not specified | 0.5 (chance AUROC) | Adopted | `anytime.py` |
| D30 | Non-inferiority margin | "provisional 0.005… justification documented" | See above | Needs your call | `base.yaml` |
| D31 | Efficiency criterion | "reduced cost" | See above | Needs your call | doc 07 |
| D32 | T_ref measurement | "training-only pilot" | Pilot partition (folds 1–6 → 7), frozen GPU, 16 LHS configurations, median; no size correction (T_ref is a unit, not a forecast) | Adopted | `run_pilot.py` |
| D33 | Threshold selection | "fixed procedure" | Per label, grid 0.01–0.99, maximize F1, ties → lowest | Adopted | `metrics.py` |
| D34 | Diverged training | Not specified | Status `diverged`; the best earlier checkpoint stays valid if any; else invalid | Adopted | `trainer.py` |
| D35 | Out of memory | "infeasible candidates… no reward" | Status `infeasible`; time charged; no reward | Adopted | `training/evaluator.py` |
| D36 | Execution order | "randomized" | Full matrix shuffled with `order_seed`; Colab shards by seed | Adopted | `runner.py` |
| D37 | Duplicate waveforms | Not specified | Report only. Data check on 8 Oct 2026 found **0** byte-identical waveform groups, so no action is needed | Resolved | `prepare_data.py` |
| D38 | Compute platform | "one GPU type" | Colab via VS Code, **G4 (NVIDIA RTX PRO 6000 Blackwell)**, chosen by the researcher on 8 Oct 2026; one GPU type for T_ref and all confirmatory runs; the runner and timed pilot stages refuse other GPUs (`hardware.required_gpu = "RTX PRO 6000"`); results on Drive. Session 1 confirmed "NVIDIA RTX PRO 6000 Blackwell Server Edition" | Adopted | doc 10, `hardware.py` |
| D39 | Interrupted jobs | Not specified | Restart from scratch (a search cannot resume mid-way on a wall clock); ≤ 2 retries | Adopted | `runner.py` |
| D40 | Selected model vs re-evaluation | Not specified | The returned configuration is the single-seed incumbent; re-evaluations only gate stopping | Adopted | `controller.py` |
| D41 | Parallel execution vs wall-clock budgets | Proposal: "search-time limit on a dedicated GPU", cost = "actual round duration" | Options: (a) one job at a time, live wall clock (as proposed; G4 mostly idle, highest cost); (b) K concurrent jobs, wall clock under a constant load K (cheaper; contention noise in the cost signal the controller relies on); (c) profiled cost per architecture × epochs actually run, measured alone on the G4, plus measured controller CPU time (parallel-safe, reproducible; documented deviation from live wall time) | Needs your call (after session 1) | doc 10, `budget.py` |
| D43 | Minimum-time floor sensitivity | Proposal: stop only after "at least 40% of the time limit" | P also run with floors 0.2 and 0.6 at B_mid (`mu_floor20`, `mu_floor60`, 10 seeds each); analysis doc 07 §7.5b | Adopted 13 Oct 2026 (researcher chose B + C) | `methods.yaml`, `experiments/stopping.yaml` |
| D44 | Naive stopping baseline | Not in proposal | `de_patience`: standard DE, stop after 16 evaluations (2 generations) without incumbent improvement, never before 40% of the cap; no confirmation; B_mid, 10 seeds | Adopted 13 Oct 2026 (researcher chose B + C) | `methods/simple.py`, `methods.yaml` |

## Frozen values (filled at G2)

| Key | Value | Source | Date |
|---|---|---|---|
| `hardware.required_gpu` | "RTX PRO 6000" (G4; confirmed: NVIDIA RTX PRO 6000 Blackwell Server Edition) | D38, session 1 | 2026-10-08 |
| `budgets.t_ref_seconds` | 9.8 (proposed; drift check pending) | Pilot P2, 13 Oct | — |
| `controller.reeval_tolerance` | 0.0035 (proposed) | Pilot P3, 13 Oct | — |
| `controller.tau` / `kappa` | 5e-4 / 1 — **post-pilot amendment**: the saved pre-specified rule chose 1e-3; the tie threshold (< 5 budget points) was defined after the pilot | Pilot P4, 13 Oct | — |
| Search domain edits | none yet | P1 | — |
| Search seeds | 1001–1010 | P5 | — |
| Non-inferiority margin | 0.005 (provisional) | D30 | — |

## Development amendments (G2 freeze pending)

| Date | ID | Change | Reason | Impact |
|---|---|---|---|---|
| 2026-10-13 | D42 (new) | **Proceed with the original Article 1 design** and continue the pilot on Colab G4 | Researcher's decision after the design review (doc 14) and evidence review (doc 15), which recommended reframing. The risks those docs record stand: small headroom on PTB-XL superclasses, H2 asymmetry, novelty precedent. They are to be revisited at G2 with the pilot results | No change to the protocol |
| 2026-10-13 | D43, D44 | Added floor-sensitivity runs and a naive stopping baseline (30 runs at B_mid) | Pilot decision logs showed the stop time follows the 40% floor | +30 runs, ≤ 4,800 T_ref (≈ 6–13 h on G4) |
| 2026-10-13 | P4 | τ kept at 5e-4 although the saved pre-specified selection chose 1e-3 | A tie threshold (< 5 budget points) was defined after observing the pilot; flagged by the assessor | Not pre-specified; disclose as an amendment |
| — | — | — | — | — |

## Assessor decision phase — 13 October 2026

The researcher requested continuation of the assessor's recommended work. This authorizes
development implementation and pilots; it does not constitute approval of a confirmatory
v1.2 protocol. D42-D44 above record the earlier design decision; the following assessment now
holds the full matrix while the evidence is gathered.

| ID | Decision | Status / reason | Artifact |
|---|---|---|---|
| D45 | Preserve post-pilot v1.1 before revisions | Completed: baseline commit `feff26a`; old results retained | Git; pilot export audit |
| D46 | Register form/rhythm headroom screen and fresh-seed confirmation before training | Completed 115 evaluations: rhythm GO, form BORDERLINE; folds 1-6/7 only, fixed winner and planning gates | docs 18, 21; `decision_pilot.yaml` |
| D47 | Replace weak cross-engine patience attribution with matched allocation x stop-proposal controls | Implemented and synthetic-tested; common any-improvement confirmation is a declared change from legacy P | doc 18; `methods.yaml` |
| D48 | Use DEHB native initialization as the external system comparator | Implemented; real-data fairness sensitivity remains development work | `dehb_native` |
| D49 | Report actual time and deadline overshoot; reject late incumbent updates | Implemented after finding legacy clamping; old summaries cannot certify exact cap compliance | tracker and runner |
| D50 | Do not approve NI margin from headroom or seed SD alone | Pending acceptable-loss rationale and paired system/patient precision; v1.2 draft emphasizes gain versus compute | doc 19; precision pilot |
| D51 | Hold confirmatory execution and freeze during decision phase | Drift, calibration and precision completed and verified; matched follow-up, final inventory and analysis definitions remain pending | `decision_status.yaml`; docs 22–23 |
| D52 | Reassess novelty with primary sources | Targeted verification completed; SHCHO is a close deep-CNN coevolution precedent. Researcher requested accepting the supplied review numbers and prioritizing experiments; no further review audit is an execution gate | doc 20; doc 19 |

## Assessor decisions after verified G4 development — 14 October 2026

These are development amendments and proposals, not approval of a frozen confirmatory protocol.

| ID | Decision | Status / reason | Artifact |
|---|---|---|---|
| D53 | Verify the G4 export and reproduce estimates from saved predictions | Completed: 227 files, 15 predictions, three paired contrasts reproduced; P/DEHB time ratio 0.572, assessment difference −0.00733; NI unestablished | docs 22–23; `g4_decision_audit.json` |
| D54 | Complete equal-cost assessment and matched stopping mechanisms before G2 | Prepared, not run: 15 searches at original settings/seeds, up to 30 additional refits; source and data identity pinned; original evidence preserved | `matched_followup.yaml`; `COLAB_G4_MATCHED_FOLLOWUP.ipynb` |
| D55 | Replace oversized draft with one 160-unit cap and declared prefix assessments | Proposed: 180 searches, measured 96.50 search-cap hours; refits, pilots and reserve still need reconciliation before G2 | doc 19 |
| D56 | Preserve the seven development-derived rhythm labels | Proposed for final freeze: refiltering on folds 1–7 adds two classes and changes the tested target; existing folds 1–6 vocabulary remains training-derived | `calibration_definition_v1.json`; docs 19, 23 |
| D57 | Retain measured calibration definitions and confirmation for the follow-up | Realized 16 configurations and seeds recorded; confirmation produced three probe vetoes and three stability rejections. No repeat calibration or tolerance retuning for this comparison | docs 22–23 |

## Reporting addendum — 16 October 2026

| ID | Decision | Status / reason | Artifact |
|---|---|---|---|
| R01 | Correct the interpretation and current status after the v2 audit | User authorized report corrections and figure preparation; remove equivalence/NI and component-attribution wording unsupported by the completed design | docs 27, 29; README |
| R02 | Add paired 160-minus-80 and absolute-score intervals for reporting | Defined after test inspection, using all six task–tuner budget contrasts, existing predictions, 10,000 patient/paired-seed draws, no new confirmatory tests; preserve original lock and primary family | `reporting_supplement_v2.py`; supplemental plan and estimates |
| R03 | Prepare figures separately from locked scientific analysis | Frozen primary/secondary results and clearly labelled supplemental intervals; PDF/SVG/PNG, plotted CSV values, captions and hash manifest | `prepare_publication_figures.py`; `figures/publication_v2/` |
