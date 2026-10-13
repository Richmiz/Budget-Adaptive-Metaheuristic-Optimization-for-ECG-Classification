# Protocol v1.1 (post-pilot) — consolidated for assessment

**Study:** Budget-Adaptive Metaheuristic Optimization for ECG Classification Using Marginal Utility-Based Search Allocation (Article 1)
**Version:** 1.1, 13 October 2026. Supersedes proposal v1.0 (7 October 2026) where they differ; all differences are listed in §7.
**Assessor disposition:** v1.1 is retained as history. The researcher authorized the decision
phase in doc 18; doc 19 is the revised design draft. The 300-run matrix is not approved to start.
**Status:** pilot complete on the confirmatory hardware; **protocol not yet frozen**. Gate G2 is pending the open decisions in §9 and the timing drift check. No confirmatory run has started and the test fold (PTB-XL fold 10) has never been read.
**Detail:** this document summarizes docs 00–15; section numbers in brackets point to them.

---

## 1. The study in one paragraph

Metaheuristic hyperparameter optimization is widely used for biomedical deep learning, but its gains are rarely reported against its computational cost; the project's systematic review found a favourable gain–cost trade-off in only 10 of 44 structured studies. This study tests a controller that allocates Differential Evolution (DE) search time across blocks of CNN hyperparameters according to observed validation improvement per unit of measured compute, and stops the search after a confirmation step. It is compared with six established methods under matched time caps on patient-separated PTB-XL data. The question is whether the controller reduces search time without loss of test performance, and which of its components are responsible.

## 2. Research questions and hypotheses [00, 07]

| RQ | Hypothesis | Primary contrast | Primary outcome |
|---|---|---|---|
| RQ1 | **H1:** adaptive allocation (without stopping) improves anytime efficiency over standard DE and round-robin allocation | `mu_nostop` vs `de`, `rr_nostop` at B_mid | Anytime efficiency (validation) |
| RQ2 | **H2 (primary):** the full controller uses less search time than DEHB and is non-inferior in test macro AUROC | `mu` vs `dehb` at B_mid | One-sided 95% lower bound of ΔAUROC > −margin; search-time ratio |
| RQ3 | **H3:** removing mandatory probes or the joint block increases premature stopping or lowers performance | `mu_noprobe`, `mu_nojoint` vs `mu` | Premature-termination rate; test AUROC |
| RQ4 | Descriptive: stability across seeds and budgets | `mu` at 3 budgets | SD across seeds; stop times |
| **Added (v1.1)** | Does the time saving come from the controller, or from the minimum-time floor or a trivial stopping rule? | `mu` vs `de_patience`; `mu_floor20` / `mu` / `mu_floor60` | Search-time ratio; test AUROC; budget used vs floor |

## 3. Design [02, 03, 05]

**Data.** PTB-XL v1.0.3, 100 Hz, 12 leads × 1,000 samples, five diagnostic superclasses (multi-label, official benchmark mapping). 21,799 → **21,388 records** (411 without a diagnostic superclass), **18,617 patients**, **0 patients in more than one fold**, 0 duplicate waveforms. Every downloaded file was verified against PhysioNet's SHA256SUMS.

| Partition | Train (folds) | Validation | Test | Records (train / val / test) |
|---|---|---|---|---|
| pilot | 1–6 | 7 | — | 12,821 / 2,134 |
| search | 1–7 | 8 | — | 14,955 / 2,129 |
| final | 1–8 | 9 | 10 (locked) | 17,084 / 2,146 / 2,158 |

**Model and search space.** 1-D ResNet (three residual groups, global pooling, five sigmoid outputs), trained with AdamW, BCE and mixed precision. Nine hyperparameters in blocks:
* **A:** learning rate, batch size
* **B:** weight decay, dropout
* **C:** width, depth, kernel, dilation, stem stride
* **J:** joint, all nine together

Candidate evaluation: up to 20 epochs, patience 5, fitness = best validation macro AUROC. Multi-fidelity methods use 5/10/20 epochs.

**Methods.** Proposed controller P (`mu`); random search; TPE (Optuna); standard DE; Hyperband; BOHB; DEHB (official package); untuned reference CNN. Ablations: `mu_nostop`, `rr_nostop`, `mu_nocost`, `mu_noprobe`, `mu_nojoint`, `mu_noconfirm`. Stop-rule checks (v1.1): `mu_floor20`, `mu_floor60`, `de_patience`.

**Budgets.** Time caps in units of T_ref (the measured median evaluation time on the confirmatory GPU). With **T_ref = 9.8 s**:

| Budget | Units | Wall time per search |
|---|---|---|
| B_low | 80 | 13.1 min |
| B_mid | 160 | 26.1 min |
| B_high | 320 | 52.3 min |

**Run inventory.** Seeds 1001–1010, paired across methods (same 8-point initial design and training seed per seed).

| Block | Runs |
|---|---|
| Main: 7 methods × 3 budgets × 10 seeds | 210 |
| Ablations: 6 × B_mid × 10 | 60 |
| Stop-rule checks: 3 × B_mid × 10 | 30 |
| **Total search runs** | **300** |

Each run's selected configuration is retrained on folds 1–8 (≤ 50 epochs, checkpoint and thresholds on fold 9) and evaluated **once** on fold 10 after the analysis lock.

**Hardware.** Every timed run uses one GPU type: Colab **G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition)**. The runner refuses any other GPU. The local RTX 3050 is used for development only.

## 4. The controller [04]

Each round runs two DE trials restricted to one block. The reward is the improvement of the global incumbent per normalized second, r = ΔM / (ΔT / T_ref). Blocks are scored with an optimistic bandit-style index, U = mean(r) + κ·σ·√(2 ln(1+t) / (1+n)), over a per-block rolling window of 8 rounds. Actions:
* **warm-up:** 3 rounds per block
* **continue / switch:** with hysteresis
* **mandatory probe:** a block unvisited for 8 decisions
* **stop proposal:** all U < τ for 3 decisions, each block ≥ 3 rounds, **≥ 40% of the cap used**
* **confirmation:** one more round per block, plus re-training the incumbent with 2 fresh seeds (tolerance 0.0035)

Pilot-selected settings: **τ = 5×10⁻⁴, κ = 1**.

## 5. Outcomes and analysis [07]

* **Primary (H2):** paired difference in test macro AUROC, P − DEHB, at B_mid. Hierarchical bootstrap over seeds and patients (10,000 replicates); non-inferiority if the one-sided 95% lower bound exceeds −margin (**provisional 0.005; see §9**). Efficiency claim only if the one-sided 95% upper bound of the P/DEHB search-time ratio is also < 1.
* **H1:** paired anytime efficiency, seed bootstrap and Wilcoxon, Holm-adjusted across the two contrasts.
* **H3:** premature-termination rate and test AUROC versus P.
* **Stop-rule checks (§7.5b):** P vs `de_patience` on time and AUROC; budget used versus floor. **H2 is interpreted together with these.** A saving that disappears against the naive stopper, or follows the floor, does not support a claim that the controller causes efficiency.
* **Secondary:** all methods and budgets; Friedman ranks; deployment latency; threshold metrics.
* Missing runs: restarted from scratch up to twice, then reported. The analysis plan is hashed at the lock.

## 6. Evidence collected so far

| Check | Result | Where |
|---|---|---|
| Code | 64 automated tests pass, locally and on the G4 (54 at the time of the G4 run); the TPE and DEHB adapters were checked against the installed library versions | doc 13 |
| Reproducibility | Data rebuilt on Colab is **byte-identical** to the local arrays (SHA-256); pilot export of 81 files verified against its manifest | doc 13 |
| Speed | G4 is about 5× faster than the RTX 3050; **T_ref = 9.8 s** | pilot P2 |
| Seed noise (validation AUROC) | SD **0.0012** (G4, 4 settings × 5 seeds); 0.0016 (local, 6 settings × 4 seeds) | P3; headroom check |
| Tuning headroom (local, fold 7) | Untuned reference **0.9351** (mean of 5) vs best tuned **0.9387** (mean of 4). The best setting beats top-quarter settings by **0.0032 = 2.0 × seed SD** (borderline) | docs 13–14 |
| All methods on real data (P6, 20 T_ref) | All 7 within the cap; multi-fidelity methods use {5, 10, 20} epochs | pilot |
| τ/κ sweep (P4, B_mid, 2 seeds each) | τ 1e-4: AUROC 0.9423, 78% of budget used; τ 5e-4 / 1e-3 with κ 0.5–2: AUROC 0.9408–0.9411, **48–53%** of budget used | pilot |
| **Stop timing (decision logs)** | Warm-up ends at ~26% of the cap. In 6 of 8 stopping runs the controller's own rule allowed stopping at 31–38%, but the stop was proposed only at the **40% floor**. After warm-up the incumbent improved 0–1 times per run. Confirmation never reversed a stop | doc 13 |

## 7. Changes from proposal v1.0

| ID | Change | Reason |
|---|---|---|
| D38 | Colab G4 fixed as the single confirmatory GPU | Budgets are wall-clock caps in T_ref units |
| D32, P2 | T_ref = 9.8 s (median of 16 evaluations on G4) | Pre-specified pilot rule |
| D21, P3 | Re-evaluation tolerance = 0.0035 (95th percentile of seed deviations) | Recommended rule (final confirmation in §9) |
| P4 | τ = 5×10⁻⁴, κ = 1 kept (**post-pilot amendment**) | The saved pre-specified rule selected **τ = 1×10⁻³** (lowest budget used). Keeping 5×10⁻⁴ relies on a "< 5 points of budget = tie" threshold defined **after** seeing the pilot. Doc 06 said ties go to the defaults but did not define a tie. Recorded as an amendment, not a pre-specified choice |
| D42 | Original design retained after critical design and evidence reviews | Researcher's decision (13 Oct 2026); risks recorded in §8 |
| **D43** | Floor sensitivity: P with 20% and 60% minimum-time floors (20 runs) | Pilot showed the stop time follows the 40% floor |
| **D44** | Naive stopping baseline `de_patience`: standard DE, stop after 16 evaluations without improvement, same 40% floor, no confirmation (10 runs) | Separates the controller's contribution from "stop when progress flattens" |
| — | Training loop: no host–device synchronization per step; fused AdamW | Pure speed-up (~20%), applied to every method before T_ref was measured |

## 8. Known limitations and risks (stated openly)

1. **Small headroom on this task.** Good settings differ by ~0.003 AUROC (≈ 2 × seed SD), so final-accuracy differences between methods are expected to be small. The non-inferiority test therefore has limited ability to discriminate unless the margin is tightened (§9).
2. **Novelty is integrative, not mechanistic.** Every component has published precedent:
   * contribution-based allocation across variable groups instead of round-robin (Omidvar, Li & Yao, 2011; Jia et al., 2019)
   * bandit selection of variable groups (De Rainville et al., 2013)
   * bandit operator selection (Fialho et al., 2009)
   * improvement-per-time acquisition (Snoek et al., 2012)
   * automatic HPO termination (Makarova et al., 2022)

   The claim is limited to the integrated controller and its controlled evaluation for ECG deep learning [15].
3. **Stop time and the floor.** Pilot logs show the 40% floor sets most stop times; D43–D44 test how much of the saving remains when this is controlled.
4. **System-level comparison.** P (single-fidelity, block DE) vs DEHB (multi-fidelity DE) compares complete systems. Component attribution relies on the ablations, which share P's engine and evaluator.
5. **BOHB is an in-house reimplementation.** It has been checked functionally, but not yet against the official hpbandster implementation (D26).
6. **Generalization:** one dataset, one task, one CNN family, and internal (patient-separated) validation only. No clinical claim.
7. **Cloud timing.** Colab may vary between sessions; a timing drift check (session 2) is pending before freezing T_ref.

## 9. Decisions required before freezing (G2)

| ID | Decision | Recommendation | Evidence |
|---|---|---|---|
| D30 | Non-inferiority margin | **Tighten from 0.005 to ~0.002–0.0025** (≈ 2 × seed SD) | Margin 0.005 exceeds the measured top-end headroom (0.0032) |
| D31 | Cost criterion for the efficiency claim | One-sided 95% upper bound of the P/DEHB time ratio < 1 **and** consistent with §7.5b | Pilot stop timing |
| D21 | Re-evaluation stability rule | Keep q95 rule (0.0035) | P3 |
| D26 | BOHB implementation | Run a synthetic comparison with hpbandster, or wrap the official implementation | Not yet checked |
| D03 | Class weighting | Unweighted BCE | Keeps comparisons about search |
| D05 | Untuned reference configuration | As in `configs/search_space.yaml` (lr 1e-3, batch 64, wd 1e-2, dropout 0.2, mid-size network) | Scored 0.9351 in the pilot |
| D07 | Mixed precision | Keep (applied identically to all methods) | Used in all pilot timings |
| — | Timing drift check (session 2) | Run once in a new G4 runtime; freeze T_ref if within ±10% | Pre-specified (doc 06) |

## 10. Compute and timeline

* **Remaining compute (estimate, not a guaranteed bound; drift check pending):** about 54,330 T_ref ≈ **148 G4-hours ≈ 1,320 compute units (≈ $130)**:
  * main 107 h
  * ablations 26 h
  * stop-rule checks ≤ 13 h
  * final training ~2 h

  Methods that stop early use less.
* **Sequence:**
  1. Decisions (§9) and drift check
  2. `lock_protocol.py freeze` (G2)
  3. Main, ablation and stop-rule runs (~1–2 weeks of Colab sessions)
  4. Final models
  5. Analysis lock and one fold-10 evaluation (G4)
  6. Analysis and manuscript

  At this compute scale, about 3–4 months to submission is realistic.

## 11. Reproducibility

* Code is embedded in each Colab notebook as a SHA-256-checked bundle.
* Data are rebuilt on the runtime and asserted byte-identical.
* Every run records: config hash, hardware fingerprint, evaluation log, controller decision log, and anytime trajectory.
* The protocol is frozen to `protocol/frozen_protocol.yaml` (G2), and `protocol/analysis_lock.json` is created before fold 10 is read (G4).
* Library versions are pinned (`requirements-lock.txt`).

## References

De Rainville, F.-M., Sebag, M., Gagné, C., Schoenauer, M., & Laurendeau, D. (2013). *Sustainable cooperative coevolution with a multi-armed bandit*. arXiv. https://doi.org/10.48550/arxiv.1304.3138
Fialho, Á., Da Costa, L., Schoenauer, M., et al. (2009). Dynamic multi-armed bandits and extreme value-based rewards for adaptive operator selection in evolutionary algorithms (pp. 176–190). Springer. https://doi.org/10.1007/978-3-642-11169-3_13
Jia, Y.-H., Chen, W.-N., Gu, T., et al. (2019). Distributed cooperative co-evolution with adaptive computing resource allocation for large scale optimization. *IEEE Transactions on Evolutionary Computation, 23*(2), 188–202. https://doi.org/10.1109/tevc.2018.2817889
Makarova, A., Shen, H., Perrone, V., et al. (2022). *Automatic termination for hyperparameter optimization*. arXiv. https://doi.org/10.48550/arxiv.2104.08166
Omidvar, M. N., Li, X., & Yao, X. (2011). Smart use of computational resources based on contribution for cooperative co-evolutionary algorithms (pp. 1115–1122). ACM. https://doi.org/10.1145/2001576.2001727
Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms*. arXiv. https://doi.org/10.48550/arxiv.1206.2944
Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics, 25*(5), 1519–1528. https://doi.org/10.1109/jbhi.2020.3022989
Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). *PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
