# Article 1 — Claims brief for the assessor (13 Oct 2026)

Full protocol: `docs/16_protocol_v1_1.md`. Evidence review with literature: `docs/15_evidence_review.md`.

## 1. The claim the study will test

> A controller that allocates Differential Evolution search time across hyperparameter blocks according to measured validation improvement per second, and stops after a confirmation step, finds ECG classifiers that are **not worse** (test macro AUROC) than those found by DEHB while using **less search time**, and this saving comes from the controller's decisions rather than from simply being allowed to stop.

This is a hypothesis, not a result. The claim is made only if all three hold:
1. **Non-inferiority:** the one-sided 95% lower bound of (P − DEHB) test macro AUROC exceeds −margin (margin to be fixed before the freeze; ~0.002–0.0025 recommended).
2. **Time:** the one-sided 95% upper bound of the P/DEHB search-time ratio is below 1.
3. **Attribution:** P beats a naive "stop after 16 evaluations without improvement" DE with the same 40% minimum time, and P's saving does not simply follow the minimum-time floor (20/40/60% sensitivity runs).

If any condition fails, the paper reports the negative or mixed result.

## 2. Why the question is worth testing

* The project's systematic review found a favourable gain-versus-compute trade-off in only **10 of 44** structured studies of metaheuristic optimization in biomedical deep learning; 19 were uncertain. Search cost is rarely measured against the gain.
* This study measures cost directly, under one fixed GPU type, matched time caps, patient-separated data and a pre-registered analysis, with the test fold locked until the end.

## 3. What is already established (evidence in hand)

| Item | Evidence |
|---|---|
| Data integrity | PTB-XL v1.0.3: 21,388 records, 18,617 patients, **0 patients across folds**. Every file checked against PhysioNet SHA-256 sums |
| Reproducibility | Data rebuilt on Colab is byte-identical to the local preparation; the pilot export of 81 files is hash-verified; 64 automated tests pass |
| Hardware and time unit | Colab G4 (RTX PRO 6000 Blackwell); T_ref = **9.8 s** per evaluation |
| All 7 methods work on real data | Each stayed within its time cap in the pilot |
| Measurement noise | Seed-to-seed SD of validation AUROC = **0.0012** |
| Controller behaves as specified | In the τ/κ sweep, **9 of 10** runs stopped and confirmed on their own (the 10th ran to the cap); the 11th controller run, the short P6 check, ended in warm-up as expected. Decision logs recorded. *(Corrected 13 Oct after assessor review: an earlier version said "16 of 20", which was a miscount.)* |

## 4. What the study does not claim, and why

| Not claimed | Reason |
|---|---|
| A new optimization mechanism | Every component has precedent: effort allocation across variable groups by contribution (Omidvar et al., 2011; Jia et al., 2019), bandit choice of variable groups (De Rainville et al., 2013), improvement-per-time (Snoek et al., 2012), automatic stopping of tuning (Makarova et al., 2022). The contribution is the integrated controller and its controlled evaluation for ECG deep learning |
| Large accuracy gains from tuning | Measured headroom is small: untuned model 0.9351 vs best tuned 0.9387 validation AUROC; the best settings beat top-quarter settings by 0.0032 (≈ 2 × noise) |
| Clinical utility or external validity | One dataset (PTB-XL), one task (five superclasses), one CNN family, internal patient-separated test only |

## 5. Risks the assessor should weigh

1. **Stop timing follows the 40% floor.** In 6 of 8 pilot runs the controller was ready to stop at 31–38% of the budget, but the floor delayed it to 40%. The added floor-sensitivity and naive-stopper runs (30 runs) test this directly; without them H2 would be uninterpretable.
2. **Small headroom.** Final-accuracy differences between methods will be small, so the conclusion rests mainly on search time and the attribution checks.
3. **P vs DEHB compares complete systems** (single-fidelity block DE vs multi-fidelity DE). Component effects come from the ablations.
4. **BOHB** is an in-house reimplementation, not yet compared with the official implementation.

## 6. Decisions requested from the assessor

1. **Non-inferiority margin:** keep the proposal's 0.005, or tighten to ~0.002–0.0025? The measured top-end headroom (0.0032) is smaller than 0.005.
2. Are the **attribution checks** (floor sensitivity plus naive stopper) sufficient to support the efficiency claim?
3. **BOHB:** accept the reimplementation after a synthetic comparison, or require the official implementation?

**Cost and time to complete (estimate, not a guaranteed bound):** about 148 G4-hours (≈ 1,320 Colab compute units, ≈ $130) at the pilot T_ref of 9.79 s. The timing drift check and protocol freeze are pending, and reruns or slower sessions would increase it. About 3–4 months to submission.

## References

De Rainville, F.-M., Sebag, M., Gagné, C., Schoenauer, M., & Laurendeau, D. (2013). *Sustainable cooperative coevolution with a multi-armed bandit*. arXiv. https://doi.org/10.48550/arxiv.1304.3138
Jia, Y.-H., Chen, W.-N., Gu, T., et al. (2019). Distributed cooperative co-evolution with adaptive computing resource allocation for large scale optimization. *IEEE Transactions on Evolutionary Computation, 23*(2), 188–202. https://doi.org/10.1109/tevc.2018.2817889
Makarova, A., Shen, H., Perrone, V., et al. (2022). *Automatic termination for hyperparameter optimization*. arXiv. https://doi.org/10.48550/arxiv.2104.08166
Omidvar, M. N., Li, X., & Yao, X. (2011). Smart use of computational resources based on contribution for cooperative co-evolutionary algorithms (pp. 1115–1122). ACM. https://doi.org/10.1145/2001576.2001727
Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms*. arXiv. https://doi.org/10.48550/arxiv.1206.2944
