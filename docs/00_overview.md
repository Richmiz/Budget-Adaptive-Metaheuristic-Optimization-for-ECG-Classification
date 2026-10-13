# 00 — Study overview and traceability

**Title.** Budget-Adaptive Metaheuristic Optimization for ECG Classification Using Marginal Utility-Based Search Allocation
**Design.** Method development + controlled, patient-separated benchmarking on PTB-XL (v1.0.3).
**Source.** `Proposal_Budget_Adaptive_ECG_Optimization.docx` (7 Oct 2026). These docs expand that proposal into an executable protocol. Where the proposal was silent or ambiguous, the choice is recorded in [08_decision_register.md](08_decision_register.md).

## Core idea in one paragraph

Differential Evolution (DE) generates candidate CNN configurations. A controller decides *where* DE spends its time: in the training block (A), the regularization block (B), the capacity block (C) or a joint block (J). It allocates each round to the block with the highest optimistic estimate of **validation-AUROC improvement per unit of measured compute**. It stops searching only after that estimate has stayed below a threshold τ for every block, and only after a confirmation step (one more probe per block plus re-evaluating the incumbent with fresh seeds). The question is whether this saves search time without losing test performance (non-inferiority against DEHB), or buys more performance within a fixed time cap.

## Research questions → hypotheses → experiments → analyses

| RQ | Hypothesis | Experiment / runs | Contrast | Primary outcome | Analysis (doc 07) |
|---|---|---|---|---|---|
| RQ1: Does adaptive allocation help under the same time cap? | **H1:** P without global termination (`mu_nostop`) has higher anytime efficiency than standard DE and round-robin (`rr_nostop`). | Ablation at B_mid (10 seeds) + main `de` at B_mid | `mu_nostop` − `de`; `mu_nostop` − `rr_nostop` | Anytime efficiency (validation); final test macro AUROC | §7.4 |
| RQ2: Does utility-based termination cut cost without losing performance? | **H2 (primary):** P uses less search time than DEHB and is non-inferior in test macro AUROC (margin 0.005). | Main benchmark at B_mid (10 seeds) | `mu` − `dehb` | Test macro AUROC difference (one-sided 95% lower bound); search-time ratio | §7.3 |
| RQ3: What do switching, mandatory exploration and joint search contribute? | **H3:** Removing mandatory probes or block J increases premature termination or lowers performance. | Ablation at B_mid | `mu` vs `mu_noprobe`, `mu_nojoint` (plus `mu_nocost`, `mu_noconfirm` descriptively) | Premature-termination rate; test macro AUROC; block time shares | §7.5 |
| RQ4: How stable are decisions and performance? | Descriptive. | `mu` at B_low/B_mid/B_high (30 runs) | Across seeds and budgets | Between-seed SD; termination time; action frequencies | §7.6 |

## Run inventory

| Block | Methods | Budgets | Seeds | Runs |
|---|---|---|---|---|
| Main benchmark | random, tpe, de, hyperband, bohb, dehb, **mu** | low (80 T_ref), mid (160), high (320) | 10 | **210** |
| Ablations | mu_nostop, rr_nostop, mu_nocost, mu_noprobe, mu_nojoint, mu_noconfirm | mid | 10 | **60** |
| Stop-rule checks (added 13 Oct, D43–D44) | mu_floor20, mu_floor60, de_patience | mid | 10 | **30** |
| Pilot | mu sweeps + calibration evaluations | mid | 2–5 | ~10 runs + ~40 evaluations |
| Final training | one model per completed run + reference CNN + shadow-stop incumbents | — | — | ~290 models |

## What success and failure look like

* **Supported efficiency claim:** non-inferiority holds (lower bound > −0.005) **and** the one-sided 95% upper bound of the P/DEHB search-time ratio is < 1.
* **Not supported:** either condition fails. The paper then reports where and why the controller loses (premature-termination analysis, doc 07 §7.7). Margins and domains are never changed after the fact.
* Search savings on PTB-XL fold 10 say nothing about clinical utility or external validity (internal, patient-separated generalization only).

## Document map

| Doc | Content |
|---|---|
| [01_study_plan.md](01_study_plan.md) | Work packages, 26-week timeline, gates, risk register |
| [02_experimental_protocol.md](02_experimental_protocol.md) | Factors, model, search space, evaluators, budgets, run procedure |
| [03_data_and_preprocessing.md](03_data_and_preprocessing.md) | PTB-XL, labels, checks, partitions, normalization |
| [04_controller_specification.md](04_controller_specification.md) | Formal controller definition, state machine, logging |
| [05_comparators.md](05_comparators.md) | Baselines, implementations, fairness rules |
| [06_pilot_calibration.md](06_pilot_calibration.md) | What is calibrated, how, and the freeze procedure |
| [07_analysis_plan.md](07_analysis_plan.md) | Statistical analysis plan (pre-registered; hashed at analysis lock) |
| [08_decision_register.md](08_decision_register.md) | Every resolved ambiguity and every open decision |
| [09_data_management_and_reproducibility.md](09_data_management_and_reproducibility.md) | Layout, log schemas, seeds, versions, release |
| [10_compute_and_colab_workflow.md](10_compute_and_colab_workflow.md) | GPU budget, Colab-via-VS-Code runbook |
| [11_novelty_assessment.md](11_novelty_assessment.md) | Literature search protocol for the novelty claim |
| [12_manuscript_plan.md](12_manuscript_plan.md) | Paper outline, figures, tables, claim discipline |
| [13_progress_log.md](13_progress_log.md) | Dated log of work done, findings, and changes |
| [14_design_review.md](14_design_review.md) | Critical review of the design: headroom, H2 asymmetry, compute scope |
| [15_evidence_review.md](15_evidence_review.md) | Evidence check of every claim (literature + our data) and the evidence-backed recommendation |
| [16_protocol_v1_1.md](16_protocol_v1_1.md) | **Consolidated post-pilot protocol for assessment (start here if reviewing)** |
