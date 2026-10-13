# 14 — Critical review of the study design (8 Oct 2026)

Written after session 1 on the G4, prompted by the question "why train 51,000 CNNs?". This reviews the **proposal's design** and my implementation of it. It does not review the code.

## 14.1 Verdict

The engineering is sound: the data pipeline, leakage controls, pre-registration, code and Colab setup all work. The **experimental design is over-sized and partly mis-aimed.** It spends most of its compute on breadth (7 methods × 3 budgets × 10 seeds on one task), while two issues that decide whether the study can show anything at all are not yet addressed:

1. **Headroom.** We don't know whether this task's hyperparameter landscape is steep enough for search methods to differ measurably.
2. **H2 is close to automatically true** as currently set up.

Both should be resolved before any large run.

## 14.2 What is good

| Area | Why it holds up |
|---|---|
| Data | Verified download, benchmark label mapping, 0 patient overlap, fold-10 lock |
| Evaluation discipline | Pre-specified analysis, decision register, test set opened once |
| Reproducibility | Byte-identical data on PC and Colab; embedded, hashed code; pinned versions |
| Cost transparency | Search, final-training and inference costs separated, which is the gap the SLR identified |
| Controller design | Clear, testable rules with ablations that isolate each component |

## 14.3 Problems

### P1. Unknown headroom: the benchmark may not be able to tell methods apart
* The untuned reference CNN already reaches **0.935** validation macro AUROC (fold 7).
* The published PTB-XL benchmark reports most strong architectures on this task within roughly **0.01** macro AUROC of each other.
* If the best and a median configuration differ by less than the seed-to-seed noise, every search method lands in the same place. The study then spends its budget showing a tie, and the controller's decisions are driven by noise.
* **Measured and confirmed (8–9 Oct 2026):** the first single-run check said GO (5×), but repeats downgraded it to **BORDERLINE**: best-setting mean 0.9387 vs top-quarter settings 0.9356, a gap of 0.0032 = 2.0× seed SD (0.0016). Final accuracy will barely separate methods; time-to-target still can. Details in doc 13.

### P2. H2 is nearly guaranteed by construction
* P can stop early; DEHB and the other comparators always run to the cap. So "P uses less search time than DEHB" is true whenever P stops.
* With a flat landscape (P1), non-inferiority at 0.005 is also likely for almost any reasonable method.
* Result: H2 could be "supported" even if block allocation adds nothing beyond "stop when progress flattens".
* **Missing control:** a naive stopping baseline, for example DE or DEHB that stops after N evaluations without improvement. Without it, the paper cannot show the *controller* is what saves the time.

### P3. Compute is spent on breadth, not on what the hypotheses need

| Part (proposal §3.5.2, §3.9.2) | Trainings | Needed for H1–H3? |
|---|---|---|
| Biggest budget, 7 methods | 22,400 (44%) | No (RQ4 only, and only P is needed) |
| Random, TPE, Hyperband, BOHB at all budgets | ~16,000 | No (context only) |
| P vs DEHB at B_mid | 3,200 | **Yes: H2** |
| mu_nostop, rr_nostop, DE at B_mid | 4,800 | **Yes: H1** |
| mu_noprobe, mu_nojoint at B_mid | 3,200 | **Yes: H3** |

About **11,200 of the ~51,000 trainings** test the hypotheses. The rest is context.

### P4. One task cannot support a general claim about a search method
A controller for hyperparameter search is a *general* method; PTB-XL is only its test case. Results from one dataset with one CNN family say little about the method in general. HPO-methods papers usually evaluate across several tasks, often with **benchmark tables** (train a sample of configurations once, then run every search method as a simulation on the recorded results), which is what makes breadth affordable.

### P5. Live wall-clock cost blocks efficient execution
Measured on G4: concurrent jobs slow each other almost linearly, so live timing forces one job at a time with the GPU mostly idle. Using profiled cost (D41), or recorded costs as in a benchmark table, removes this constraint.

### P6. Novelty is unverified
HABO (2025 preprint, bandits over hyperparameter subsets), adaptive operator selection in EAs, cost-aware BO and automatic HPO termination all overlap. The claim may need to narrow, and that should be known before the paper's framing and design are fixed.

### P7. My own over-engineering
14 documents, a 40-item decision register and an in-house BOHB are heavier than this study needs, and I should have raised the proposal's compute scale (§3.9.2: ~813 GPU-hours) on day one instead of building around it.

## 14.4 Recommendations, in order

| # | Action | Cost | Decides |
|---|---|---|---|
| **R1** | **Headroom check** on pilot data (folds 1–6 → 7): 24 random configurations + reference, plus 5 seeds of 2 configurations. Compare the spread across configurations with the seed noise | ~40 trainings: free locally (~45 min) or ~2 CU on G4 | Go / redesign |
| **R2** | Add a **naive stopping baseline** (DE + patience stop) and report **time-to-target** for all methods from the anytime curves | Small | Makes H2 meaningful |
| **R3** | Right-size to the **core design**: P, DEHB, DE, the naive stopping baseline, the ablations at B_mid, and P at all 3 budgets | ~15,000–18,000 trainings ≈ $50–65 on G4 | Budget |
| **R4** | For generality, prefer a **benchmark-table design**: train ~2,000–3,000 sampled configurations per task once, simulate all searches; add a second task (e.g. another PTB-XL label set) instead of more budgets | ~$10–15 per task | Strength of claim |
| **R5** | Run the novelty search (doc 11) now | Time only | Framing |

If R1 shows little headroom, the useful options are a harder task (for example the 71-statement or rhythm/form tasks in PTB-XL), a broader search space, or the benchmark-table design across several tasks. More seeds or budgets on the current task would not help.
