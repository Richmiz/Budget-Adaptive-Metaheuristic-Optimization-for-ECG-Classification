# 07 — Statistical analysis plan

Pre-specified before any confirmatory run. The SHA-256 of this file is recorded in `protocol/analysis_lock.json`. Any later change must be logged in §7.10 with a date and reason, and must be described as a deviation in the paper.

## 7.1 Outcomes

| Outcome | Definition | Data |
|---|---|---|
| **Test macro AUROC** | Mean of the 5 per-label AUROCs, fold 10, final model | primary |
| Search time | Elapsed search clock (s); also single-GPU hours | `summary.json` |
| Anytime efficiency | ∫₀ᴮ incumbent search-validation AUROC dt / B; step function, value 0.5 before the first valid result (D29), carried forward after stopping | `trajectory.csv` |
| Macro AUPRC, label-wise AUROC | fold 10 | secondary |
| Threshold metrics | macro F1, sensitivity, specificity with fold-9 thresholds | secondary |
| Deployment efficiency | batch-one median and p95 latency, parameters, size, peak memory | secondary |
| Controller behaviour | time share per block, switch/probe frequency, stop reasons, confirmation cost, overhead | descriptive |
| Premature termination | see §7.5 and §7.7 | ablation |

## 7.2 Units of analysis and uncertainty

* The unit of replication for search is the **search seed** (10, paired across methods).
* Test-set sampling uncertainty comes from **patients**: resample patients with all their records (cluster bootstrap).
* **Hierarchical bootstrap** (primary): in each replicate, resample seeds with replacement *and* patients with replacement, then compute the mean over the resampled seeds of the paired metric difference. B = 2,000 replicates for development and 10,000 for the reported analysis. Percentile intervals.
* Seed-to-seed variation is always reported separately as SD and range. Ten seeds are not ten datasets.

## 7.3 Primary analysis: H2 (P vs DEHB at B_mid)

* **Estimand:** Δ = E_s[ AUROC_test(P, s) − AUROC_test(DEHB, s) ] at B_mid.
* **Non-inferiority:** supported if the **one-sided 95% lower bound** of Δ is **> −0.005** (provisional margin, D30; final value and justification fixed before the analysis lock).
* **Cost:** geometric-mean ratio R = search time(P) / search time(DEHB) over paired seeds, with a seed-bootstrap CI.
* **Efficiency claim:** supported only if non-inferiority holds **and** the one-sided 95% upper bound of R is < 1 (D31).
* **Report:**
  * Δ with its two-sided 95% CI
  * R with its 95% CI
  * absolute GPU-hours
  * margin-sensitivity plot over margins {0.0025, 0.005, 0.01}, which supplements but never replaces the primary criterion
* A non-significant difference is **not** evidence of equivalence. Only the non-inferiority criterion is.

## 7.4 H1: allocation without stopping (B_mid)

Two contrasts, Holm-adjusted together:
* `mu_nostop` vs `de`
* `mu_nostop` vs `rr_nostop`

**Primary H1 outcome:** paired difference in **anytime efficiency** (search-validation). Seed-bootstrap 95% CI; Wilcoxon signed-rank p-value for the Holm step.

**Secondary H1 outcome:** test macro AUROC at the cap (hierarchical bootstrap).

**H1 supported:** both Holm-adjusted p < 0.05 with positive differences. Report effect sizes regardless.

## 7.5 H3: probes and joint block (B_mid)

* **Premature termination**, for a run of variant V with seed s that stopped by utility: indicator that its final search-validation incumbent is below the incumbent reached at the cap by `mu_nostop` with the same seed by more than 0.005.
* **Contrasts:**
  * `mu_noprobe` vs `mu`
  * `mu_nojoint` vs `mu`

  For each: premature-termination rate (exact McNemar-style paired comparison over seeds; descriptive given n = 10) and test macro AUROC difference (hierarchical bootstrap).
* **Mechanism evidence for block J:** share of incumbent improvements produced by J rounds; how often a J improvement changed coordinates from ≥ 2 blocks.
* `mu_nocost` and `mu_noconfirm` are reported descriptively against `mu`:
  * time used
  * AUROC
  * premature-termination rate
  * confirmation cost
  * how often confirmation reversed a stop

## 7.5b Stop-rule checks (added 13 Oct 2026 after the pilot; D43, D44)

The pilot showed the controller's stop time was set mainly by the 40% minimum-time floor (doc 13). Two pre-specified checks separate the controller's contribution from that rule. Both run at B_mid, with seeds 1001–1010 and the same evaluator.

* **Naive stopper (`de_patience`, D44):** standard DE that stops after 16 evaluations without improvement, never before 40% of the cap (P's floor).
  * Contrast `mu` − `de_patience`: paired search-time ratio (geometric mean, seed bootstrap) and test macro AUROC difference (hierarchical bootstrap, as in §7.3).
  * If P is neither faster nor more accurate than `de_patience`, the paper states that the time saving does not require the marginal-utility controller.
* **Floor sensitivity (`mu_floor20`, `mu`, `mu_floor60`, D43):** mean and range of budget used, termination reasons, and test macro AUROC per floor (0.2 / 0.4 / 0.6). Report the fraction of runs whose stop was proposed at the floor (stop allowed by the controller's own rule earlier than the floor).
  * If budget used tracks the floor (stop ≈ floor + confirmation), the H2 time saving is attributed to the floor in the paper.
* **H2 is interpreted together with these checks.** A time saving that disappears relative to `de_patience`, or simply follows the floor, cannot support the claim that the controller's utility signal causes efficiency.

## 7.6 RQ4: stability (descriptive)

For `mu` at each budget:
* SD and range of test AUROC across seeds
* distribution of termination time as a fraction of B
* termination reasons
* action frequencies
* block time shares

The same is reported for every method: between-seed SD of test AUROC and of search time.

## 7.7 Premature-termination opportunity analysis (proposal §3.7.3)

For each `mu_nostop` run, take the shadow stop (the first moment the stop rule held, D24). Then compute:
* **Validation foregone gain** = incumbent at the cap − incumbent at the shadow stop
* **Test foregone gain** = test AUROC of the final model of the cap incumbent − that of the shadow-stop incumbent (`train_final.py --shadow`)

Report the distribution and the frequency of foregone gain > 0.005.

## 7.8 Secondary comparisons and sample adequacy

* All 7 methods × 3 budgets:
  * test macro AUROC, AUPRC and label-wise AUROC (mean ± SD over seeds)
  * search time
  * anytime efficiency
  * deployment metrics of the selected models
* **Friedman test** per budget on test macro AUROC (seeds as blocks), with mean ranks. Conclusions are limited to this benchmark.
* **Exploratory pairwise P vs each comparator**, per budget: Wilcoxon signed-rank over seeds, Holm-adjusted within budget. Labelled exploratory.
* **Seed adequacy:** before the freeze, `seeds_for_noninferiority(sd_paired)` uses the pilot variance (doc 06 P5). Seeds can only be added before fold-10 access; otherwise imprecision is reported.

## 7.9 Test-set access procedure

1. All 270 search runs, final models, reference models and shadow models are complete (G3).
2. Any amendments to this plan are logged in §7.10.
3. `python scripts/lock_protocol.py analysis` writes `protocol/analysis_lock.json` (the hashes of the frozen protocol and this file).
4. `python scripts/evaluate_test.py` runs **once**. Re-running is allowed only for technical failures and must be logged.
5. `python scripts/analyze.py primary`, then the secondary scripts.

## 7.10 Missing data, failures and deviations

* A job that crashes is restarted from scratch up to 2 times (D39). If it still fails, it is reported. Paired analyses use seeds with complete pairs only, and the excluded seeds are listed.
* Runs with no valid incumbent are reported as failures and never imputed.
* **Deviation log:**

| Date | Section | Change | Reason | Before/after test access |
|---|---|---|---|---|
| — | — | — | — | — |

## 7.11 Tables and figures (pre-specified)

| ID | Content |
|---|---|
| T1 | Dataset and partition characteristics (doc 03) |
| T2 | Methods, settings, fidelities, shared-design handling (doc 05) |
| T3 | Main results: test macro AUROC (CI), AUPRC, search GPU-hours, anytime efficiency per method × budget |
| T4 | Primary non-inferiority and cost-ratio result |
| T5 | Ablations: time used, AUROC, premature-termination rate, confirmation reversals |
| T6 | Deployment efficiency of the selected architectures |
| F1 | Controller schematic (blocks, scores, state machine) |
| F2 | Anytime curves (median and IQR over seeds) per budget |
| F3 | Performance vs search cost (one point per run; method colour; budget facet) |
| F4 | Time allocation across blocks over the search (P, stacked area) |
| F5 | Non-inferiority forest plot with margin sensitivity |
| F6 | Ablation forest plot; premature-termination analysis |
| S-* | Label-wise results, threshold metrics, per-seed tables, decision-log examples |
