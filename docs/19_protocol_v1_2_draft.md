# Protocol v1.2 draft — tuning gain versus compute across ECG tasks

Prepared after the assessor review, 13 October 2026; revised after the verified G4 run on 14 October. Status: **development decision phase**;
not frozen or externally preregistered. The feasibility evidence and amendments are in doc 18.
The new task outcomes, G4 drift, precision diagnostic and novelty review determine the final
inventory. The source proposal and protocol v1.1 remain preserved in Git and in their documents.

## Question and contribution

When does hyperparameter tuning improve ECG discrimination enough to justify its compute,
and when do adaptive allocation and confirmed stopping improve on simpler policies?

The contribution is a reproducible comparison of gain over a declared default, compute-quality
trade-offs and matched mechanism controls. It does not assert a new bandit mechanism, universal
HPO superiority, clinical utility or external validity. A negative controller comparison is
informative only when the tested landscapes, baselines and uncertainty are adequately reported.

Two PTB-XL tasks still represent one dataset and one CNN family. This can support a carefully
scoped empirical paper; it does not by itself establish broad algorithmic generalization or
guarantee a strong publication. If the intended claim spans architectures or ECG populations,
add a prespecified second CNN family or external dataset by replacing lower-priority rows,
with its own feasibility/calibration and protocol review before test access.

## Task decision and evaluation roles

Completed decision pilot: rhythm GO (gain +0.02168; conditional CI95 [+0.01144, +0.03225]);
form BORDERLINE. The completed G4 development checks calibrated **diagnostic superclasses + rhythm**.
The rhythm precision comparison showed tuning gains, but did not establish P's non-inferiority.
Equal-cost assessment and the matched follow-up in doc 23 remain pending before the full matrix.

Use diagnostic superclasses plus one harder task that passes the registered decision pilot.
Do not select tasks using fold 10. If both harder tasks pass, prefer the one with the larger
fresh-seed lower bound of gain over reference; a tie within 0.001 uses the cheaper measured
evaluation. Record this selection as development-informed design, not a preregistered result.
If neither passes, retain the decision evidence and revisit scope before any large benchmark.

The same official SCP mapping flags are used. The proposed rhythm vocabulary is the **seven
classes already derived from development training folds 1–6**: AFIB, PACE, SARRH, SBRAD, SR,
STACH and SVARR. These folds are a subset of the confirmatory training folds. Freeze this list
before G2; do not reselect it using a >50 filter on folds 1–7, which would add AFLT and BIGU
and create a different nine-label task. Report the target as seven common rhythm labels, not
all PTB-XL rhythm labels. The fixed vocabulary must be audited
for positive/negative label support in the validation and test roles without inspecting model
performance. A training-only vocabulary differs from the historical benchmark count filter;
publish the exact classes and exclusions. Do not claim direct leaderboard comparability.

* Development headroom: train 1-6, validation 7.
* Independent development precision: search 1-4/5; refit 1-5/6; assessment 7. Fold 7 has already
  informed study planning through the headroom pilot, so this is an assessment held out from
  the pilot models' search/checkpoint selection, not untouched confirmatory data.
* Confirmatory search: train 1-7, validation 8.
* Final training: train 1-8, checkpoint/threshold validation 9, test 10 only after the analysis lock.

Within each task every method uses the same patient partitions, architecture family, nine-dimensional
domain, optimizer, loss, precision, maximum fidelity and final-training protocol. Normalization is
estimated from the relevant training role. Report label-wise AUROC and macro AUPRC alongside macro
AUROC so a mean does not hide deterioration on rare labels.

## Focused inventory and compute replacement

Provisional ten search seeds are paired across methods; a precision diagnostic can justify adding
seeds before G2. One candidate refit seed per search seed is shared between methods; replication
therefore reflects the complete search-plus-refit pipeline, not search randomness alone.

Per task:

| Block | Methods | Caps | Seeds | Searches | T_ref units at caps |
|---|---|---|---|---|---|
| Fixed-cost frontier | random, DE, official DEHB-native, adaptive_full, rr_full | 160 | 10 | 50 | 8,000 |
| Matched stop mechanisms | adaptive_utility, rr_utility, adaptive_patience, rr_patience | 160 | 10 | 40 | 6,400 |
| Untuned anchor | Prespecified reference, retrained with matched seeds | outside search | 10 | no search | measured separately |

Two tasks: **180 search runs and 28,800 task-specific T_ref units**, replacing the previous
280-run draft and the original 300-run plan. Select 80-unit incumbents from the full methods'
160-unit trajectories, with independent refits, to report a declared intermediate prefix policy.
This is not an independently rerun 80-unit algorithm; a separate shorter cap can change
multi-fidelity scheduling. All matched stopping arms retain the same 160-unit cap.

Search-hour cap = `14,400 * (11.170414 + 12.954911) / 3600` = **96.50 G4 hours** using the
measured common-workload units. A 20% reserve on that search estimate gives 115.80 hours;
refits, pilots, setup and analysis are additional. The discarded 280-run draft would require
123.31 search hours, or 147.97 with the reserve, before any refits or pilots.

Refit upper inventory: 180 returned models + 20 references + 100 full-method 80-unit prefixes
+ 40 adaptive/round-robin full-method 64-unit (40%) prefixes = **340 model refits**. Reuse exact
configuration/seed/partition matches with explicit provenance and cost accounting. The 15
development refits took 9.0–51.8 seconds each, but the final training workload is larger;
these are observations, not final cost bounds. Add at least 7h20m already reported for the
13/14 October G4 sessions, the new follow-up (at most 5.81 search hours plus refits), earlier
setup usage and final refit/setup estimates. Reconcile the total with the 148-hour envelope
before G2. No final-task runtime is inferred from the local RTX 3050.

Define the confirmatory unit using the same declared calibration workload for every task
(development folds 1-6/7, the exact 16 LHS settings and training seed recorded in
`configs/calibration_definition_v1.json`). These settings were specified before the G4 run;
no repeat timing run is needed merely to define them. The precision pilot separately calibrates its smaller
search workload (1-4/5) for development runs. Its unit cannot silently replace the final task
unit or certify the final inventory's cost. Confirmatory calibration and the measured refit
costs must be written into the final compute table before G2.

The 40% floor can be evaluated as a deterministic comparator by selecting incumbents from the
full-run trajectories at 40% of cap; this requires additional refits but no additional search.
Report it as a fixed-cutoff policy. A retrospectively truncated trajectory is not a replay of an
adaptive policy with a different floor or an alternative sequence of actions. Optional floor
20/60 sensitivity replaces contextual rows only if needed; it is not automatically added.

## Cost and comparison fairness

Use a single dedicated G4 job at a time for live wall-clock experiments. Different sessions must
pass the drift check; record GPU, environment and order. Run order is randomized within seed shards.
Log training cost, controller/optimizer overhead, confirmation cost, final training and inference
cost separately. Saving in search time is reported in seconds as well as ratios.

Specify physical wall time and accounted search time separately. The current clock adds the
cost of cached common initialization to physical elapsed time so all matched arms pay for it;
DEHB-native evaluates its own initialization. The runner's residual overhead includes cleanup
and logging as well as optimizer work. Isolated controller overhead requires explicit profiling,
not relabeling that residual.

Within matched block-DE arms, common initial configurations, training seeds and proposal RNG
streams are identical until allocation decisions diverge. External DEHB uses native initialization
and pays for its own work rather than an external design it cannot learn from. The legacy shared-
design adapter is a development fairness sensitivity, not the primary DEHB baseline. Native
initialization means the external comparison is system-level; exact component attribution is
provided by the matched arms.

The matched stopping arms use common confirmation that vetoes any incumbent improvement and then
checks the same two fresh-seed re-evaluations. This is revised P, distinct from legacy `mu`'s
utility-threshold confirmation. The optimistic allocation index is a heuristic: do not describe
it as a valid confidence bound without assumptions and proof. No asymptotic bandit regret claim
is made for the nonstationary, dependent search rewards.

## Outcomes, uncertainty and claims

Primary report: within each task, final test gain over the untuned reference versus measured
search time, with paired seed/patient-cluster uncertainty. Report all policies and the compute-
quality frontier; dominance means lower cost with no material loss supported by the intervals,
not merely an overlapping point estimate. Do not rank seeds as though they were independent datasets.

Mechanism contrasts, defined before freeze:

1. adaptive_full versus rr_full for validation anytime performance and test AUROC at fixed caps.
2. Adaptive versus round-robin allocation under the same stop rule.
3. Utility versus patience proposal under the same allocation and confirmation procedure.
4. All stopping policies versus the fixed-40%-cutoff policy and DEHB-native at matched cost.

Use task-specific paired contrasts, Holm adjustment within the declared mechanism family, and
effect sizes with confidence intervals. Report interactions rather than assuming stop and
allocation effects add independently. External method comparisons are secondary unless the
precision pilot establishes a substantively justified primary contrast.

Validation time-to-target uses a **fixed gain above the reference evaluated with the same
candidate-training seed and search partition**, with gain targets {0.002, 0.005, 0.01}. Targets
are definitions rather than chosen after seeing which method wins. Charge each method's own
search and report reference evaluation cost separately. Retain failures to reach a target as
censored at the common cap; do not average times only over successful runs. Target attainment
and anytime validation results are not evidence of test generalization by themselves.

No non-inferiority margin is approved by the seed-noise result. A secondary NI analysis requires
a written acceptable-loss rationale and paired precision evidence before G2; report it as
inconclusive if its bound does not establish NI. The development five-pair precision projection
cannot guarantee power on fold 10 and may underestimate search variance. Margin-sensitivity
curves cannot replace the frozen primary criterion.

## Gates and reproducibility

Before G2: headroom decision; second-session G4 drift; task-specific T_ref/noise; matched-control
real-data smoke; paired precision diagnostic; final
inventory and compute estimate; analysis-plan revision; Git commit and protocol tag.

The researcher requested prioritizing experiments and accepting the supplied SLR numbers.
Additional systematic literature retrieval is not a compute-execution gate for this scoped
empirical study. Retain the verified precedents and avoid unsupported first-study claims.

`configs/decision_status.yaml` currently blocks freezing. `run_search.py` rejects real non-pilot
experiments without a matching frozen protocol. The G4 decision notebook runs development checks,
not the confirmatory matrix. No analysis or test lock is created during this phase.
