# Literature verification and systematic-search handoff — 13 October 2026

Status: targeted primary-source verification completed for the mechanisms listed below.
Systematic novelty review is **not complete**. No authenticated Scopus or Web of Science exports
are present in this workspace; web rankings do not provide complete database retrieval counts.
No claim of absence or "first ECG tunability study" is permitted from this search.

## What was verified

| Primary record | Evidence inspected | Consequence for this paper |
|---|---|---|
| Omidvar et al., 2011, contribution-based cooperative coevolution | Authors' university record/abstract | Allocation to contributing variable groups predates this controller |
| De Rainville et al., 2013, sustainable cooperative coevolution | arXiv abstract | Bandit resource allocation across species is established |
| Bandit-based cooperative coevolution, 2019 | Author manuscript, Algorithms 2 and framework sections | Dynamic contribution imbalance and matched round-robin comparisons are especially close precedents |
| ECCRA, 2021 | Publisher abstract | Allocating to promising subproblems and pruning unpromising ones also have precedent; detailed stopping equivalence still requires full-text assessment |
| HyperArm/HABO, 2025 preprint | arXiv HTML sections 23 and 26 | Super-arms are individual hyperparameters, sub-arms their values; reported examples are Titanic and House Prices, not ECG. This is related allocation, not the same block-DE mechanism |
| Makarova et al., 2022 | AutoML proceedings abstract | Automatic HPO termination based on validation estimation error is established; a statistical stopping claim needs more than a heuristic score |
| CARBS, 2023 | Authors' arXiv abstract and official code repository | Cost-performance Pareto search exists for deep training. Distinguish training cost of a selected model from total HPO search cost |
| SHCHO, Chen et al., 2023 | Author preprint, Algorithms 2-3 and section 4 | Cooperative optimization of CNN hyperparameter subsets already exists. Reports GPU days and compares against original ResNet; generic integration of coevolution and deep HPO is not sufficient novelty. Its overlapping sub-CNN/surrogate mechanism differs from our complete-network block DE |
| Collaborative random search / hierarchical collaborative HPO, 2022-23 | Primary arXiv abstracts | Distributed collaborative HPO is another precedent. SciSpace's journal author metadata disagrees with the author preprints; use primary records for attribution |
| Frugal Optimization, Wu et al., 2021 | AAAI proceedings abstract | Cost-frugal randomized search is an established alternative; consider it if broad cost-aware algorithm superiority becomes the paper's claim |
| BO-R-1D-CNN, Fki et al., online 2023 / issue 2024 | Springer abstract; full methods behind subscription | BO for residual ECG CNNs is established. Search-cost accounting and patient separation remain unverified; do not assert their absence from the abstract |
| Chen et al., DE-optimized ECG CNN, 2023 | Springer abstract; full methods behind subscription | Explicitly compares tuned/untuned CNN accuracy and selected-model training time on MIT-BIH/SCDH. Broad claims that ECG tuning lacks default comparisons are false; total HPO search cost still needs full-text verification |
| PSO-Pelican ECG tuning, 2024 | Publisher full-text HTML, sections 4.2.3-5.3.2 and Table 6 | Reports model-specific training times with selected hyperparameters; a clearly isolated total search cost is not established by Table 6. Search/checkpoint/test terminology is ambiguous, so inspect code before assigning a leakage label |
| Quantity vs. Quality, Hertel et al., 2020 | Author arXiv abstract | Studies training-seed noise, pruning and repetitions for HPO in two RL tasks. Relevant motivation, not evidence that its conclusions transfer to ECG |
| PTB-XL benchmark, 2021 | Authors' repository and aggregation/selection code | Supports form/rhythm task definitions and patient fold protocol; its test leaderboard is not evidence of our validation tuning headroom |
| BOHB, 2018 / DEHB, 2021 | Primary paper records and official implementation interfaces | Use official baselines; a similar synthetic curve is insufficient to establish implementation equivalence |

Primary URLs and screening notes are retained in `docs/evidence/literature_records.csv`.
The new headroom and mechanism experiments address an empirical question, not a novel-mechanism
claim. The contribution should be rewritten after systematic screening and the development results.
The researcher supplied the project's 10/44 SLR count and asked that it be accepted as study
context without a separate audit. It is accepted on that basis; it was not independently
rederived here. Further extraction-table retrieval is outside the current work.

SHCHO is particularly consequential: its preprint reports gains over an original ResNet together
with GPU days, uses five independent runs but reports the best, and imports many comparator
results from earlier papers. A new paper can improve common-hardware comparisons, seed/patient
uncertainty, stopping attribution and ECG coverage. It cannot claim that cooperative deep-CNN
tuning or gain-versus-compute evaluation is new. The journal record is DOI
`10.1016/j.asoc.2023.110794`; its detailed methods were checked in the author preprint,
not assumed identical to the inaccessible publisher version.

## Search log

Provider: web research tool; date 13 October 2026, Asia/Jakarta. Ranked records screened for relevance,
primary sources used for substantive notes. Full hit counts unavailable; no PRISMA count is inferred.

1. `(ECG OR PTB-XL) "hyperparameter optimization" "computational cost"`
2. `(ECG OR PTB-XL) "hyperparameter tuning" "random search"`
3. `"hyperparameter" "adaptive" "subsets" "bandit" HABO`
4. `"cooperative coevolution" "contribution" "time" budget allocation`
5. `PTB-XL "hyperparameter" "compute" tuning optimization`
6. `ECG "hyperparameter optimization" "search time"`
7. `"HyperArm" "bandit optimization"`
8. `"hyperparameter" "cost benefit" "ECG"`

Mechanism queries retrieved close allocation precedents; targeted ECG queries did not supply a
verified, direct counterpart to the full proposed study. This is a retrieval observation, not
proof that no counterpart exists. Secondary summaries of HABO were excluded because they described
experiments differently from its primary text.

SciSpace and Consensus were also used for discovery. Four natural-language SciSpace questions
returned 10 ranked records each; three Consensus queries returned 20 each. These are **100 returned
entries before deduplication**, not complete hit counts. Queries and returned identifiers/titles
are saved in `docs/evidence/targeted_search_log.json` and `targeted_search_supplement.json`. SciSpace retrieved SHCHO and ECG DE/default comparisons; Consensus retrieved
the close ECG BO study and Frugal Optimization. Three relevant Consensus records were fetched
before interpretation. Primary sources were checked for the claims above. Scite retrieval was
unavailable because its connector reported a monthly quota limit; it contributed no verified
citation-context evidence. This multi-provider search remains targeted discovery, not an
exhaustive systematic review.

## Reproducible database queries still to execute

Scopus (all years for mechanisms; 2015 onward for ECG; no language exclusion at retrieval):

```text
TITLE-ABS-KEY(("hyperparameter optimization" OR "hyperparameter tuning" OR AutoML)
AND (bandit OR "adaptive allocation" OR "credit assignment")
AND (subspace OR block OR coordinate OR group))

TITLE-ABS-KEY(("hyperparameter optimization" OR "algorithm configuration")
AND ("cost-aware" OR "per unit time" OR "per second" OR "budget-aware")
AND (termination OR "stopping criterion" OR "early stopping"))

TITLE-ABS-KEY((ECG OR electrocardiogram OR "PTB-XL")
AND ("hyperparameter optimization" OR "hyperparameter tuning" OR metaheuristic)
AND (cost OR compute OR time OR efficiency OR tunability)) AND PUBYEAR > 2014
```

Use corresponding Web of Science `TS=(...)` searches and record exact strings, execution time,
database filters and full result counts. Export titles, abstracts, DOIs, authors, years and citations
to `docs/evidence/literature_exports/`; deduplicate by DOI, then normalized title/year. Screen all
unique abstracts with recorded reasons. Backward/forward snowball the close records above; retrieve
full texts before making feature absence claims. Record search-cost versus model-training-cost
accounting, number of tasks/datasets, untuned controls, stopping baselines and patient separation.

Outcome rule: narrow integration/application claims whenever a close earlier method covers a
component. A systematic search may identify an existing ECG cost-benefit study; then justify the
new work through the improved controls, tasks, uncertainty or released benchmark rather than an
unsupported claim that the research question has never been investigated.
