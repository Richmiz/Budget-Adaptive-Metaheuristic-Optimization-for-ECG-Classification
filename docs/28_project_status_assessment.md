# Project status assessment — 16 October 2026

## Verdict

**The core v2 experiment is complete and ready for manuscript preparation. It is not yet ready
for submission.** The main remaining work is accurate interpretation, complete reporting and
reproducibility packaging. The verified evidence does not require rerunning the main study.

The project changed from a controller non-inferiority/attribution study to the **route A
compute–quality study** in docs 25–26. The frozen v2 design contains random search, native
official DEHB and P at two budgets, on diagnosis and rhythm. Standard DE, matched allocation
controls and the proposed 15-search follow-up are not part of the completed confirmatory study.
Those older recommendations should not be treated as the current mandatory experiment queue.

## What I checked independently

Evidence is saved in [v2_independent_status_audit.json](evidence/v2_independent_status_audit.json).
The audit can be rerun with `python scripts/audit_v2_status.py`.

- **Archives:** all 779 main-export files and 302 test-export files match their ZIP manifests.
  All extracted source inputs also match. One derived file, the extracted
  `analysis_v2/search.json`, has changed since export; its original is preserved in the ZIP.
- **Completion:** 120 searches, 150 final-model records and 150 test prediction/report pairs;
  all ten seeds are present for every arm. Selected configurations match final-training
  configurations, and final-training seeds match the declared shared seed policy.
- **Predictions:** every saved per-seed test macro AUROC and per-class mean reproduces from
  predictions and local labels. Prediction rows match the complete official fold-10 task rows.
- **Timing:** 12,148 accepted evaluation records contain no completion after their run's cap.
  The earlier audit records six runtime drift checks passing within 5.3%; small deadline
  overshoot and discarded late results are disclosed.
- **Locks and executed code:** current frozen-protocol and locked-analysis hashes match the
  recorded lock. Both executed main notebooks and the test notebook have valid bundle hashes;
  all 36 bundled core Python files match the current scientific implementation. Test logs
  reference the analysis lock. This supports the documented sequence; it is a local lock,
  not an external preregistration or proof that unrecorded activity was impossible.
- **Statistics:** all stored primary paired-difference arithmetic and Holm adjustments check.
  I replayed the full 10,000-replicate diagnosis random/80 versus D0 bootstrap: difference
  +0.0028502, CI [0.0000771, 0.0057740], bootstrap tail value 0.0225, exactly matching the locked
  result. Other bootstrap intervals were not independently replayed in this status audit.
- **Data and software:** both patient-fold audits report zero overlap and no duplicate waveform
  groups. A diagnostic replay of the canonical 10,000 patient resamples found no zero-positive
  label draws. The current full test suite passes **87 tests**.

## The good

### 1. There is a real final result

These are final fold-10 results under the adopted v2 plan, rather than development fold-7
results. Completion and the saved artifacts are much stronger than at the previous assessment.

| Task | Original default D0 | Task default D1 | DEHB 80 / 160 | P 80 / 160 |
|---|---:|---:|---:|---:|
| Diagnosis | 0.92369 | D0 | 0.92712 / 0.92773 | 0.92807 / 0.92793 |
| Rhythm | 0.94863 | 0.95542 | 0.96009 / 0.95912 | 0.95554 / 0.95602 |

### 2. Diagnosis gains are supported under the declared analysis

All six tuned arms improve on D0, with mean gains +0.00285 to +0.00438 and Holm-adjusted
bootstrap p values at most 0.025. These are modest discrimination gains, with no demonstrated
clinical benefit. The strongest P point estimate is at the smaller budget.

### 3. The rhythm comparison now has a stronger baseline

D1 was chosen using development data and frozen before the main test. This answers a more useful
question than comparison only with D0. None of the six tuned arms establishes a positive gain
over D1 in the primary family. DEHB's secondary gains over D0 have intervals excluding zero.
This is informative evidence about dependence on the default, even when the result is mixed.

### 4. P returns models using less search time

Geometric mean P/DEHB time ratios imply about 16% lower time at the small budget and 36–42%
lower time at the large budget. These are comparisons of complete policies. Logs support that
P often terminates after confirmation. The experiment does not establish which controller
component is responsible for a favorable quality/cost trade-off.

### 5. The measured compute is manageable

Completed searches sum to **43.42 accounted G4 hours**, or **41.57 physical search hours** after
excluding charged cache reuse. Final training adds **1.11 physical hours**. Thus 42.68 hours
of completed search/training work are documented. These are not the full bill: interrupted
attempts, setup, drift checks, test inference, development and idle runtime usage are extra.

## The bad and the risks

### 1. Some conclusions overstate the evidence

The following language in doc 27 needs correction before it becomes an abstract or conclusion:

| Current wording | What the results support |
|---|---|
| “P matched DEHB's accuracy” | P used less time; the quality differences are uncertain. Equivalence/NI was not tested. |
| “a task-appropriate default does as well as searching” | No tuner established a gain over D1. DEHB could still provide a useful gain: its rhythm/80 versus D1 CI reaches +0.0103. |
| “doubling the search budget does not help” | Mean changes were small (−0.00097 to +0.00060). The locked script does not report direct paired 160−80 intervals. |
| “at most a few thousandths” / “little or no cost” | Point estimates do not bound the true gain or quality loss. Rhythm DEHB−D0 is about +0.0115, and P−DEHB intervals allow meaningful losses. |

The distinction between a failure to establish a difference and an established small difference
matters here. For example, P−DEHB on rhythm is −0.00454 [−0.0140, +0.0025] at the small budget:
the interval permits a loss much larger than the previous NI margins.

### 2. Original controller attribution remains unanswered

There is no completed matched round-robin control, patience control or P-without-stopping arm
in v2. Comparisons with random/DEHB establish differences between complete methods. They cannot
identify whether adaptive allocation, the utility stop signal, confirmation or the minimum-time
rule creates an advantage. This is acceptable for the adopted empirical scope if stated clearly;
it cannot support a mechanism-superiority paper.

### 3. D1 has a development cost and a selection history

The rhythm default was chosen after inspecting development tuning outcomes and then checked
locally. It has no per-run search cost once fixed, but it is not an independently supplied,
cost-free expert baseline. Disclose this history and separate one-time development compute
from the future per-run cost of using the published default. Its test advantage over D0,
+0.00679 [−0.0019, +0.0157], is itself uncertain.

### 4. Generalization and novelty remain limited

Two tasks on one dataset and one CNN family do not establish a general HPO principle.
Rhythm's rarest labels have only 14 SVARR and 28 PACE positive test records. Ten seeds do not
create ten independent datasets. A useful empirical paper is feasible; these results alone
do not establish a broadly superior optimizer or guarantee a strong publication.

### 5. Project documentation and provenance need cleanup

- README still points to main runs and the old matched follow-up as next actions, says G2 is
  pending, and reports 76 tests. Those statements conflict with the completed v2 results.
- The progress log stops at 14 October. Docs 25–26 also retain an obsolete future schedule.
- The existing figure layout is acknowledged as broken. Its error bars show seed uncertainty,
  whereas the main contrast intervals also include patient resampling; make that distinction clear.
- Neither exported result archive includes model checkpoint files. Preserve and hash the 150
  original checkpoints on Drive before considering the reproducibility package complete.
- `require_analysis_lock()` currently checks that the lock exists, without verifying its hashes.
  The audit found the present hashes intact, so this is an enforcement weakness, not evidence
  of a breached lock. Strengthen this for future studies without changing the historical lock.
- Software hashes cannot reconstruct actual billed runtime or the complete cost of interrupted
  jobs. Reconcile saved logs with runtime/billing records and disclose any unavailable quantities.

## Way forward, in priority order

1. **Finish the current study's reporting.** Preserve the original ZIPs, executed notebooks,
   frozen tags, lock and prediction files. Correct the claims and stale status instructions.
   Do not rerun the main matrix merely because the expected controller headline did not appear.
2. **Complete the cost–quality presentation.** Redraw publication figures from the frozen
   numbers in a separate script. Present paired gains and uncertainty, all methods/budgets,
   per-class results and positive counts. Report search, final training, development and retries
   separately; avoid treating accounted cache charges as physical billing.
3. **Answer the budget question explicitly.** Add paired 160−80 effect estimates and intervals
   from the existing predictions. Record this as a supplemental analysis added after the lock;
   preserve the frozen primary analysis and its conclusions. No GPU retraining is needed.
4. **Write the scoped empirical paper.** Organize it around tuning gain, choice of default and
   compute-quality uncertainty. The original NI/attribution hypothesis and the development-informed
   move to v2 belong in the study history and limitations. State that the freeze was local.
5. **Add experiments only for a defined broader claim.** A P-without-stopping/matched-component
   experiment would address attribution. An independently planned architecture or dataset
   replication would address generalization. Any new comparisons chosen after seeing fold 10
   are exploratory on this test fold; a new prospective confirmation needs independent held-out
   evidence. The optional v2 no-stop add-on is already designated exploratory in the protocol.

## Suggested central claim

> Under the prespecified analysis, tuning produced modest diagnosis AUROC gains on PTB-XL.
> Gains beyond a development-informed rhythm default were not established. P returned models
> with less search time, while comparative quality remained uncertain. Increasing the search
> budget produced small changes in observed mean AUROC.

**Recommendation:** proceed to interpretation, figures and manuscript development. Keep
additional G4 work conditional on a specific unanswered claim. This is now a defensible scoped
empirical study, with important uncertainty to report.
