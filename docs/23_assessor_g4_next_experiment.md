# Assessor statement after the G4 decision run

14 October 2026. **Development experiment verified; proceed to the matched follow-up.
The confirmatory study is not frozen.** This statement can be given directly to the research agent.

## What the completed run establishes

The export contains 227 hash-verified files. The executed source bundle matches its registered
precision plan; all 15 prediction files verify. Recomputing patient/seed bootstrap estimates
from the saved predictions reproduces the reported contrasts. No accepted candidate result
finished after its cap. Budget-limited jobs have measured deadline overshoot of about 0.1 seconds.
The fresh-session timing check passed (+1.74% median duration). Evidence: [audit JSON](evidence/g4_decision_audit.json).

Rhythm assessment on fold 7, five paired seeds:

| Policy | Mean macro AUROC | Gain over default, 95% hierarchical CI |
|---|---:|---|
| Untuned reference | 0.93260 | — |
| P / adaptive utility | 0.95177 | +0.01917 [0.00579, 0.03316] |
| Official DEHB, native initialization | 0.95910 | +0.02650 [0.01044, 0.04104] |

P/DEHB accounted search-time ratio is 0.572 [0.488, 0.698]. P−DEHB assessment AUROC is
−0.00733 [−0.02120, +0.00680]; its one-sided 95% lower bound is −0.01870. **None of the
proposed NI margins is established.** The interval also does not establish superiority or
inferiority. Fold 7 previously informed task selection, so these remain development results.

Tuning gains are supported on this development task. A controller efficiency advantage at
comparable final quality and a mechanism contribution remain open questions. An overlap in
intervals is not equivalence; a smaller point gain at a lower cost is not a dominance result.

## Corrections to carry into the study

1. **Compare held-out quality at the same search cost.** At P's stop, the validation difference
   versus DEHB is only +0.00011 on average. All five DEHB prefix configurations differ from the
   final DEHB configurations. They need separate refits; validation ties cannot answer this question.
2. **Confirmation is active.** Across the run there were 13 attempts: seven confirmed stops,
   three improvement-probe vetoes and three seed-stability rejections. Preserve this rule for the
   matched comparison. Its tolerance is a heuristic and does not guarantee an acceptable AUROC loss.
3. **No repeat calibration just to fix the definition.** The task calibration's 16 settings and
   training seed were already specified. Their realized settings, label lists and measured units
   are saved in [calibration_definition_v1.json](../configs/calibration_definition_v1.json).
   They are a recorded development definition, not a protocol freeze. Different settings and
   training seeds explain why units can differ while the matched hardware drift check passes.
4. **Retain the seven rhythm labels tested.** Refiltering training folds 1–7 would add AFLT and
   BIGU. Freeze AFIB, PACE, SARRH, SBRAD, SR, STACH and SVARR; clearly report the exclusions.
5. **Treat the five-pair precision projection cautiously.** It is an unvalidated diagnostic,
   not proof that NI is impossible or a fold-10 power calculation. Do not widen a margin to make
   a desired conclusion easier to obtain. No secondary NI margin is approved by this assessment.

## Next experiment: bounded matched follow-up

Open [COLAB_G4_MATCHED_FOLLOWUP.ipynb](../notebooks/COLAB_G4_MATCHED_FOLLOWUP.ipynb) in VS Code,
choose a fresh Colab G4 kernel, and **Run All**. It reads the original Drive folder
`My Drive/bamo_decision` (or its previous exported ZIP) and writes to
`My Drive/bamo_matched_followup_2026_10_14`. Original inputs are hash-pinned and remain unchanged.
It stops if the scientific source, dataset identity, original settings, GPU/Torch/CUDA/cuDNN
versions or GPU timing gate differ. Session fingerprints also retain Python and host information.

| Work | Count | Purpose |
|---|---:|---|
| Reuse original P, DEHB and reference searches/predictions | 5 pairs + 5 reference refits | Preserve completed evidence |
| New rr_utility, adaptive_patience, rr_patience searches | 3 × 5 = 15 | Complete the four matched allocation × stop-proposal arms at 160 units |
| Refit DEHB incumbents completed by P's stopping time | 5 | Assessment quality at equal accounted search cost |
| Refit returned models from the three new arms | 15 | Compare component decisions with the same confirmation |
| Refit adaptive/RR utility 40% prefix configurations | up to 10 | Assess a fixed cutoff using search trajectories |

Each new search retains the precision pilot's T_ref 8.709123 s, 160-unit cap, tolerance 0.0246,
40% minimum time and seeds 9401–9405. The initial design and proposal RNG group are shared;
cached initialization is charged identically. Search order is randomized and runs use one GPU
job at a time. Search uses folds 1–4/5; refits use 1–5/6 and assessment uses 7. All configurations
are selected from search logs and saved before assessment predictions are opened. Original
predictions and any exact configuration/refit-seed matches are reused with explicit provenance.

**Cost:** at most 5.81 new accounted search hours, plus up to 30 independent 50-epoch/patience-8
refits, setup, drift checks and analysis. This is a search-cap estimate, not a total runtime bound.
Cross-session pairing is a development limitation even when drift passes. The main study must
randomize complete method groups across sessions.

The declared DEHB prefix and fixed 40% prefixes are policies selected from trajectories. They
are not independent shorter-cap DEHB runs or adaptive controllers rerun with new floors. Compare
search cost and refit cost separately. This follow-up cannot validate the earlier 20/40/60%
floor-sensitivity claim; that claim would require actual controller reruns at those floors.

Export **bamo_matched_followup_results.zip** and save the executed notebook to the workspace.
The output contains registered plans, search logs, preselected configurations, prediction hashes,
refit provenance/costs and paired assessment contrasts. Completed work resumes after interruption;
changed plans require a new output directory.

## Decision rules after the follow-up

- If P has no supported quality benefit over DEHB at the same cost, do not claim a DEHB efficiency
  advantage. Retain the transparent quality/cost comparison.
- If adaptive allocation does not improve on round-robin under matched rules, do not attribute
  savings to allocation. Report effect estimates and uncertainty, including mixed or negative results.
- If utility stopping does not improve on patience or the fixed-cutoff policy at a useful quality/cost
  trade-off, do not claim that the utility signal caused the saving. Smaller cost alone is insufficient.
- If estimates remain wide, report that uncertainty; five developmental pairs do not establish a
  confirmatory mechanism result. Avoid tuning parameters to these assessment outcomes.

## Direction for a credible paper

Use **final test discrimination gains at declared search costs** as the central empirical question.
Validation anytime curves and censored time-to-target analyses explain search behavior. Matched
controls assess allocation and stopping decisions. This is a feasible route to a rigorous ECG
compute-quality study; it does not guarantee a strong publication or a novel optimizer claim.

The revised draft in [doc 19](19_protocol_v1_2_draft.md) proposes 180 searches across two tasks,
with 80-unit prefix assessments and matched stopping at 160 units. Measured search caps total
96.50 G4 hours; 20% search reserve brings that to 115.80 before refits, pilots and setup. Up to
340 final refits need a workload-specific cost forecast. The previous 280-run draft nearly used
the whole 148-hour envelope on search plus reserve alone. Reconcile all spent and future usage
before G2. Ten paired seeds are provisional for estimation, not a validated power guarantee.

Before the full study, finish the task-aware runner, final estimands and multiplicity family,
fixed task definitions, complete cost inventory, protocol commit/tag and analysis/test locks.
No fold-10 model evaluation is authorized by a development notebook. Two tasks remain one
dataset and one CNN family; wider algorithm claims require a separately planned architecture
or dataset replication. The supplied systematic-review numbers are accepted; no further review
audit is requested.

## Implementation verification

The local suite passes **76 tests**, including GPU training tests and new protections against
changed core source, changing a registered plan, using a final DEHB model at a prefix cost,
late accepted results, changed GPU/training-framework versions, and prediction hash/dimension mismatches. Input preflight verifies the
original export and registers the concrete follow-up without training. Notebook code cells and
embedded source bytes are verified separately. **The new G4 experiment has not yet run.**

A [mocked orchestration check](evidence/matched_followup_orchestration_check.json) completed all
45 prediction-artifact paths and seven contrasts using simulated search/training. This checks
reuse and execution wiring; its synthetic outputs are not experimental results.
