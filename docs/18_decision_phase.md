# Assessor decision phase — 13 October 2026

Status: authorized for implementation and development-only experiments by the researcher.
Protocol v1.1 and its 300-run matrix remain historical. G2 is pending. Baseline source/protocol
snapshot: Git commit `feff26a`; results stay in their original output directories.

## Headroom pilot registered before outcomes

Executable specification: `configs/decision_pilot.yaml`; command:

```powershell
D:\envs\bamo\python.exe -s scripts/decision_pilot.py --tasks form rhythm
```

The script writes a plan and implementation hashes before training and refuses incompatible
resumes. Outputs: `runs/decision_pilot_v1/<task>/`. Every completed evaluation is persisted so
an interruption loses only its current training. Truncated console values are never reused.

* Development train = PTB-XL folds 1-6; validation = fold 7. No waveform from folds 8-10 is opened.
* Labels use the official SCP `form` / `rhythm` flags and record keys, including likelihood zero.
* Retain labels with >50 positive records in training folds; drop records with no retained label.
  This training-only vocabulary rule differs from the official benchmark's whole-dataset count
  filter. These are development tasks, not directly comparable leaderboard results.
* Stop with a feasibility error if any retained validation label lacks positives or negatives;
  never silently change the macro-AUROC label set. Report supports and patient overlap.
* Screen 24 LHS configurations plus reference with one common seed. Select ranks 1-3, the two
  settings around Q75, the median setting and reference. Duplicate selections are trained once.
* Confirm all selected configurations with five fresh common training seeds. The screened winner
  is fixed before confirmation; its original screening score is excluded from confirmation means.
* Train <=20 epochs, patience 5, existing AdamW/BCE/AMP and training-only normalization.
  Save fold-7 predictions at each best checkpoint for patient-cluster uncertainty estimates.
* Paired hierarchical bootstrap over fresh seeds and patients (2,000 replicates) estimates the
  screened winner minus reference. A resample missing a retained label is marked invalid, not
  scored on a smaller label set; >5% invalid replicates blocks the task assessment.

Operational planning gates, fixed before observing outcomes:

| Decision | Criterion |
|---|---|
| GO | Fresh-seed gain over reference >=0.002, gain's two-sided 95% bootstrap lower bound >0, and confirmed winner-minus-Q75 gap >=3 pooled seed SD |
| REDESIGN | Gain <0.002 and gap <=1.5 pooled seed SD |
| BORDERLINE | All remaining cases; report uncertainty and use a separate development assessment before extending compute |

The 0.002 threshold and SD ratios are planning heuristics, not a clinical threshold, a
non-inferiority margin, a global-optimum certificate or proof of method superiority. Two tasks
share one dataset. A one-fold headroom result cannot establish external validity.
Fresh seeds address training luck, not validation-patient selection bias: fold 7 is used for
screening and confirmation, and the bootstrap is conditional on the chosen configuration.
Its interval is a development diagnostic; it does not adjust for searching 25 configurations.

## Matched attribution controls

New method names preserve historical `mu`:

| Method | Allocation | Stop proposal | Confirmation |
|---|---|---|---|
| adaptive_utility | Adaptive utility scores | Existing utility streak | Common procedure below |
| rr_utility | Round-robin | Same utility streak | Same |
| adaptive_patience | Adaptive utility scores | 16 completed distinct evaluations without incumbent improvement | Same |
| rr_patience | Round-robin | Same patience | Same |
| adaptive_full / rr_full | Adaptive / round-robin | Disabled; cap reached | No stop confirmation |

All four stopping arms share DE parameters, warm-up, blocks, initialization, candidate-training
seeds, random-number stream and 40% floor. Streams match until actions diverge. Patience counts
warm-up evaluations but cannot trigger before warm-up completes; cached duplicates do not
advance it. Stop eligibility is assessed after complete two-trial rounds.

Common confirmation executes one round per block and vetoes if any global-incumbent improvement
occurs; it then reevaluates the incumbent with the same two fresh seeds and stability tolerance.
It exits early on a veto, charging actual work. This differs from legacy `mu`'s per-second utility
veto and is a declared revised method, not a silent change to the exported pilot. The patience
arms isolate the proposal signal with the same confirmation mechanism and cost accounting.

The development experiment `matched` can be simulated with `run_search.py --experiment matched
--synthetic`. Real matched experiments need task-specific T_ref calibration on G4; the 9.8-second
superclass value is never reused blindly on form/rhythm or local hardware.

`calibrate_task.py` registers and measures 16 LHS durations and four configurations x five
noise seeds on the common G4 development workload (1-6/7), saving each evaluation for resume.
The decision notebook runs it for superclass and the selected harder task. The separate
precision pilot calibrates its smaller search workload (1-4/5) for its own development budgets.

`dehb_native` uses official DEHB initialization without charging an external design its search
state cannot ingest. Compare with the legacy adapter as a fairness sensitivity before choosing
the external baseline. BOHB is omitted from the focused revised matrix; retaining a named BOHB
comparator later requires the official implementation.

Budget accounting amendment: the legacy runner stored `min(actual_elapsed, cap)` as elapsed.
Its exported summaries cannot prove zero deadline overshoot. The revised runner reports actual
elapsed and overshoot separately, discards evaluations completed late, and reports interrupted
training time separately from optimizer overhead. Old pilot outputs remain immutable.

The budget clock includes physical wall time plus charged cached initialization evaluations.
Summaries now expose both physical wall time and this accounted search time. Reusing a common
initial result saves physical execution, but every method pays its recorded evaluation cost.
The residual overhead field includes optimization, logging and evaluator cleanup; it is not
an isolated controller-CPU measurement. Report these definitions when claiming time savings.

## Margin and precision

No non-inferiority margin is approved. The primary revised question is gain over a declared
untuned model versus measured search cost. Quality at fixed cost and time to development-defined
validation targets support that question; final held-out AUROC is assessed only after G4.

Fresh-seed headroom predictions can diagnose training and validation-patient uncertainty, but
cannot power P-DEHB on fold 10. A dedicated precision pilot uses folds 1-7 with separate search,
checkpoint and assessment roles, and paired full-system outputs. Its decision depends on paired
differences, not the SD of one fixed CNN. Patients are shared across seeds: adding seeds cannot
eliminate uncertainty from the patient sample.

## Next decision

1. If a harder task passes GO, propose superclass plus that task with matched controls and a
   focused external comparison. Reallocate the original 148-hour projection instead of adding
   experiments to it.
2. If only BORDERLINE results occur, require a separate development assessment; a favourable
   single screening run is insufficient.
3. If both tasks require REDESIGN, consider a broader negative/mixed tuning cost-benefit paper
   if systematic novelty review supports it, or assess Article 2 separately.
4. Complete second-session G4 timing drift, precision pilot and revised
   inventory before G2. The local host has an RTX 3050. Prepare reproducible G4 execution artifacts
   for checks requiring the confirmatory hardware.

Researcher steering: accept the supplied review figures and prioritize experiments. Further
systematic retrieval is outside this decision phase; narrow claims to the measured study.

Mapping source: https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/utils/utils.py
(SCP flag aggregation and minimum-sample filtering). Use this for mapping, not for transferring
leaderboard scores into the new experiment.
