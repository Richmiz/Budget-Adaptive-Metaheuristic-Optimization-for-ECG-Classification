# 06 — Pilot calibration and protocol freeze

**Purpose.** Fix every value the proposal leaves "pilot-defined", using **only policy-development data** (train folds 1–6, validate on fold 7). Each stage has a pre-specified selection rule so that the pilot cannot quietly tune the confirmatory result. Script: `scripts/run_pilot.py`; configuration: `configs/experiments/pilot.yaml`.

| Stage | Where | Determines | Cost (approx.) |
|---|---|---|---|
| P0 Smoke | Local RTX 3050 | Pipeline works on real data | < 1 GPU-h |
| P1 Feasibility | Candidate Colab GPUs | GPU type; final domains | ~0.5 GPU-h per GPU type |
| P2 T_ref | Chosen GPU, 2 sessions | T_ref, budget seconds, compute projection | ~16 + 4 evaluations |
| P3 Noise | Chosen GPU | Re-evaluation tolerance; seed-count check | 20 evaluations |
| P4 Controller sweep | Chosen GPU | τ, κ | 10 runs × 160 T_ref ≈ 1,600 T_ref |
| P5 Budget feasibility | From P2–P4 logs | B_low adequacy; G2 compute check | none |
| P6 Comparator verification | Chosen GPU | DEHB/TPE adapters; BOHB check | short runs |
| P7 Freeze | — | `frozen_protocol.yaml`, git tag | — |

## P0 — Smoke test (local)
`prepare_data.py`. Then, on the pilot partition, train the reference configuration for a few epochs and one short `mu` run with a provisional T_ref (for example 60 s). Pass: no errors, logs complete, AUROC well above 0.5.

## How the pilot is run (13 Oct 2026)

Open `notebooks/COLAB_G4_PILOT.ipynb` in VS Code → **Select Kernel → Colab → New Colab Server → G4** → **Run All**.

The notebook starts `scripts/pilot_driver.py`, which runs P1 → P2 → P3 → proposal → P6 → P4 → summary in the background, logs to `My Drive/bamo_runs/pilot/pilot_log.txt` and writes `pilot_summary.json`. Rerunning resumes at the first unfinished step. Later, run the same notebook with `SESSION = 2` in a new runtime for the P2 drift check.

## P1a — Speed check (`run_pilot.py speed`, ~5 min)
2-epoch timing of the 16 P2 configurations at the benchmark's training-set size (folds 1–7; fold 7 serves as the validation role, timing only), after one warm-up epoch. Prints median s/epoch, a T_ref estimate (× 17 epochs) and GPU-hour / compute-unit projections for the full and reduced designs. Not guarded, so it can compare GPU types. Output: `p1_speed.json`.

## P1 — Feasibility and GPU choice
`run_pilot.py feasibility` trains the domain corners (largest capacity with stride 1 at batch 32 and 128; smallest capacity) for 2 epochs. It records seconds per epoch and peak memory.

**Rules**
* If a corner runs out of memory on the candidate GPU, shrink the domain (for example drop `stem_stride = 1` with `blocks_per_group = 3`) **or** choose a larger GPU. Record which (D-new).
* Choose the Colab GPU type by, in order:
  1. availability on your plan, so you can get it repeatedly
  2. projected all-in GPU-hours × the compute-unit rate within budget
  3. speed

  Write the result to `hardware.required_gpu`.

## P2 — T_ref
`run_pilot.py tref` runs 16 Latin-hypercube configurations through the single-fidelity evaluator (20 epochs, patience 5). **T_ref = median duration.**

**Drift check:** in a second session on the same GPU type, re-run 4 of the 16 configurations. If the median ratio between sessions is outside [0.9, 1.1], investigate (data on local disk? other processes? throttling?) before freezing.

**Outputs:** `p2_tref.json` with the quartiles and the projected GPU-hours, which feed G2.

## P3 — Seed noise and re-evaluation tolerance (D21)
`run_pilot.py noise` runs the reference configuration plus 3 LHS configurations, 5 training seeds each. It reports:
* the pooled seed SD of validation AUROC
* the 95th percentile of |m_seed − m_first|, which mimics re-evaluating an incumbent

**Pre-specified rule (proposed, confirm in D21):** tolerance = the **95th percentile of |m_seed − m_first|**. Re-evaluation then rejects a stop about 5% of the time when the incumbent is merely average-lucky.

## P4 — τ and κ sweep
`run_pilot.py sweep --t-ref <P2> --reeval-tol <P3>` varies one factor at a time at B_mid on the pilot partition, 2 seeds each:
* τ ∈ {1e-4, 5e-4, 1e-3} at κ = 1
* κ ∈ {0.5, 2} at τ = 5e-4

**Pre-specified selection rule:**
1. Discard settings where any run ends in `budget_during_warmup`, or where the confirmation was infeasible in more than half the runs.
2. Among the remaining settings, find the best mean pilot-validation AUROC.
3. Keep the settings whose mean is within **0.002** of that best (less than half the non-inferiority margin).
4. Choose the one with the **lowest mean budget used**.
5. Ties, or nothing clearly better → keep the defaults τ = 5e-4, κ = 1.

The pilot is too small to optimize τ precisely. The aim is to rule out settings that never stop or that stop pathologically early.

## P5 — Budget feasibility and seed count
* **Warm-up and confirmation in B_low:** median (8 + 24 evaluations + confirmation ≈ 10 evaluations) must be ≤ 0.8 × B_low. If not, increase B_low as the proposal allows, and record it.
* **G2 compute check:** projected all-in GPU-hours (doc 10 §10.2) ≤ available capacity for W9–W17.
* **Seed count:** use the P3 seed SD as a proxy for the paired SD of final AUROC differences, then `seeds_for_noninferiority(sd)`. If more than 10 seeds are needed, add seeds **now** (before any test access), or plan to report imprecision explicitly (proposal §3.7.2).

## P6 — Comparator verification
See doc 05 §5.4. Run TPE and DEHB for a short real budget (for example B = 20 T_ref) on the pilot partition, and run the BOHB check on the synthetic objective.

## P7 — Freeze
1. Fill in the FREEZE values:
   * `hardware.required_gpu`
   * `budgets.t_ref_seconds`
   * `controller.reeval_tolerance`
   * `controller.tau` and `kappa` if they changed
   * domain edits
   * seed list if extended
2. Resolve every "needs your call" decision in doc 08.
3. `python scripts/lock_protocol.py freeze` → `protocol/frozen_protocol.yaml` with a config hash.
4. `git commit` + `git tag protocol-frozen`. Optional: deposit the frozen protocol and doc 07 on OSF as a preregistration.
5. From then on, `run_search.py` refuses confirmatory runs if `configs/` no longer match the frozen hash.

Pilot runs live under `runs/pilot*/` and are **never** pooled with confirmatory runs.
