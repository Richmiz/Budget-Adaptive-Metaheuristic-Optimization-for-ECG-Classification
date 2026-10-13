# 02 — Experimental protocol

This expands proposal Chapter 3 into an executable procedure. Code references point to the implementation; `Dxx` refers to [08_decision_register.md](08_decision_register.md).

## 2.1 Design

| Factor | Levels | Notes |
|---|---|---|
| Method | random, tpe, de, hyperband, bohb, dehb, **mu** (P) | Doc 05 |
| Search-time cap B | low = 80 T_ref, mid = 160 T_ref, high = 320 T_ref | Caps, not evaluation counts |
| Search seed s | 1001–1010 (pre-specified in `configs/base.yaml`) | Paired across methods |
| Ablation variant (B_mid only) | mu_nostop, rr_nostop, mu_nocost, mu_noprobe, mu_nojoint, mu_noconfirm | Doc 04 §4.8 |

**Pairing.** For a given seed s, every method gets the same eight initial configurations, the same candidate-training seed, the same data partition and the same GPU type. Methods differ only in how they search.

## 2.2 Data and partitions (summary; full detail in doc 03)

| Partition | Train (strat_fold) | Validation | Test | Used for |
|---|---|---|---|---|
| `pilot` | 1–6 | 7 | — | Calibration and policy development only |
| `search` | 1–7 | 8 | — | Candidate training and search fitness |
| `final` | 1–8 | 9 | 10 | Final retraining, threshold selection, locked test |

Fold 8 is never used as an estimate of final performance. Fold 9 never influences search. Fold 10 is read once, after the analysis lock.

## 2.3 Model

1-D ResNet ([resnet1d.py](../src/bamo/models/resnet1d.py)):
* **Stem:** Conv1d(12 → w₀, k = 15, stride = `stem_stride`), followed by BN and ReLU.
* **Residual groups:** three groups with base widths 32/64/128 × `width_multiplier`, rounded to a multiple of 8. Each group has `blocks_per_group` basic blocks (two convs, kernel `kernel_size`, dilation `dilation`). Group strides are 1/2/2 (D06).
* **Head:** global average pooling, dropout (`dropout`), then Linear → 5 logits.
* **Training:** AdamW (`learning_rate`, `weight_decay`), BCE with logits, unweighted (D03), fp16 autocast on CUDA (D07), drop last incomplete batch.
* **Checkpoint selection:** best validation macro AUROC (threshold-free).

## 2.4 Search space

| Block | Parameter | Domain | Encoding |
|---|---|---|---|
| A | learning_rate | log-uniform [1e-5, 3e-3] | log scale |
| A | batch_size | {32, 64, 128} | 3 equal bins |
| B | weight_decay | log-uniform [1e-6, 1e-2] | log scale |
| B | dropout | [0, 0.5] | linear |
| C | width_multiplier | {0.5, 1, 1.5} | bins |
| C | blocks_per_group | {1, 2, 3} | bins |
| C | kernel_size | {5, 9, 15} | bins |
| C | dilation | {1, 2} | bins |
| C | stem_stride | {1, 2} | bins |
| J | all nine | — | — |

All methods search the full nine-dimensional domain. Blocks restrict only the controller's DE variation. Domains are provisional until P1 (doc 06) and are frozen at G2.

## 2.5 Candidate evaluation

| Property | Single-fidelity evaluator | Multi-fidelity evaluator |
|---|---|---|
| Methods | mu (+ ablations), de, random, tpe | hyperband, bohb, dehb |
| Max epochs | 20 | rungs 5 / 10 / 20 (η = 2) |
| Early stopping | patience 5 on fold 8 (search partition) | patience 5 within each rung |
| Fitness | best-epoch macro AUROC on the validation role | same; only 20-epoch results can become the incumbent |
| Weight transfer | none | none: each rung retrains from scratch (D25) |

* **Cost measured:** wall time from just before model construction to after the final CUDA sync. It includes initialization, training, per-epoch validation and cleanup.
* **Interruptibility:** `clock.check()` runs before every mini-batch and every validation pass. At the cap the evaluation is abandoned and can never become the incumbent.
* **Failures:** CUDA OOM → `infeasible` (cost charged, no reward, D35). Non-finite loss → `diverged`; the best earlier checkpoint stays valid if one exists (D34). Other runtime errors → `failed`, logged with the error text.
* **Caching (D11):** a result is reused only for an identical decoded configuration, training seed, partition and fidelity. Within a run, a re-proposed identical configuration costs nothing new. DE trials that decode to their target's configuration are regenerated, and after 10 attempts one active coordinate is forced into a different value.

## 2.6 Seeds

| Seed | Derivation | Shared across methods? |
|---|---|---|
| Search seed s | Pre-specified list | — |
| Initial design (8-point Latin hypercube) | `derive_seed("init", s)` | Yes |
| Candidate training seed | `derive_seed("train", s)`, constant within a run (D08) | Yes |
| Re-evaluation seeds (2) | `derive_seed("reeval", s, i)` | Yes |
| Method RNG | `derive_seed("method", method, s)` | No (method-specific) |
| Final-model training seed | `derive_seed("final", s)` | Yes |

## 2.7 Budget accounting

**Charged to B:**
* all candidate evaluations, including failed and interrupted ones (up to the cap)
* controller computation between rounds (D18)
* confirmation probe rounds and incumbent re-evaluations
* the shared initial design, which is charged at its recorded cost even when served from the cache (D09)
* model-fitting time of TPE, BOHB and DEHB

**Not charged to B, reported separately:**
* session start-up and data loading to the GPU
* final-model training
* deployment profiling and test evaluation

Each run records elapsed, unused budget, charged, evaluation wall time and overhead (`summary.json`).

## 2.8 Run matrix and execution

1. `scripts/run_search.py --experiment main` creates `runs/main/run_matrix.csv`: the full factorial in a random order fixed by `order_seed` (D36).
2. Parallel Colab sessions take disjoint **seed shards** (`--shard i/n`). Each seed's shared-initial-design cache then has a single writer.
3. A job is complete when `summary.json` exists. An interrupted job restarts from scratch (D39).
4. Matched caps do not mean matched consumption. Actual time used is an outcome.

## 2.9 Session checklist (every Colab session)

- [ ] GPU type equals `hardware.required_gpu` (the runner enforces this)
- [ ] `frozen_status()` is `match` (confirmatory phase)
- [ ] Processed data copied to the VM's local disk; no other workload on the runtime
- [ ] Shard assignment logged in the run log (sheet or `runs/<exp>/sessions.md`)
- [ ] After the session: `analyze.py search` shows the new completed jobs; failures noted

## 2.10 Final models

For each completed run, the incumbent configuration is retrained on folds 1–8 (≤ 50 epochs, patience 8 on fold 9, checkpoint by fold-9 macro AUROC). Per-label F1 thresholds are chosen on fold 9 (grid 0.01–0.99, ties → lowest, D33). The same procedure applies to:
* the untuned reference configuration (10 seeds)
* the shadow-stop incumbents of `mu_nostop` (D24)

## 2.11 Deployment profiling

For each distinct selected architecture, on the frozen GPU and on CPU:
* batch-one latency (median and p95 over 200 runs, after 20 warm-up runs)
* parameter count, fp32 size, peak GPU memory

Energy is reported only if a sound measurement method is available.

## 2.12 Outputs per run

`evaluations.jsonl`, `decisions.jsonl` (controller), `trajectory.csv`, `summary.json`, and after WP8 `final/` (`model.pt`, `final.json`, `val_predictions.npz`, later `test_predictions.npz`, `test_report.json`). Schemas are in doc 09.
