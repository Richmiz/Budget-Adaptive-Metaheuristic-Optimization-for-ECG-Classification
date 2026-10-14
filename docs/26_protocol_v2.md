# 26 — Protocol v2: is tuning worth its compute?

**Status: draft for approval (14 Oct 2026). Not frozen.** Once approved, this file is frozen with the configs (`scripts/lock_protocol.py freeze`, git tag `protocol-v2-frozen`). Before fold 10 is opened, this file and `scripts/analyze_v2.py` are hashed into the analysis lock. Plan summary: [doc 25](25_study_plan_v2.md). Development evidence: docs 21–24. This protocol replaces protocol v1.1 (doc 16) and the v1.2 draft (doc 19).

## 1. Questions
1. **Gain.** How much does hyperparameter search improve held-out discrimination over (a) the original default D0 and (b) a sensible task default?
2. **Cost.** How does that gain depend on the search budget (80 vs 160 time units)?
3. **Stopping.** How much search compute does the early-stopping tuner P use, compared with tuners that use their whole budget, and what accuracy does it reach?

## 2. Data and tasks
- **Source:** PTB-XL v1.0.3 (PhysioNet), 12-lead, 100 Hz, 10 s per record. Raw files are checked against PhysioNet SHA256SUMS. Official `strat_fold` assignment; no patient is in more than one fold (audited).
- **Diagnosis task:** 5 superclasses (NORM, MI, STTC, CD, HYP). Benchmark mapping (Strodthoff et al., 2021): every diagnostic SCP statement contributes its class, and likelihoods are not thresholded. Records with no superclass are excluded (411). **21,388 records, 18,617 patients.**
- **Rhythm task:** 7 frozen labels (AFIB, PACE, SARRH, SBRAD, SR, STACH, SVARR), taken from the record's SCP rhythm statements, with no likelihood threshold.
  - The labels were chosen from training folds 1–6 (more than 50 positives).
  - AFLT and BIGU would pass that rule on folds 1–7 but are excluded and reported.
  - Records with none of the 7 labels are excluded (865).
  - **20,934 records, 18,221 patients.** Every label has positives in every fold; fold 10 has 14 SVARR and 28 PACE positives.
- **Prepared arrays (SHA-256)** are rebuilt on every Colab runtime and must match:

  | Task | `signals_100hz.npy` | `labels.npy` |
  |---|---|---|
  | Diagnosis | `630948cd…` | `3a95ec71…` |
  | Rhythm | `782c5475…` | `0bb63eb5…` |

- **Normalization:** per-lead mean and SD from the training role only.

## 3. Splits (patient-separated)
| Role | Folds | Used for |
|---|---|---|
| Search | train 1–7, validate 8 | candidate training and search fitness (validation macro AUROC) |
| Final | train 1–8, early stopping and F1 thresholds on 9 | final models |
| **Test** | **10** | **one evaluation, after the analysis lock** |

Folds 8–10 were not used in any development experiment (docs 21–25 used folds 1–7).

## 4. Model and search space
- **Model:** 1-D ResNet with fixed parts (12 leads; widths 32/64/128 × width multiplier; group strides 1/2/2; stem kernel 15). Training: AdamW, BCE-with-logits, no class weighting, fp16 autocast.
- **Search space (9 settings):** learning rate 1e-5–3e-3 (log); batch size {32, 64, 128}; weight decay 1e-6–1e-2 (log); dropout 0–0.5; width multiplier {0.5, 1, 1.5}; blocks per group {1, 2, 3}; kernel {5, 9, 15}; dilation {1, 2}; stem stride {1, 2}. The ranges are unchanged by the step-1 check (doc 25).
- **Candidate evaluation:**
  - Random search and P: at most 20 epochs, patience 5.
  - DEHB: fidelities 5/10/20 epochs, η = 2.
  - Search fitness is validation macro AUROC at the best epoch.

## 5. Arms (per task)
| Arm | Description |
|---|---|
| **D0** | Original default: lr 1e-3, batch 64, weight decay 1e-2, dropout 0.2, width 1.0, 2 blocks per group, kernel 9, dilation 1, stem stride 2. |
| **D1** (rhythm only) | D0 with kernel 15 and dilation 2. **Chosen from development data** (docs 24–25). For diagnosis the task default is D0, because D1 lowered diagnosis AUROC by 0.0097 (doc 25). |
| **Random search** | 8 shared Latin-hypercube initial configurations, then uniform random sampling, until the cap. |
| **DEHB** | Official `dehb` 0.1.1 through ask/tell, native initialization (no shared design), until the cap. |
| **P** (`adaptive_utility`) | DE (population 8, F 0.5, CR 0.7) with the block controller: allocation by improvement per unit compute (κ 1, window 8, 3 warm-up rounds per block, mandatory probe after 8 decisions). It stops when every block's utility stays below τ = 5e-4 for 3 decisions, with each block run ≥ 3 rounds and ≥ 40% of the budget used. A stop is confirmed only if one more probe per block finds no improvement and 2 re-evaluation seeds stay within the task tolerance (diagnosis 0.00438; rhythm 0.01150). This is the version run on G4 on 14 Oct. |

Random search and P share each search seed's initial design. Its evaluations are cached and charged to every method that uses them.

## 6. Budgets and timing
- **Time unit T_ref** (median duration of 16 fixed evaluations on G4, 14 Oct; `configs/calibration_definition_v1.json`): diagnosis **11.170414 s**, rhythm **12.954911 s**.
- **Budgets:** low = 80 T_ref, mid = 160 T_ref. That gives diagnosis 893.6 s and 1,787.3 s; rhythm 1,036.4 s and 2,072.8 s.
- **Cost accounting:** caps are wall-clock, including charged cached evaluations. Results that finish after the cap are discarded.
- **Hardware:** Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition); the runner refuses any other GPU. One job at a time per GPU.
- **Session check:** every session first runs the timing check against the 13 Oct baseline. A median ratio outside 0.9–1.1 stops the session before any search starts.

## 7. Replication and order
- **Seeds:** 10 search seeds (1001–1010) per task × tuner × budget, giving **120 searches**. Final-model training seed = `derive_seed("final", search seed)`, so all arms in a task are paired by seed.
- **Order:** each task's run matrix is randomly ordered (seed 20261007). The two tasks' matrices are interleaved, so methods and tasks are spread over sessions.
- **Interruptions:** an interrupted search restarts from scratch.
- **Failures:** a search that ends without a valid incumbent is reported as a failure, not rerun. The analysis uses the seeds that have a test prediction for every arm of the task (complete cases), and reports which seeds were excluded.

## 8. Final models
- Each completed search's incumbent, plus D0 (and D1 for rhythm) per seed, is retrained on folds 1–8 (≤ 50 epochs, patience 8 on fold 9). Per-label F1 thresholds are set on fold 9.
- Total: 120 tuned + 30 default = **150 models**.
- Model choice never uses fold 10.

## 9. Analysis (`scripts/analyze_v2.py`)
**Primary family (per task): test macro AUROC gain of each tuner × budget over the task default** (D1 for rhythm, D0 for diagnosis). That is 6 contrasts per task.
- Each contrast is the mean over seeds of the paired difference.
- 95% CI from a hierarchical bootstrap that resamples patients (with all their records) and seeds, with 10,000 replicates.
- One-sided bootstrap p-value for gain ≤ 0, Holm-adjusted within the task.
- A gain is called **supported** when the Holm p is below 0.05. Estimates and intervals are reported for all contrasts, whatever the p-value.

**Secondary estimates (no hypothesis tests):**
- Rhythm: D1 − D0, and each tuner × budget − D0.
- P − DEHB and P − random at each budget (AUROC difference with CI), with the geometric-mean search-time ratio and its seed-bootstrap CI.
- Search G4 hours per arm; P's stop reasons and fraction of budget used.
- Per-class test AUROC per arm, with test positive counts. This matters for the rare rhythm classes (SVARR 14, PACE 28).
- Validation: anytime efficiency and final-model fold-9 AUROC.

**Figure:** per task, test macro AUROC against search compute (mean ± 95% CI over seeds), with defaults drawn as horizontal lines.

**Not tested:** non-inferiority of P against DEHB. Development data showed the margin cannot be supported reliably (docs 22–24).

## 10. Locks and integrity
1. **Freeze**, before any main run: `lock_protocol.py freeze` records the merged config and one hash per task. Then a git commit and tag `protocol-v2-frozen`. The Colab runner refuses to search unless the bundled config matches the frozen hash for its task.
2. **Analysis lock**, after all 150 final models and before the test notebook is built: `lock_protocol.py analysis` hashes this file and `scripts/analyze_v2.py`. The test notebook embeds the lock, and `evaluate_test.py` refuses to run without it.
3. **Fold 10** is evaluated once. Any later deviation is reported as an amendment with its reason.

## 11. Compute and schedule
| Item | G4 hours |
|---|---|
| Search caps, 120 runs | 48.3 (about 42 expected, because P stops early) |
| Final models (150) | ≈ 2 |
| Session setup (≈ 10 sessions) | ≈ 3 |
| **Total** | **≈ 45–53 h ≈ 400–470 compute units** |

Schedule: lock by 25 Oct 2026; main runs 26 Oct–15 Nov; test by 22 Nov; analysis by 6 Dec; submission target Feb–Mar 2027.

Not part of this protocol: P without stopping (`v2_addon`, about 11 h). If it is run, it is exploratory and reported as such.

## 12. Limitations stated in advance
- One dataset and one CNN family.
- D1 was chosen after seeing development results, which favours D1 in the rhythm contrasts.
- Timing is on one GPU type.
- Ten seeds measure search variability on one fixed test fold, not dataset variability.
- Rare rhythm classes make per-class estimates imprecise.
