# 13 — Progress log

Dated record of what was done, what was found, and what changed. Newest entries first. Numbers here come from local development runs unless stated otherwise. They are **not** confirmatory results and must not feed T_ref or any frozen value (those come from the Colab GPU, doc 06).

## 2026-10-14 — G4 decision export verified; matched follow-up prepared

* Reproduced the G4 development results from the researcher's completed export: 227 file hashes,
  executed bundle `21dce7d4…`, registered precision plan `ce2f24c988f8` and 15 prediction hashes.
  Assessment mean AUROC: reference 0.93260, P 0.95177, native DEHB 0.95910. P−DEHB −0.00733
  [−0.02120, +0.00680]; one-sided lower −0.01870. NI is not established at any proposed margin.
* P/DEHB accounted search-time ratio 0.572 [0.488, 0.698]. The validation difference at P's
  stopping cost is +0.00011, but all five DEHB prefix models differ from the final models and
  require independent assessment refits. No accepted result exceeded its cap; actual overhead
  overshoot is reported. Confirmation did act: seven confirmations, three probe vetoes and
  three stability rejections across 13 attempts.
* Recorded common-workload T_ref: superclass 11.170414 s, rhythm 12.954911 s. The calibration
  configurations and training seed were already fixed; retained their measured definition.
  Identified that refiltering rhythm labels on 1–7 would add AFLT and BIGU. Proposed retaining
  the seven tested labels in the final task definition.
* Prepared a separate G4 notebook for 15 matched searches and at most 30 new refits. It reuses
  original P/DEHB/reference evidence, original initialization and settings; preselects DEHB
  at P's cost and fixed 40% prefixes before assessment. Search caps total 5.81 G4 hours;
  refits/setup are extra. **The new G4 experiment has not run.**
* Local suite: 76 tests pass. Input preflight passed. A mocked end-to-end orchestration check
  exercised 45 prediction artifacts, reuse, seven contrasts and output handling without new GPU
  experiments; its outputs are synthetic and cannot be used as scientific evidence. Notebook
  packaging/static code verification passed. Script progress is now relayed into cells and
  saved to Drive logs.
* Revised the provisional full inventory to 180 searches, 96.50 measured search-cap hours,
  plus up to 340 refits and other costs. Full study remains gated. Supplied review counts are
  accepted; further literature retrieval is not requested. Formal assessor statement: doc 23.

## 2026-10-13 — Assessor decision phase implemented; actual task pilots

* Preserved the post-pilot v1.1 source in baseline commit `feff26a`; revisions on
  `codex/decision-pilot`. G2 and the original full matrix are held.
* Recomputed the immutable pilot-export audit: 81/81 file hashes match; 9/10 sweep runs
  confirmed a stop. Original selection was tau=0.001; retaining 0.0005 used a post-pilot
  tie amendment. Old elapsed values were clamped, so exact cap compliance is unverified.
* Implemented six matched allocation/stopping arms, common improvement-veto confirmation,
  native DEHB initialization, actual deadline/overshoot accounting, separate physical versus
  charged time, late-result rejection and a guard against premature confirmatory execution.
  Completed 60 synthetic matched-control jobs. These establish execution, not superiority.
* Registered form/rhythm development headroom plan `d3e8f18d5470` before training:
  24 LHS configurations + reference per task; selected settings confirmed with five fresh seeds.
  Training/validation folds 1-6/7 only. New task arrays use training-derived label vocabularies.
* Form preparation: 6,231 records, 15 labels, no train/validation patient overlap. Form
  completed: reference mean 0.89919; fixed winner 0.89449; gain -0.00470, conditional CI95
  [-0.01294, 0.00266]; top-gap/noise 2.34; registered verdict BORDERLINE. Other repeated
  settings perform better; they are exploratory and do not replace the registered winner.
* Rhythm: 14,665 records, 7 labels, no patient overlap. Completed: reference mean 0.93998;
  fixed winner 0.96166; gain +0.02168, conditional CI95 [+0.01144, +0.03225]; top-gap/noise
  6.36; registered verdict GO. Recommend superclass + rhythm for the G4 development checks.
  Both tasks completed 115 training evaluations (50 screening, 65 confirmation); all were valid.
* Prepared the G4 drift and paired development precision workflow (search 1-4/5, refit 1-5/6,
  assess 7), matched real-data smoke, default anchors and cost/precision diagnostics. A local
  drift invocation correctly refused the RTX 3050; no G4 gate is claimed complete.
* Targeted SciSpace/Consensus discovery and primary-source checks found SHCHO cooperative
  deep-CNN HPO and prior ECG DE/default comparisons. The researcher requested accepting the
  supplied SLR figures and prioritizing experiments; further review retrieval is outside scope.
* Verification: full suite **71 tests pass**, including GPU tests, after the pilot completed.
  All 57 Python sources parsed. G4 notebook code-cell syntax, source bytes and ZIP hash verified.
  Formal statement, fixed-winner outcomes, all-setting figure and hashed evidence: doc 21.
  Next notebook defaults to rhythm with development calibration/precision enabled; G4 is
  required. G2 stays held until those results and the final inventory/analysis plan are reviewed.

## 2026-10-13 — B + C adopted; Protocol v1.1 for assessment

* **Researcher chose B + C** (D43, D44):
  * `mu_floor20` and `mu_floor60` (P with 20% / 60% minimum-time floors)
  * `de_patience` (standard DE, stop after 16 evaluations without improvement, same 40% floor)

  Experiment `configs/experiments/stopping.yaml`: 30 runs at B_mid. Analysis plan doc 07 §7.5b; H2 is interpreted together with these checks.
* Code: `PatienceDE` in `methods/simple.py`, registered. Tests extended to **64 (all pass)**, including "every stopper respects its floor". A synthetic dry run of the stopping experiment works mechanically.
* Pilot-proposed values entered in configs (T_ref 9.8 s, tolerance 0.0035); they are frozen at G2 after the drift check.
* Compute updated (doc 01 §1.4, doc 10): remaining ≤ 54,330 T_ref ≈ ≤ 148 G4-hours ≈ 1,320 CU.
* **`docs/16_protocol_v1_1.md`:** consolidated post-pilot protocol for the assessor, covering design, evidence so far, changes from v1.0, limitations, decisions required before freezing, compute and timeline. Stale status lines in README and docs fixed.

## 2026-10-13 — Pilot decision-log analysis (export zip verified)

`bamo_pilot_results.zip` downloaded by the researcher, moved to `runs/colab/` and extracted to `runs/colab/pilot_2026-10-13/`. **81 files, 0 SHA-256 mismatches** against the Colab manifest; bundle `c4c8efd3c045`.

Stop timing per sweep run, as a fraction of the B_mid budget (`decisions.jsonl`):

| Run | Warm-up ends | All U_k < τ first | Streak ≥ 3 (stop allowed if no 40% floor) | Stop proposed | Ended | Reason |
|---|---|---|---|---|---|---|
| τ 1e-4 / 9001 | 26% | 94% | 98% | — | 100% | budget |
| τ 1e-4 / 9002 | 26% | 51% | 53% | 53% | 57% | confirmed stop |
| τ 5e-4 κ0.5 / 9001 | 27% | 28% | 32% | 41% | 50% | confirmed stop |
| τ 5e-4 κ0.5 / 9002 | 26% | 36% | 38% | 42% | 48% | confirmed stop |
| τ 5e-4 κ1 / 9001 | 26% | 27% | 31% | 40% | 49% | confirmed stop |
| τ 5e-4 κ1 / 9002 | 26% | 39% | 43% | 43% | 48% | confirmed stop |
| τ 5e-4 κ2 / 9001 | 26% | 30% | 33% | 40% | 50% | confirmed stop |
| τ 5e-4 κ2 / 9002 | 26% | 43% | 45% | 45% | 53% | confirmed stop |
| τ 1e-3 κ1 / 9001 | 26% | 27% | 31% | 41% | 49% | confirmed stop |
| τ 1e-3 κ1 / 9002 | 26% | 30% | 34% | 42% | 48% | confirmed stop |

**Findings**
1. **The 40% floor sets the stop time in most runs.** In 6 of 8 runs with τ ≥ 5e-4, the controller's own rule allowed stopping at 31–38% of the budget, and the stop was proposed only when the 40% floor allowed it (40–42%). Confirmation then adds ~6–9 points, so runs end at 48–53%. In the other 2 runs (43%, 45%) the controller's own rule was later than the floor.
2. **The search plateaus right after warm-up.** Warm-up (8 shared initial + 24 warm-up evaluations) ends at ~26% of the budget. After it, the incumbent improved **0–1 times** per run (τ ≥ 5e-4). Even the full-budget run (τ 1e-4 / 9001) improved only 4 times in 37 search rounds.
3. **Adaptive allocation barely acts before stopping:** 9–12 search rounds per run, of which 3 are mandatory probes and only 1–4 are switches.
4. **Confirmation never reversed a stop:** 8/8 re-evaluations stable, every first proposal confirmed. It cost ~6–9% of the budget without changing an outcome in this pilot.
5. **Cost of patience:** τ 1e-4 (mean 78% of budget) reached mean val AUROC 0.9423 vs 0.9408–0.9411 for the stopping settings, a +0.0013 gain (~1 seed SD) for ~30 points more budget.

**Implications for H2 (to decide before the freeze, G2):** the ~50% search-time saving of P vs DEHB will largely reflect the fixed 40% floor plus the confirmation cost, not the utility signal. Options:
* report this transparently and add a **sensitivity analysis of `min_budget_fraction`** (e.g. 0.2 / 0.4 / 0.6);
* add the **naive stopping baseline** (doc 14, R2), so the controller's contribution is separated from "stop when progress flattens".

Both are cheap at T_ref 9.8 s.

## 2026-10-13 — Pilot session 1 on Colab G4 (`COLAB_G4_PILOT.ipynb`, bundle `c4c8efd3c045`)

Run by the researcher from VS Code; 15:07–17:37 local time (08:07–10:37 UTC), **2 h 30 min**. Data rebuilt on the runtime and identical to the local arrays. Results: `My Drive/bamo_runs/pilot*` (export zip pending).

**Proposed frozen values (P2, P3, P5)**
* **T_ref = 9.8 s** (median of 16 full single-fidelity evaluations; the stage took 3 min 13 s).
* **Re-evaluation tolerance = 0.0035** (q95 |m_seed − m_first|, D21 recommended rule). Pooled seed SD on G4: **0.0012**.
* Warm-up + confirmation fit in B_low: **yes**.
* Compute projection at T_ref 9.8 s: full design ≈ 51,200 T_ref ≈ **139 GPU-h** (≈ 1,240 CU at 8.9 CU/h).

**P6: every method on real data, 20 T_ref, seed 9001.** All stayed within the cap. Multi-fidelity methods used fidelities {5, 10, 20}; single-fidelity methods used {20}.

| Method | Evals | Best val AUROC | Note |
|---|---|---|---|
| random | 13 | 0.9384 | |
| tpe | 12 | 0.9407 | |
| de | 16 | 0.9410 | |
| hyperband | 20 | 0.9384 | |
| bohb | 16 | 0.9384 | |
| dehb | 14 | 0.9384 | |
| mu | 16 | 0.9384 | still in warm-up (expected at 20 T_ref) |

0.9384 is the best of the 8 shared initial settings.

**P4: τ/κ sweep, B_mid (160 T_ref), seeds 9001/9002**

| Setting | Mean val AUROC | Mean budget used | Termination |
|---|---|---|---|
| τ 1e-4, κ 1 | 0.9423 | 78% | budget (100%), confirmed stop (57%) |
| τ 5e-4, κ 0.5 | 0.9408 | 49% | confirmed stop ×2 |
| **τ 5e-4, κ 1 (default)** | 0.9411 | 49% | confirmed stop ×2 |
| τ 5e-4, κ 2 | 0.9409 | 51% | confirmed stop ×2 |
| τ 1e-3, κ 1 | 0.9408 | 49% | confirmed stop ×2 |

* The automatic rule picked τ 1e-3 by a fraction of a percentage point of budget. Per docs/06 ("ties, or nothing clearly better → keep the defaults") the **default τ 5e-4, κ 1 is kept**. The driver's tie rule is now fixed (< 5 points of budget = tie).
* **Observation that matters for H2:** with τ ≥ 5e-4, all 8 runs stopped at **48–53% of the budget**, regardless of τ (5e-4 vs 1e-3) and κ (0.5–2). That points to the stopping time being set mainly by the **minimum-budget rule (stop allowed only after 40%)** plus the 3-decision streak and confirmation rounds, rather than by τ. To verify, read the decision logs (`decisions.jsonl`: when the stop was first proposed) from the export zip.
* Confirmation never vetoed a stop in τ ≥ 5e-4 runs (all `utility_stop_confirmed` on the first proposal, to be verified in logs).

## 2026-10-13 — Decision: continue the original design; pilot ready for Colab G4

* **Decision (researcher):** keep Article 1 as proposed and continue the pilot on Colab G4 (D42). Docs 14–15 remain the record of the risks.
* **New `scripts/pilot_driver.py`:** runs the whole pilot (docs/06) as one resumable job.
  * Order: P1 feasibility → P2 T_ref → P3 noise → proposal (T_ref = median; tolerance = q95 |m_seed − m_first|, D21 recommended; P5 warm-up check) → P6 all 7 methods at 20 T_ref → P4 τ/κ sweep (5 settings × 2 seeds at B_mid) → P4 selection rule → `pilot_summary.json`
  * Session 2 (`--session 2`) is the T_ref drift check.
  * Finished stages and runs are skipped on rerun.
* **Local smoke test** (`--dev`: any GPU, 2-epoch evaluations): all stages complete in ~4 min; a rerun skips everything (1.4 s).
* **New notebook `notebooks/COLAB_G4_PILOT.ipynb`** (bundle `c4c8efd3c045`): mounts Drive, rebuilds the data, starts the driver in the background, shows progress every minute, and exports `My Drive/bamo_pilot_results.zip`. Built-in checks: all cells compile; 54/54 tests pass from the extracted bundle.
* **Expected cost:** about 7 h on G4 ≈ 60–65 CU (sweep ≈ 1,600 T_ref; checks ≈ 140 T_ref; P2/P3 ≈ 36 evaluations).

## 2026-10-08 — Headroom check (doc 14 R1), local RTX 3050, pilot data

`scripts/headroom_check.py`: 24 Latin-hypercube settings + reference, one seed each; plus the reference and lhs00 retrained with 5 seeds each. Train folds 1–6, validate on fold 7. Output: `runs/local_dev/pilot/headroom.json`.

* **Spread across settings** (validation macro AUROC): best **0.9416**, Q75 **0.9360**, median **0.9323**, worst **0.8950**.
* **Seed noise:** SD **0.0011** (reference 0.0012, lhs00 0.0011). Five runs of the reference: 0.9341–0.9370.
* **Top-end gap** (best − Q75) = **0.0057 = 5.0× noise SD** → **verdict GO**: tuning moves AUROC measurably more than chance.
* **Learning rate dominates:** lr < 1e-4 averages 0.921 (n = 10); lr ≥ 3e-4 averages 0.936 (n = 11, SD 0.0026, best 0.9416). Among sensible learning rates the remaining spread is about 2.4× noise.
* The best setting was a *small* network (width 0.5, 3 blocks/group, batch 32, lr 1.1e-3). Capacity is not the main lever.
* **Implications:**
  1. The study can distinguish search methods, but the achievable gain at the top is modest (~0.005–0.006 AUROC).
  2. **The provisional non-inferiority margin (0.005) is about the size of the entire top-end headroom**, so a method that stops at a Q75-quality setting would still pass. Tighten D30 to ~0.002–0.0025 (≈ 2× seed SD).
  3. Seed noise is small, so fewer than 10 seeds may suffice. Re-check with search-level variability after the pilot.
  4. Block A (learning rate, batch size) will dominate early gains, which gives the block-allocation controller something real to exploit.

## 2026-10-08 — Colab G4 session 1 (`COLAB_G4_SETUP.ipynb`, bundle `4d7436f6f33b`)

Run by the researcher from VS Code on a Colab G4. Outputs are saved in the notebook file and in `My Drive/bamo_runs/` (`pilot/p1_speed.json`, `pilot/concurrency.json`, `environment/`).

* **Runtime:** NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GiB; 48 CPUs; 177 GiB RAM; torch 2.11.0+cu130. Drive mounted from the VS Code kernel with `drive.mount`. The device string matches `hardware.required_gpu = "RTX PRO 6000"`.
* **Data:** PTB-XL 100 Hz downloaded from the S3 mirror at 10.4 MB/s (43,598 files, 0 failures) and rebuilt; **arrays byte-identical** to the local preparation. Total 74 s.
* **Tests:** 54/54 pass on the G4.
* **Speed (P1a), single process, 16 LHS configurations, 2 epochs:** median **0.80 s/epoch** (min 0.30, max 1.62), about 5× the RTX 3050 (4.0). Batch-32 models remain overhead-bound (~1.2 s/epoch, ~2.6 ms/step). **T_ref estimate ≈ 14 s** (×17 epochs; P2 will measure full evaluations).
* **Projection, sequential, at 8.9 CU/h:** full design **193 GPU-h ≈ 1,720 CU**; reduced **145 GPU-h ≈ 1,290 CU**.
* **Concurrency probe (K processes sharing the G4), throughput gain vs K = 1:**

| Model | K = 4 | K = 8 | K = 16 | K = 24 |
|---|---|---|---|---|
| reference (w1, b2, bs64) | ×1.6 | ×1.6 | ×1.5 | ×1.5 |
| small (w0.5, b1, bs32) | ×2.3 | ×2.3 | ×2.3 | ×2.3 |

  The gain plateaus at K ≈ 4. Without CUDA MPS, kernels from different processes time-slice on the GPU instead of running together, so extra processes only fill idle gaps. That is much less than the LMT-Net image models achieved (25 chains), because our per-kernel work is tiny. Each process's s/epoch grows almost linearly with K, which confirms that live wall-clock timing under concurrency would be badly contaminated (D41 option b is not viable).

## 2026-10-08 (later): G4 chosen, faster training loop, Colab export

* **Platform decision:** Colab, GPU **G4** (RTX PRO 6000 Blackwell). `hardware.required_gpu = "RTX PRO 6000"` (provisional; confirm the device string in session 1). The timed pilot stages (`feasibility`, `tref`, `noise`, `sweep`) and `run_search.py` now refuse any other GPU.
* **Training loop:** removed the per-step host-device syncs (`float(loss)`, a per-step `isfinite`, and `GradScaler.step`'s `.item()`). The loss is now accumulated on the GPU and checked once per epoch, using fused AdamW. No scientific change; divergence is detected at epoch end (D34). Local median went from **5.0 → 4.0 s/epoch** (local T_ref estimate ~85 → **~68 s**). Small batch-32 models remain launch-overhead-bound (~10 ms/step under Windows WDDM).
* cuDNN autotuning (`cudnn.benchmark`) showed no reliable gain, so it stays off. The same configuration measured 4.1 then 3.2 s/epoch on consecutive passes (cold GPU), so an **untimed GPU warm-up epoch** now runs before pilot timing and before the first search job of each worker.
* New: `run_pilot.py speed` (P1a), a 5-minute compute estimate on any GPU.
* **Colab workflow corrected** (the researcher pointed to the LMT-Net project's proven setup). The earlier "upload code + data to Drive" plan and `colab_runner.ipynb` were removed. Now:
  * code is embedded in the notebook as a hash-checked zip (`build_colab_notebook.py`)
  * data are downloaded and rebuilt on the runtime and asserted byte-identical to the local arrays
  * results go to `My Drive/bamo_runs/`
  * Drive mounts from VS Code via the Colab command if `drive.mount` fails
* Verified locally from the extracted bundle: 54/54 tests pass, and the rebuilt `signals_100hz.npy` and `labels.npy` hashes are identical.
* New `scripts/concurrency_probe.py`. The LMT-Net project ran 25 trainings concurrently on one G4 at ~96% utilization; our small CNNs leave a G4 mostly idle when run one at a time, so session 1 measures the throughput gain at K = 1/4/8/16/24 processes. Local RTX 3050, K = 2: reference model ×0.8 (already saturated), small batch-32 model ×1.7.
* Notebook: library versions pinned to the lock file; first-session order documented.
* **Estimate on G4 (unmeasured):** T_ref ~17–27 s; full design ~240–385 GPU-h (~2,100–3,350 CU); reduced ~180–290 GPU-h. See doc 10 §10.3.

## 2026-10-08 — Environment, data, P0 smoke test

**Environment**
* Conda environment at `D:\envs\bamo` (C: had only 21 GB free): Python 3.12, torch 2.14.1+cu126 (CUDA available on the RTX 3050), optuna 5.0.0, DEHB 0.1.1, ConfigSpace 1.2.2, numpy 2.5.3, pandas 3.0.6, wfdb 4.3.1. Versions pinned in `requirements-lock.txt`.
* **Problem found and fixed:** the user-level Python 3.12 site-packages (`%APPDATA%\Python\Python312`, which holds tensorflow, sympy, networkx and others) leaked into the environment, so pip skipped some of PyTorch's dependencies. Fixes:
  * set `PYTHONNOUSERSITE=1` as a conda environment variable
  * installed the missing dependencies into the environment
  * `pip check` is now clean

  When calling `D:\envs\bamo\python.exe` without activating the environment, pass `-s`.
* Tests: 54 pass (core, controller, synthetic end-to-end for all methods including TPE and DEHB, and GPU training stack).
* **DEHB adapter:** confirmed the ask/tell API with 0.1.1; fidelities used are exactly {5, 10, 20}. Set `save_freq="end"` and disabled DEHB's loguru output: by default it writes state to disk on every incumbent change, which would be charged to its budget (and would hit Drive on Colab).
* **Bug fixed:** `reference_config` was at the top level of `configs/search_space.yaml` but read as `search_space.reference_config`. It is now nested, and the GPU tests caught it.

**Data (WP2)**
* Download source: the PhysioNet open-data mirror on AWS S3. The physionet.org zip ran at ~20 KB/s (about a day for 1.84 GB); S3 ran at ~0.3 MB/s. Only the metadata and 100 Hz records were downloaded: 43,598 record files, 511 MB, **all verified against PhysioNet's SHA256SUMS.txt**, 0 failures (`data/raw/ptbxl_download.json`).
* `prepare_data.py`:
  * 21,799 → **21,388 records** (411 without a diagnostic superclass; 0 signal issues)
  * **18,617 patients**
  * **0 patients in more than one fold**
  * **0 duplicate waveforms** (D37 resolved)
* Label cardinality: 1 label 16,244; 2 labels 4,068; 3 labels 919; 4 labels 157.
* Prevalence: NORM 44.5%, MI 25.6%, STTC 24.5%, CD 22.9%, HYP 12.4%. Class counts are even across folds.
* `validated_by_human`: 100% in folds 9 and 10, 65–69% in folds 1–8. This matches the dataset documentation.

| Partition | Train records / patients | Val records / patients | Test records / patients | Patient overlap |
|---|---|---|---|---|
| pilot (1–6 / 7) | 12,821 / 11,123 | 2,134 / 1,846 | — | 0 |
| search (1–7 / 8) | 14,955 / 12,969 | 2,129 / 1,854 | — | 0 |
| final (1–8 / 9 / 10) | 17,084 / 14,823 | 2,146 / 1,917 | 2,158 / 1,877 | 0 |

* Processed signals: (21,388, 12, 1,000) float32, 0.96 GB. The pilot training role takes 0.67 GB on the GPU.

**P0 smoke test (local RTX 3050, pilot partition)**
* Reference configuration, single-fidelity evaluator: status ok, **best validation macro AUROC 0.9354** at epoch 13, early-stopped after 19 epochs, **49.9 s total (2.63 s/epoch)**.
* Domain corners (2 epochs each, informative only, `runs/local_dev/pilot/p1_feasibility.json`):

| Corner | s/epoch | Peak GPU memory |
|---|---|---|
| width 1.5, 3 blocks, k15, dil 2, stride 1, batch 128 | 13.4 | 1.31 GB |
| same, batch 32 | 15.2 | 1.31 GB |
| width 0.5, 1 block, k5, dil 1, stride 2, batch 128 | 1.1 | 0.78 GB |

* **Implications:**
  * no memory risk anywhere in the domain, even on 8 GB
  * evaluation cost varies about 13× across the domain, which supports cost-normalized rewards
  * on the RTX 3050 a 20-epoch evaluation of the largest model would take up to ~300 s, so the local GPU is too slow for the benchmark: confirmatory runs go to Colab as planned

**Next**
1. Colab setup: upload `data/processed/` and code to Drive, run the notebook with `TASK="tests"`, check `drive.mount` from VS Code.
2. P1 on candidate Colab GPUs, then choose `hardware.required_gpu`.
3. Decisions still open in doc 08: D03, D05, D07, D21, D26, D30, D31, D38.
