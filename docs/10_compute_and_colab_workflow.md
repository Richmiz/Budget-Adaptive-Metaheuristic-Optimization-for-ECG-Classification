# 10 — Compute plan and Colab workflow

## 10.1 Hardware roles

| Machine | Used for | Not used for |
|---|---|---|
| **Local PC** (Windows 11, RTX 3050 8 GB) | Development, unit tests, synthetic dry runs, data download and preparation, P0 smoke tests, debugging, analysis and figures | Any timed run that feeds T_ref, budgets or confirmatory results |
| **Google Colab GPU** (opened from VS Code with the Colab extension, or in the browser) | P1–P6 pilot, all 270 search runs, final training, test evaluation, deployment profiling | Mixing GPU types inside the confirmatory benchmark |

**Why one GPU type matters.** Budgets are wall-clock caps measured in T_ref units, and the controller's reward divides by measured time. If a run lands on a faster or slower GPU, its budget means something different and its rewards are rescaled. `hardware.required_gpu` is frozen at G2, and `run_search.py` refuses any other GPU (`bamo/hardware.py`).

## 10.2 Compute projection

| Item | T_ref units |
|---|---|
| Main search: 7 methods × 10 seeds × (80 + 160 + 320) | 39,200 |
| Ablation search: 6 × 10 × 160 | 9,600 |
| Pilot (P2 + P3 + P4) | ~1,650 |
| Final training (~290 models × ~2.5 T_ref) | ~730 |
| **Total** | **~51,200** |

| T_ref on the frozen GPU | All-in GPU-hours | Sessions of 10 h needed |
|---|---|---|
| 20 s | ~285 | ~29 |
| 30 s | ~430 | ~43 |
| 45 s | ~640 | ~64 |
| 60 s | ~855 | ~86 |

* P stops early in some runs, which saves a little, but P is only 1/7 of the main matrix. Plan for the full amount.
* **Cost:** multiply GPU-hours by the current Colab compute-unit rate for the chosen GPU (check the rate in your Colab account at P1, because it changes) and compare with your plan's monthly units. Record the result in G2.
* **Calendar:** W9–W17 is about 9 weeks. At one 10-hour session per day, that is ~630 GPU-hours with one session, or ~1,260 with two parallel sessions on separate shards.

## 10.3 Chosen GPU: Colab G4 (decided 8 Oct 2026)

G4 = NVIDIA RTX PRO 6000 Blackwell (96 GB). Third-party measurements put it at roughly **8.7 compute units per hour** on Colab (versus about 1.7 for L4 and 5.4 for A100 40 GB). Google publishes no official rate, so check the live rate in the session's resources panel.

**Measured in the pilot (13 Oct 2026): T_ref = 9.8 s** (median of 16 full evaluations). Remaining compute ≤ 54,330 T_ref ≈ **≤ 148 G4-hours ≈ 1,320 CU** (doc 01 §1.4). The session-1 estimate below (2-epoch timing × 17 epochs) is kept for the record.

**Session-1 estimate (8 Oct 2026):** median **0.80 s/epoch** → **T_ref ≈ 14 s**.

| | Full design | Reduced design |
|---|---|---|
| GPU-hours, one job at a time | ~193 h | ~145 h |
| Compute units (8.9 CU/h) | ~1,720 CU | ~1,290 CU |
| Approx. cost (~$0.10/CU) | ~$170 | ~$130 |
| Calendar, one session ~20 h/day | ~10 days | ~7 days |
| With 4 concurrent jobs (measured gain ×1.5–2.3, needs D41 profiled cost) | ~85–130 h, ~$75–115 | ~65–95 h, ~$55–85 |

Earlier estimate, kept for the record:

| Basis | Median s/epoch | T_ref (17 epochs) |
|---|---|---|
| RTX 3050, Windows, measured (after removing per-step syncs) | 4.0 | ~68 s |
| G4, assumed 2.5–4× faster (large models gain most; small models are launch-overhead-bound) | ~1.0–1.6 | **~17–27 s** |

| | Full design (51,200 T_ref) | Reduced design (38,400 T_ref) |
|---|---|---|
| GPU-hours on G4 | ~240–385 h | ~180–290 h |
| Compute units (8.7 CU/h) | ~2,100–3,350 CU | ~1,600–2,500 CU |
| Approx. cost (~$0.10/CU on Pro) | ~$210–335 | ~$160–250 |
| Calendar, one session ~20 h/day | ~12–19 days | ~9–14 days |
| Pilot P1–P5 (included above) | ~10–13 h ≈ 90–115 CU | same |

**Cost check worth 5 minutes:** G4 costs about 5× L4 per hour. Unless it is more than 5× faster on these small 1-D CNNs (unlikely when small models are overhead-bound), L4 is cheaper per evaluation, though slower in wall time. Running the `speed` stage once on an L4 runtime (~0.2 CU) settles it. The timing guard only applies to the timed stages, so `speed` runs on any GPU.

## 10.3a Original GPU-choice criteria (P1)

In order:
1. you can get it **reliably** on your plan
2. the projected compute units fit the budget
3. speed

A slower but reliably available GPU is better than a fast one that is often unavailable: work stalls when the frozen type is not available. Typical options are T4, L4 and A100; availability depends on the plan.

## 10.4 Mitigation ladder (if G2's compute check fails)

Apply in order and record each step in doc 08 as an amendment **before** the benchmark starts:
1. Speed up the evaluator without changing the science: GPU-resident data (already the case), AMP (D07), larger evaluation batch.
2. Run two or three parallel sessions on disjoint seed shards.
3. Reduce the seed count to the number the P5 power check supports (never below 7 for the primary contrast).
4. Run B_high for P, DEHB and standard DE only; report the other methods at B_low and B_mid.
5. Drop `mu_nocost` and `mu_noconfirm` (keep `rr_nostop` and `mu_nostop`, as the proposal says).

## 10.5 How code, data and results reach Colab

This follows the pattern already proven in the LMT-Net project (VS Code + Colab G4, Oct 2026). **Nothing is uploaded by hand.**

| What | How |
|---|---|
| **Code** | Embedded in each notebook as a base64 zip with its SHA-256 (`scripts/build_colab_notebook.py`). The notebook verifies the hash, extracts it to `/content/bamo` and installs the pinned libraries. The bundle hash is the code version and is written to `My Drive/bamo_runs/environment/`. Rebuild the notebook after every code change. |
| **Data** | Downloaded on the runtime from PhysioNet's S3 mirror (`download_ptbxl.py`, every file checked against PhysioNet's SHA256SUMS), then rebuilt with `prepare_data.py`. The notebook asserts the arrays are **byte-identical** to the local preparation (hashes embedded; verified locally 8 Oct 2026). The data live on the VM's local disk, never read from Drive during training. |
| **Results** | Written to `My Drive/bamo_runs/` while running, so a disconnect loses at most the job in progress. Final cells zip results to Drive for download to the workstation. |
| **Environment record** | `nvidia-smi`, `pip freeze` and the bundle hash in `My Drive/bamo_runs/environment/`. |

**Drive mount from VS Code:** the first cell calls `drive.mount`. If that fails under the VS Code kernel, run the Command Palette command **Colab: Mount Google Drive to Server...**, run the cell it adds, then **Run All** again.

```
My Drive/bamo_runs/
  environment/           nvidia_smi.txt, pip_freeze.txt, bundle_sha256.txt
  pilot/                 p1_speed.json, concurrency.json, p2_tref.json, …
  main/, ablation/       run directories (doc 09), run_matrix.csv, sessions.md
  *_results.zip          exports for the workstation
```

## 10.6 Notebooks

| Notebook | Purpose | Time / cost on G4 |
|---|---|---|
| `notebooks/COLAB_G4_SETUP.ipynb` | Session 1: verify code and data, tests on GPU, speed check (T_ref estimate), concurrency probe | ~20–30 min, ≈ 3–5 CU |
| (next) pilot P2–P4, main benchmark, ablation, final training | Generated after session 1, once the concurrency decision is made | — |

**In VS Code:** open the notebook → **Select Kernel → Colab → New Colab Server → G4** → **Run All**.

## 10.7 Session rules

1. Run only the notebook's own work on the runtime: wall time is the budget (unless the design moves to profiled cost, see D41).
2. **If disconnected:** reconnect and Run All. Completed jobs are skipped, and an interrupted job restarts from scratch (D39).
3. Disconnect the runtime when the notebook finishes. G4 bills while idle.

## 10.8 Things that silently break budgets (avoid)

* Reading training data from the Drive mount during a run. The notebook builds the data on local disk.
* Running a different notebook on the same runtime.
* Changing `configs/` mid-benchmark. The runner checks the frozen hash.
* Comparing runs from different GPU types. The guard plus the `hardware` field in every `summary.json` let you audit this.
