# Budget-Adaptive Metaheuristic Optimization for ECG Classification

A study of ECG hyperparameter-tuning quality versus compute on patient-separated PTB-XL. The current draft compares random search, DE, official DEHB and matched block-DE controllers on five diagnostic superclasses and seven common rhythm labels, using one 1-D ResNet family.

The controller allocates search time across training, regularization, capacity and joint hyperparameter blocks by **observed validation improvement per unit of measured compute**, and stops after confirmation. Development experiments support tuning gains; its advantage over simpler policies remains under assessment.

* Proposal: `Proposal_Budget_Adaptive_ECG_Optimization.docx`
* Protocol and study plan: [docs/](docs/README.md). Start at [docs/00_overview.md](docs/00_overview.md).
* Open decisions: [docs/08_decision_register.md](docs/08_decision_register.md)

## Status (16 Oct 2026)

**Results: [docs/27_results_v2.md](docs/27_results_v2.md)** (fold-10 test under the frozen protocol v2 and locked analysis).


**Current plan: [docs/25_study_plan_v2.md](docs/25_study_plan_v2.md), one page (route A: is tuning worth its compute?).** **Protocol v2 ([docs/26](docs/26_protocol_v2.md)) frozen 14 Oct 2026; next: main runs with `notebooks/COLAB_G4_V2_MAIN.ipynb`.** Everything below and docs 00–24 are the history that led to it.

**Earlier results: [verified G4 results](docs/22_g4_decision_results.md); [assessor statement and next experiment](docs/23_assessor_g4_next_experiment.md); [v1.2 design draft](docs/19_protocol_v1_2_draft.md).**
The full study is held pending equal-cost assessment, matched controls and the final analysis plan.

| Phase | State |
|---|---|
| Code: search methods, matched controls, training, analysis | ✅ 76 tests pass |
| Data: PTB-XL 1.0.3 (100 Hz) downloaded, verified, patient-separated | ✅ 21,388 records, 18,617 patients, 0 overlap |
| Colab G4 environment (VS Code, embedded-code notebooks) | ✅ data byte-identical to local |
| Historical G4 pilot | ✅ T_ref 9.8 s; tolerance 0.0035; tau 0.0005 is a disclosed post-pilot amendment |
| Protocol freeze (G2) | ⏳ matched follow-up + task/inventory/analysis definitions |
| Original 300-run benchmark | Held; replaced by a focused draft pending decision evidence |
| Fresh-seed task headroom | ✅ Rhythm GO (+0.0217 AUROC); form BORDERLINE; 115 completed training evaluations |
| G4 task calibration, paired precision and matched real-data smoke | ✅ verified 14 Oct, [doc 22](docs/22_g4_decision_results.md): drift 1.7%; P +0.019 vs DEHB +0.027 over default at 57% of DEHB's search time; NI not established; equal-cost assessment and attribution follow-up pending |
| Final models, locked test evaluation, analysis, manuscript | ⏳ |

## Setup

Run the authorized local decision pilot:

```powershell
D:\envs\bamo\python.exe -s scripts/decision_pilot.py --tasks form rhythm
```

**Next G4 experiment:** [COLAB_G4_MATCHED_FOLLOWUP.ipynb](notebooks/COLAB_G4_MATCHED_FOLLOWUP.ipynb).
Run All in a fresh G4 session. It verifies the original `My Drive/bamo_decision` evidence,
runs 15 matched searches, assesses DEHB at P's stopping cost and fixed 40% prefixes, and saves
to a separate Drive folder. Search caps total 5.81 hours; up to 30 refits and setup are extra.
No new timing/noise sweep is required. The completed decision notebook is historical evidence;
do not rerun it over its original output folder with a new code bundle.
Real non-pilot searches require a matching frozen protocol; the decision-status file blocks
freezing until the assessment is complete. Historical pilot artifacts are not overwritten.

**Local (Windows, RTX 3050), for development, data preparation and tests.** The environment lives on D: (`D:\envs\bamo`) because C: is nearly full. Install with `--no-cache-dir` so pip does not cache wheels on C:.

```bash
conda create -p D:/envs/bamo python=3.12 -y
```
```bash
conda activate D:/envs/bamo
```
```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install --no-cache-dir -r requirements.txt
```
```bash
pip install -e . --no-deps
```
```bash
python -m pytest
```

**Colab G4 (pilot, benchmark, final training):** open `notebooks/COLAB_G4_SETUP.ipynb` in VS Code, choose *Select Kernel → Colab → New Colab Server → G4*, then *Run All*. The code is embedded in the notebook and the data are rebuilt on the runtime, so nothing is uploaded. See [docs/10_compute_and_colab_workflow.md](docs/10_compute_and_colab_workflow.md).

## Pipeline (study plan v2, docs/26)

| Step | Command | Where |
|---|---|---|
| Prepare both task datasets | `python scripts/prepare_data.py` and `python scripts/prepare_data.py --task rhythm` | local |
| Freeze protocol (after approval) | `python scripts/lock_protocol.py freeze`, then commit and tag `protocol-v2-frozen` | local |
| All searches, then all final models | `notebooks/COLAB_G4_V2_MAIN.ipynb` (runs `scripts/v2_driver.py`; resumable) | Colab |
| Validation summary | `python scripts/analyze_v2.py search --out-root <results>` | either |
| Lock analysis | `python scripts/lock_protocol.py analysis`, then build `v2_test` | local |
| One-time test on fold 10 | `notebooks/COLAB_G4_V2_TEST.ipynb` | Colab |
| Results | `python scripts/analyze_v2.py primary --out-root <results>` | local |

## Pipeline (v1, historical)

| Step | Command | Where |
|---|---|---|
| Download PTB-XL 1.0.3 | `python scripts/download_ptbxl.py` | local |
| Prepare data and audit splits | `python scripts/prepare_data.py` | local |
| Synthetic dry run | `python scripts/run_search.py --experiment main --synthetic` | local |
| Pilot stages P1–P4 | `python scripts/run_pilot.py {feasibility,tref,noise,sweep}` | Colab |
| Freeze protocol (G2) | `python scripts/lock_protocol.py freeze` | local |
| Matched development follow-up | `python scripts/matched_followup.py --source-root <original-export> --out-root <new-root> --drift <session-json>` | Colab |
| Confirmatory benchmark | 180-run draft; task-aware execution and frozen analysis plan pending | gated |
| Final models | `python scripts/train_final.py --experiment main` (+ `--reference`, `--shadow`) | Colab |
| Lock analysis (G4) | `python scripts/lock_protocol.py analysis` | local |
| Test evaluation (once) | `python scripts/evaluate_test.py` | Colab |
| Analyses | `python scripts/analyze.py {search,primary}` | either |

## Layout

```
configs/     base.yaml, search_space.yaml, controller.yaml, methods.yaml, experiments/
docs/        study plan, protocols, decision evidence and progress log (00-23)
notebooks/   COLAB_G4_*.ipynb (generated by scripts/build_colab_notebook.py)
protocol/    frozen protocol and analysis lock (written by scripts)
scripts/     command-line entry points
src/bamo/    search/ (space, budget, DE, controller, comparators, runner), data/, models/,
             training/, analysis/, metrics.py, hardware.py
tests/       unit and synthetic end-to-end tests
```
