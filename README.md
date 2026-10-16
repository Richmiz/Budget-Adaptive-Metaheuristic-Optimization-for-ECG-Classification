# Budget-Adaptive Metaheuristic Optimization for ECG Classification

A study of ECG hyperparameter-tuning quality versus compute on patient-separated PTB-XL. The completed v2 experiment compares random search, official DEHB and P (a block-DE controller with confirmed stopping) on five diagnostic superclasses and seven common rhythm labels, using one 1-D ResNet family.

The controller allocates search time across training, regularization, capacity and joint hyperparameter blocks by **observed validation improvement per unit of measured compute**, and stops after confirmation. The final study supports modest diagnosis tuning gains. Rhythm gains beyond a development-informed default and the controller's comparative quality remain uncertain.

* Proposal: `Proposal_Budget_Adaptive_ECG_Optimization.docx`
* Current results: [docs/27_results_v2.md](docs/27_results_v2.md).
* Protocol and reporting: [docs/](docs/README.md), [frozen protocol v2](docs/26_protocol_v2.md).

## Status (16 Oct 2026)

**The main experiment and locked fold-10 analysis are complete. Current work: corrected reporting, publication figures and manuscript preparation.**

The adopted question is [study plan v2](docs/25_study_plan_v2.md): is tuning worth its compute?
Protocol v2 was frozen on 14 October; analysis was locked and the final test evaluated on
16 October. The older 300-run design, matched-follow-up notebook and docs 00–24 are historical
development work. They are outside the completed v2 study and are not the current execution queue.

| Phase | State |
|---|---|
| Code and analysis verification | ✅ 90 tests pass, including reporting-supplement equivalence checks |
| Patient-separated data | ✅ Diagnosis: 21,388 records / 18,617 patients; rhythm: 20,934 / 18,221; no fold overlap |
| Protocol v2 freeze | ✅ 14 Oct; tag `protocol-v2-frozen` |
| Main searches / final models | ✅ 120/120 searches; 150/150 final models |
| Analysis lock / fold-10 evaluation | ✅ 16 Oct; tag `analysis-v2-locked`; all 10 seeds complete |
| Independent audit | ✅ 1,081 archived files verified; all test AUROC point estimates reproduce; [assessment](docs/28_project_status_assessment.md) |
| Reporting supplement | Paired budget effects, absolute-score intervals for figures and separate compute categories; [addendum](docs/29_reporting_addendum_v2.md) |
| Publication package | Figures and manuscript in preparation; original checkpoint archive and complete billing reconciliation remain pending |

## Current reporting commands

```powershell
D:\envs\bamo\python.exe -s scripts/audit_v2_status.py
D:\envs\bamo\python.exe -s scripts/reporting_supplement_v2.py
D:\envs\bamo\python.exe -s scripts/prepare_publication_figures.py
```

These use saved predictions and logs. The supplement is labelled as specified after test
inspection; its results are separate from the locked primary analysis. Original runtime
notebooks are retained as execution evidence. A new experiment requires a separate scope
and output directory because fold 10 has already informed interpretation.

## Environment setup

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

## Completed execution pipeline (study plan v2, docs/26)

| Step | Command | Where |
|---|---|---|
| Prepare both task datasets | `python scripts/prepare_data.py` and `python scripts/prepare_data.py --task rhythm` | local |
| Protocol freeze — completed | `protocol-v2-frozen` | local |
| All searches, then all final models | `notebooks/COLAB_G4_V2_MAIN.ipynb` (runs `scripts/v2_driver.py`; resumable) | Colab |
| Validation summary | `python scripts/analyze_v2.py search --out-root <results>` | either |
| Analysis lock — completed | `analysis-v2-locked` | local |
| Fold-10 evaluation — completed | `notebooks/COLAB_G4_V2_TEST.ipynb` | Colab |
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
| Earlier benchmark proposal | Superseded 180-run draft | historical |
| Final models | `python scripts/train_final.py --experiment main` (+ `--reference`, `--shadow`) | Colab |
| Lock analysis (G4) | `python scripts/lock_protocol.py analysis` | local |
| Test evaluation (once) | `python scripts/evaluate_test.py` | Colab |
| Analyses | `python scripts/analyze.py {search,primary}` | either |

## Layout

```
configs/     base.yaml, search_space.yaml, controller.yaml, methods.yaml, experiments/
docs/        study plans, frozen protocol, results, reporting addendum and evidence
figures/     publication figures, captions, source tables and provenance manifest
notebooks/   COLAB_G4_*.ipynb (generated by scripts/build_colab_notebook.py)
protocol/    frozen protocol and analysis lock (written by scripts)
scripts/     command-line entry points
src/bamo/    search/ (space, budget, DE, controller, comparators, runner), data/, models/,
             training/, analysis/, metrics.py, hardware.py
tests/       unit and synthetic end-to-end tests
```
