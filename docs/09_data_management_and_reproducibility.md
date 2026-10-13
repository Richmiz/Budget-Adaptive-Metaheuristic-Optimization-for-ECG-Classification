# 09 — Data management and reproducibility

## 9.1 Repository layout

```
configs/                 base, search space, controller, methods, experiments/{main,ablation,pilot}
docs/                    this protocol (00–16)
notebooks/               COLAB_G4_*.ipynb (generated, code embedded)
protocol/                frozen_protocol.yaml, analysis_lock.json (written by scripts/lock_protocol.py)
scripts/                 download, prepare, pilot, search, final training, lock, test evaluation, analysis
src/bamo/                package: search/ (space, budget, DE, controller, comparators, runner),
                         data/, models/, training/, analysis/, metrics.py, hardware.py
tests/                   unit + synthetic end-to-end tests
data/                    raw/ and processed/ (git-ignored)
runs/                    local outputs (git-ignored); confirmatory outputs live on Drive
```

## 9.2 Run outputs (`runs/<experiment>/<method>/<budget>/seed_<s>/`)

**`evaluations.jsonl`**, one line per evaluation:

| Field | Meaning |
|---|---|
| index, tag | order; `init`, `random`, `de`, `warmup:A`, `search:C`, `confirm:J`, `reeval`, `hyperband_b2`, `tpe`, `dehb`, … |
| source | `evaluated`, `run_cache` or `shared_cache` |
| t_start, t_end | search clock (s) |
| config, x | decoded configuration and its [0, 1]⁹ encoding |
| fidelity, seed | epochs, training seed |
| fitness, status | validation macro AUROC; ok, diverged, infeasible or failed |
| cost_s, epochs_run, best_epoch | measured cost and training detail |
| new_incumbent, incumbent_fitness | incumbent bookkeeping |

**`decisions.jsonl`**: controller events (doc 04 §4.10).
**`trajectory.csv`**: `elapsed_s, incumbent_val_auroc` at each incumbent change.
**`summary.json`**:
* the job and the partition
* **config hash** and **hardware fingerprint** (GPU, driver, CUDA, cuDNN, torch, Python, host, Colab flag)
* budget, T_ref, fidelities
* termination reason, elapsed, unused, charged, evaluation wall time, overhead
* counts by status and source, the incumbent, anytime efficiency
* method-specific extras

**`final/`**: `model.pt` (state dict, configuration, normalizer, seed), `final.json` (training history, thresholds, validation report, parameter count), `val_predictions.npz`. After the lock, also `test_predictions.npz` and `test_report.json`.

`runs/_shared_init/<partition>/seed_<s>.json` holds the cached shared initial-design results (D09).
`runs/<experiment>/run_matrix.csv` is the randomized job order.

## 9.3 Provenance chain

| Artifact | Fingerprint |
|---|---|
| Raw data | archive SHA-256 (`data/raw/ptbxl_download.json`) |
| Processed data | array SHA-256s and partition `ecg_id` hashes (`data_report.json`, `split_manifest.json`) |
| Code | git commit; tag `protocol-frozen` at G2 and `analysis-locked` at G4 |
| Configuration | merged-config hash in every `summary.json`; must equal `frozen_protocol.yaml` |
| Analysis plan | SHA-256 of doc 07 in `analysis_lock.json` |
| Environment | `requirements-lock.txt`; per-run fingerprint |

**On Colab without git** (code copied from Drive), record the commit hash of the local repository you uploaded in `runs/<exp>/sessions.md`. Better: push to a private GitHub repository and set `CODE_SOURCE = "git"` in the notebook.

## 9.4 Seeds and determinism

Seed derivations are in doc 02 §2.6. GPU training is not bit-for-bit deterministic, because cuDNN kernels and fp16 autocast can differ between runs. The design treats training noise as part of the measured variability: the re-evaluation step and the 10 search seeds absorb it. `cudnn.benchmark` is off for stable kernel choice. Do not switch on `torch.use_deterministic_algorithms`: it changes the speed, and therefore the budget semantics.

## 9.5 Storage and backup

* **Confirmatory outputs** live in `My Drive/bamo_runs/`, written directly by the runner; exported zips are archived locally.
* **Weekly backup:** zip `runs/` (without `model.pt`) to a dated archive, and keep one off-Drive copy (local D: drive).
* **Model checkpoints** are small (≤ a few MB each) and kept for the release.

## 9.6 Release package (W25–W26)

* code at the `analysis-locked` tag plus the release commit
* configs, the frozen protocol, the analysis lock
* run matrices, all `summary.json`, `decisions.jsonl`, `evaluations.jsonl`, `trajectory.csv`
* final-model predictions on folds 9 and 10 (allowed under the dataset's CC BY 4.0 licence, with attribution)
* split manifests and label mapping
* analysis scripts and notebooks regenerating every table and figure

Raw PTB-XL is not redistributed; users download it from PhysioNet.

## 9.7 Ethics and licensing

PTB-XL is de-identified and openly licensed (CC BY 4.0). Record the institutional ethics position (review or exemption for secondary analysis of public de-identified data) before submission, and cite the dataset (Wagner et al.) and PhysioNet (Goldberger et al., 2000).
