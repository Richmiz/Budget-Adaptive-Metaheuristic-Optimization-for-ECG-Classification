# 03 — Data and preprocessing

## 3.1 Source

PTB-XL v1.0.3, PhysioNet (Wagner et al., 2022; DOI 10.13026/kfzx-aw45), CC BY 4.0. According to the dataset's documentation it contains 21,799 ten-second 12-lead records from 18,869 patients, at 100 Hz and 500 Hz. Each record has a `strat_fold` (1–10) assigned at patient level and stratified by labels. The dataset documentation also says folds 9 and 10 contain only records validated by at least one cardiologist; confirm this against `validated_by_human` in `data_report.json`. **All counts below are recomputed by `scripts/prepare_data.py`. Do not quote the documentation's numbers as the analytic sample.**

## 3.2 Download and inputs

`scripts/download_ptbxl.py` fetches the metadata and the 100 Hz records only (~0.5 GB, the study's input) from PhysioNet's open-data mirror on AWS S3 (`physionet-open/ptb-xl/1.0.3/`). Every file is verified against PhysioNet's `SHA256SUMS.txt`, whose own hash is recorded in `data/raw/ptbxl_download.json`. The direct physionet.org archive (1.84 GB, `--zip`) holds the same files but downloaded at ~20 KB/s here. The 500 Hz records are not needed (`--with-500hz` if ever required).

Inputs are the 100 Hz files (`filename_lr`), giving 12 × 1,000 samples per record. They are read with `wfdb.rdsamp` and stored unnormalized as float32 (`signals_100hz.npy`).

## 3.3 Labels (D01)

Five diagnostic superclasses: NORM, MI, STTC, CD, HYP. This is multi-label.
1. Parse `scp_codes` (a dict of statement → likelihood).
2. Keep statements with `diagnostic == 1` in `scp_statements.csv`.
3. Map each to `diagnostic_class`. Likelihood values are **not** thresholded, which matches the official benchmark code (Strodthoff et al., 2021).
4. Exclude records with no superclass, and report the count.

## 3.4 Quality checks (run before any search; the rules are fixed now)

| Check | Rule | Action |
|---|---|---|
| Lead count | ≠ 12 | Exclude, record reason |
| Length | ≠ 1,000 samples | Exclude, record reason |
| Non-finite values | any NaN or Inf | Exclude, record reason |
| Duplicate waveforms | byte-identical arrays | **Report only** (D37: needs your call before G1) |
| Patient consistency | patient in more than one fold | Stop: must be zero |
| No eligible label | empty superclass set | Exclude (D01) |

`data_report.json` records:
* source and final record and patient counts
* exclusions by reason
* duplicate groups
* class counts per fold
* the label-cardinality distribution
* array SHA-256s

## 3.5 Partitions and audit

Partition definitions are in `configs/base.yaml`. `split_manifest.json` stores, for each partition and role, the record and patient counts, the pairwise patient overlap (which must be 0), and a hash of the sorted `ecg_id`s. Re-running preparation must reproduce the hashes.

Because `strat_fold` is assigned per patient, any union of folds is patient-separated. The audit checks this after filtering anyway, as the proposal requires.

## 3.6 Normalization (D04)

Per-lead z-score. The mean and SD are computed on the **training role of the partition in use**:

| Partition | Statistics from |
|---|---|
| pilot | folds 1–6 |
| search | folds 1–7 |
| final | folds 1–8 |

The statistics are stored with each final model. There is no augmentation, filtering or resampling in the primary experiments.

## 3.7 Class imbalance (D03)

The primary analysis uses unweighted BCE, so the comparison isolates the search method. If class weighting is adopted instead:
* `pos_weight` is estimated from the training role only
* it is applied identically to every method, the pilot, final training and the reference model
* it must be decided before G2

## 3.8 Data products and integrity

See `data/README.md`. Data never enter git. On Colab, the processed folder is copied from Drive to the VM's local disk at session start, and checksums are verified against `data_report.json` when the session begins.
