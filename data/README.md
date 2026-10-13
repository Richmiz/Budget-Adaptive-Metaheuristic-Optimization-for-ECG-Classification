# Data

Nothing in this folder is committed (see `.gitignore`).

| Path | Content | Produced by |
|---|---|---|
| `raw/ptb-xl-1.0.3/` | PTB-XL v1.0.3: metadata + 100 Hz records (`records100/`), from PhysioNet's open-data mirror on AWS S3, each file verified against PhysioNet's `SHA256SUMS.txt` | `scripts/download_ptbxl.py` |
| `raw/ptbxl_download.json` | Source, subset, file counts, SHA-256 of `SHA256SUMS.txt`, failures, date | `scripts/download_ptbxl.py` |
| `processed/signals_100hz.npy` | (N, 12, 1000) float32 waveforms, unnormalized | `scripts/prepare_data.py` |
| `processed/labels.npy` | (N, 5) multi-hot NORM/MI/STTC/CD/HYP | `scripts/prepare_data.py` |
| `processed/records.csv` | Row → ecg_id, patient_id, strat_fold, labels | `scripts/prepare_data.py` |
| `processed/data_report.json` | Exclusions, duplicates, class counts per fold, patient audit | `scripts/prepare_data.py` |
| `processed/split_manifest.json` | Partitions, counts, overlap audit, ID hashes | `scripts/prepare_data.py` |

PTB-XL is released under CC BY 4.0. Cite Wagner et al. (2020/2022) and PhysioNet (Goldberger et al., 2000) in any output.
On Colab the notebook downloads and prepares the data on the runtime and asserts the arrays are byte-identical to the local ones, so nothing is uploaded.
