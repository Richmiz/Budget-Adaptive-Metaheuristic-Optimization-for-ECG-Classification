"""Loads the prepared arrays and builds normalized, device-resident tensors per partition.

The whole 100 Hz dataset is ~1 GB in float32, so each partition is kept on the GPU and
mini-batches are drawn by index. This avoids DataLoader worker overhead (slow on Windows,
variable on Colab) and keeps evaluation timing stable across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ptbxl import partition_rows


@dataclass
class PreparedData:
    X: np.ndarray            # (N, 12, 1000) float32
    Y: np.ndarray            # (N, 5) uint8
    folds: np.ndarray        # (N,) strat_fold
    patient_ids: np.ndarray  # (N,)
    ecg_ids: np.ndarray      # (N,)
    classes: tuple[str, ...]


def load_prepared(processed_dir: str | Path, mmap: bool = True) -> PreparedData:
    import pandas as pd

    processed_dir = Path(processed_dir)
    records = pd.read_csv(processed_dir / "records.csv")
    X = np.load(processed_dir / "signals_100hz.npy", mmap_mode="r" if mmap else None)
    Y = np.load(processed_dir / "labels.npy")
    classes = tuple(c for c in records.columns if c.startswith("y_"))
    return PreparedData(X=X, Y=Y, folds=records["strat_fold"].to_numpy(),
                        patient_ids=records["patient_id"].to_numpy(), ecg_ids=records["ecg_id"].to_numpy(),
                        classes=tuple(c[2:] for c in classes))


def lead_statistics(X: np.ndarray, rows: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean and SD over the training rows only (protocol §3.1.1, D04)."""
    total = np.zeros(X.shape[1])
    total_sq = np.zeros(X.shape[1])
    count = 0
    for start in range(0, len(rows), chunk):
        block = np.asarray(X[np.sort(rows[start:start + chunk])], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        total_sq += (block ** 2).sum(axis=(0, 2))
        count += block.shape[0] * block.shape[2]
    mean = total / count
    sd = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return mean.astype(np.float32), sd.astype(np.float32)


@dataclass
class TensorSplit:
    X: "object"              # torch.Tensor (n, 12, 1000) on device
    Y: "object"              # torch.Tensor (n, 5) float32 on device
    rows: np.ndarray         # row indices into PreparedData
    Y_np: np.ndarray


def build_tensors(data: PreparedData, partition_cfg: dict, device, dtype=None) -> tuple[dict[str, TensorSplit], dict]:
    """Normalize every role with training-role statistics and move it to ``device``."""
    import torch

    dtype = dtype or torch.float32
    rows = partition_rows(data.folds, partition_cfg)
    mean, sd = lead_statistics(data.X, rows["train"])
    out = {}
    for role, idx in rows.items():
        idx = np.sort(idx)
        x = (np.asarray(data.X[idx], dtype=np.float32) - mean[None, :, None]) / sd[None, :, None]
        out[role] = TensorSplit(X=torch.from_numpy(x).to(device=device, dtype=dtype),
                                Y=torch.from_numpy(data.Y[idx].astype(np.float32)).to(device),
                                rows=idx, Y_np=data.Y[idx])
    return out, {"lead_mean": mean.tolist(), "lead_sd": sd.tolist()}
