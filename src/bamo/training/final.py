"""Final-model training (protocol §3.5.1) and the locked test evaluation.

The selected configuration is retrained from initialization on folds 1-8 (<= 50 epochs,
patience 8 on fold 9). Per-label F1 thresholds are chosen on fold 9. Fold 10 is only read
by ``evaluate_on_test``, which refuses to run until the analysis lock exists.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..metrics import classification_report, select_f1_thresholds
from ..models import build_model
from ..protocol import require_analysis_lock
from ..utils import read_json, write_json
from .trainer import predict, train_model


def train_final(config: dict, cfg: dict, tensors: dict, normalizer: dict, seed: int, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outcome = train_model(config, cfg["model"], tensors["train"], tensors["val"],
                          max_epochs=cfg["final_training"]["max_epochs"], patience=cfg["final_training"]["patience"],
                          seed=seed, amp=cfg["training"]["amp"], drop_last=cfg["training"]["drop_last"],
                          eval_batch_size=cfg["training"]["eval_batch_size"], keep_best_state=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    if outcome.best_state is None:
        raise RuntimeError(f"final training produced no valid checkpoint (status={outcome.status})")

    model = build_model(config, cfg["model"], n_classes=tensors["train"].Y.shape[1]).to(tensors["train"].X.device)
    model.load_state_dict(outcome.best_state)
    val_probs = predict(model, tensors["val"].X, cfg["training"]["eval_batch_size"], cfg["training"]["amp"])
    thresholds = select_f1_thresholds(tensors["val"].Y_np, val_probs)
    classes = cfg["data"]["classes"]
    torch.save({"state_dict": outcome.best_state, "config": config, "normalizer": normalizer, "seed": seed},
               out_dir / "model.pt")
    np.savez_compressed(out_dir / "val_predictions.npz", probs=val_probs, rows=tensors["val"].rows)
    record = {"config": config, "seed": seed, "train_seconds": train_s, "epochs_run": outcome.epochs_run,
              "best_epoch": outcome.best_epoch, "status": outcome.status, "history": outcome.history,
              "thresholds": thresholds.tolist(),
              "val_report": classification_report(tensors["val"].Y_np, val_probs, classes, thresholds),
              "n_parameters": sum(p.numel() for p in model.parameters())}
    write_json(out_dir / "final.json", record)
    return record


def evaluate_on_test(final_dir: Path, cfg: dict, tensors: dict) -> dict:
    require_analysis_lock()
    final_dir = Path(final_dir)
    checkpoint = torch.load(final_dir / "model.pt", map_location=tensors["test"].X.device)
    record = read_json(final_dir / "final.json")
    model = build_model(checkpoint["config"], cfg["model"], n_classes=tensors["test"].Y.shape[1])
    model = model.to(tensors["test"].X.device)
    model.load_state_dict(checkpoint["state_dict"])
    probs = predict(model, tensors["test"].X, cfg["training"]["eval_batch_size"], cfg["training"]["amp"])
    np.savez_compressed(final_dir / "test_predictions.npz", probs=probs, rows=tensors["test"].rows)
    report = classification_report(tensors["test"].Y_np, probs, cfg["data"]["classes"],
                                   np.asarray(record["thresholds"]))
    write_json(final_dir / "test_report.json", report)
    return report


@torch.no_grad()
def deployment_profile(config: dict, cfg: dict, device, n_repeats: int = 200, warmup: int = 20) -> dict:
    """Batch-one latency (median, p95), parameter count, size and peak memory (protocol §3.7)."""
    model = build_model(config, cfg["model"]).to(device).eval()
    x = torch.randn(1, cfg["model"].get("n_leads", 12), 10 * cfg["data"]["sampling_rate"], device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(warmup + n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    n_params = sum(p.numel() for p in model.parameters())
    return {"device": str(device), "latency_median_ms": 1e3 * float(np.median(times)),
            "latency_p95_ms": 1e3 * float(np.percentile(times, 95)), "n_parameters": n_params,
            "size_mb_fp32": n_params * 4 / 2**20,
            "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None}
