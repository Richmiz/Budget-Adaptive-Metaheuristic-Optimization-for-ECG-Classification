"""Training loop shared by candidate evaluation and final-model training.

AdamW + BCE-with-logits; checkpoint selection by validation macro AUROC; early stopping
with a patience counter. ``check_deadline`` is called before every mini-batch so a search
evaluation can be interrupted at the time cap (protocol §3.5.2).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..metrics import macro_auroc
from ..models import build_model


@dataclass
class TrainOutcome:
    best_score: float = math.nan
    best_epoch: int = -1
    epochs_run: int = 0
    status: str = "ok"                   # ok | diverged
    history: list = field(default_factory=list)
    best_state: dict | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 1024, amp: bool = True) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        with torch.autocast(device_type=X.device.type, dtype=torch.float16, enabled=amp and X.device.type == "cuda"):
            logits = model(X[i:i + batch_size])
        outs.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(outs).numpy()


def train_model(config: dict, model_cfg: dict, train, val, *, max_epochs: int, patience: int, seed: int,
                amp: bool = True, drop_last: bool = True, eval_batch_size: int = 1024,
                pos_weight: torch.Tensor | None = None, keep_best_state: bool = False,
                check_deadline: Callable[[], None] | None = None) -> TrainOutcome:
    device = train.X.device
    seed_everything(seed)
    model = build_model(config, model_cfg, n_classes=train.Y.shape[1]).to(device)
    # Fused AdamW lets GradScaler skip inf/NaN steps on the device instead of calling .item() every step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]), fused=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(seed)
    n, batch_size = train.X.shape[0], int(config["batch_size"])
    stop_at = n - n % batch_size if drop_last else n

    outcome = TrainOutcome()
    since_best = 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator).to(device)
        # No host-device sync inside the batch loop: the loss is accumulated on the GPU and checked
        # once per epoch, so the CPU can queue kernels ahead (small models are launch-bound otherwise).
        loss_sum, batches = torch.zeros((), device=device), 0
        for start in range(0, stop_at, batch_size):
            if check_deadline is not None:
                check_deadline()
            idx = perm[start:start + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(train.X[idx])
            loss = loss_fn(logits.float(), train.Y[idx])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach()
            batches += 1
        epoch_loss = float(loss_sum) / max(batches, 1)
        if not math.isfinite(epoch_loss):  # D34: the best earlier checkpoint, if any, stays valid
            outcome.status = "diverged"
            break
        if check_deadline is not None:
            check_deadline()
        probs = predict(model, val.X, eval_batch_size, amp)
        score = macro_auroc(val.Y_np, probs) if np.isfinite(probs).all() else math.nan
        outcome.epochs_run = epoch + 1
        outcome.history.append({"epoch": epoch, "train_loss": epoch_loss, "val_macro_auroc": score})
        if math.isfinite(score) and (not math.isfinite(outcome.best_score) or score > outcome.best_score):
            outcome.best_score, outcome.best_epoch, since_best = score, epoch, 0
            if keep_best_state:
                outcome.best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break
    return outcome
