"""GPU candidate evaluator: trains the CNN for one configuration and returns its
search-validation macro AUROC (best checkpoint) and measured cost."""

from __future__ import annotations

import gc
import time

import torch

from ..search.budget import BudgetClock, BudgetExhausted
from ..search.evaluator import EvalResult
from .trainer import train_model


class TorchEvaluator:
    def __init__(self, tensors: dict, model_cfg: dict, training_cfg: dict, patience: int):
        self.train = tensors["train"]
        self.val = tensors["val"]
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.patience = patience

    def _sync(self) -> None:
        if self.train.X.device.type == "cuda":
            torch.cuda.synchronize()

    def evaluate(self, config: dict, fidelity: int, seed: int, clock: BudgetClock) -> EvalResult:
        self._sync()
        t0 = time.perf_counter()
        status, outcome = "ok", None
        try:
            outcome = train_model(
                config, self.model_cfg, self.train, self.val, max_epochs=fidelity, patience=self.patience,
                seed=seed, amp=self.training_cfg.get("amp", True), drop_last=self.training_cfg.get("drop_last", True),
                eval_batch_size=self.training_cfg.get("eval_batch_size", 1024), check_deadline=clock.check)
            status = outcome.status
        except torch.cuda.OutOfMemoryError:
            status = "infeasible"                       # D35: cost is still charged, no reward
        except BudgetExhausted:
            raise
        except RuntimeError as err:                     # e.g. cuDNN failures; recorded, never silently retried
            status = "failed"
            outcome = None
            failure = repr(err)
        finally:
            self._sync()
            cost = time.perf_counter() - t0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        info = {"history": outcome.history} if outcome else {}
        if status == "failed":
            info["error"] = failure
        return EvalResult(config=config, fidelity=fidelity, seed=seed,
                          fitness=outcome.best_score if outcome else float("nan"), cost_s=cost, status=status,
                          epochs_run=outcome.epochs_run if outcome else 0,
                          best_epoch=outcome.best_epoch if outcome else -1, info=info)
