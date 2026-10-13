"""Common interface for all search methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..budget import BudgetClock
from ..evaluator import EvalResult
from ..space import SearchSpace
from ..tracker import SearchTracker


@dataclass
class SearchContext:
    space: SearchSpace
    tracker: SearchTracker
    clock: BudgetClock
    rng: np.random.Generator
    initial_X: np.ndarray            # shared initial design for this search seed
    t_ref: float                     # reference evaluation duration (s), frozen after the pilot
    fidelities: tuple[int, ...]      # epochs; the last entry is the maximum fidelity
    train_seed: int
    reeval_seeds: tuple[int, ...] = ()
    run_dir: Path | None = None
    log_decision: Callable[[dict], None] = field(default=lambda d: None)

    @property
    def max_fidelity(self) -> int:
        return self.fidelities[-1]


class SearchMethod:
    """A method searches until it stops on its own (``run`` returns a termination reason)
    or the tracker raises ``BudgetExhausted``, which the runner catches."""

    name = "base"
    multi_fidelity = False

    def run(self, ctx: SearchContext) -> str:
        raise NotImplementedError

    def budget_reason(self) -> str:
        return "budget"

    def summary_extra(self) -> dict:
        return {}

    @staticmethod
    def evaluate_initial(ctx: SearchContext) -> list[EvalResult]:
        """Evaluate the shared initial design at maximum fidelity (charged to every method)."""
        return [ctx.tracker.evaluate(x=x, fidelity=ctx.max_fidelity, tag="init") for x in ctx.initial_X]
