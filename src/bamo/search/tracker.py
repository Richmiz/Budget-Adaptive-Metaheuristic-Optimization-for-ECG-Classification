"""Single entry point through which every search method evaluates candidates.

The tracker enforces the cap, applies the caching rules (protocol §3.2.1), keeps the
incumbent (best valid result at maximum fidelity with the run's training seed) and records
the evaluation log and the anytime trajectory.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .budget import BudgetClock, BudgetExhausted
from .evaluator import EvalResult, Evaluator, SharedInitCache
from .space import SearchSpace


class SearchTracker:
    def __init__(self, space: SearchSpace, evaluator: Evaluator, clock: BudgetClock, train_seed: int,
                 max_fidelity: int, shared_cache: SharedInitCache | None = None,
                 on_record: Callable[[dict], None] | None = None):
        self.space = space
        self.evaluator = evaluator
        self.clock = clock
        self.train_seed = train_seed
        self.max_fidelity = max_fidelity
        self.shared_cache = shared_cache
        self.on_record = on_record
        self.records: list[dict] = []
        self.trajectory: list[tuple[float, float]] = []
        self.incumbent: dict | None = None
        self.evaluation_wall_s = 0.0
        self.interrupted_evaluation_wall_s = 0.0
        self.late_results: list[dict] = []
        self._memo: dict[str, EvalResult] = {}

    @property
    def best(self) -> float:
        return self.incumbent["fitness"] if self.incumbent else -math.inf

    def key(self, config: dict, seed: int, fidelity: int) -> str:
        return f"{self.space.config_key(config)}|s{seed}|f{fidelity}"

    def evaluate(self, x: np.ndarray | None = None, config: dict | None = None, fidelity: int | None = None,
                 seed: int | None = None, tag: str = "search") -> EvalResult:
        if config is None:
            config = self.space.decode(x)
        x = self.space.encode(config) if x is None else np.asarray(x, dtype=float)
        fidelity = self.max_fidelity if fidelity is None else int(fidelity)
        seed = self.train_seed if seed is None else int(seed)
        key = self.key(config, seed, fidelity)

        self.clock.check()
        t_start = self.clock.elapsed()
        shared = self.shared_cache if tag == "init" else None
        if key in self._memo:
            # Identical configuration, seed, partition and fidelity within this run.
            result, source = self._memo[key], "run_cache"
        elif shared is not None and (hit := shared.get(key)) is not None:
            self.clock.charge(hit.cost_s)
            result, source = hit, "shared_cache"
        else:
            try:
                result = self.evaluator.evaluate(config, fidelity, seed, self.clock)
            except BudgetExhausted:
                self.interrupted_evaluation_wall_s += max(self.clock.elapsed() - t_start, 0.0)
                raise
            self.evaluation_wall_s += result.cost_s
            source = "evaluated"
        t_end = self.clock.elapsed()
        if t_end > self.clock.limit_s:
            self.late_results.append({"config": dict(config), "seed": seed, "fidelity": fidelity,
                                      "completed_s": t_end, "source": source,
                                      "overshoot_s": t_end - self.clock.limit_s})
            raise BudgetExhausted("evaluation finished after the cap; result discarded")
        if shared is not None and source == "evaluated":
            shared.put(key, result)
        self._memo[key] = result

        improved = (result.valid and fidelity == self.max_fidelity and seed == self.train_seed
                    and result.fitness > self.best)
        if improved:
            self.incumbent = {"config": dict(config), "x": x.tolist(), "fitness": result.fitness,
                              "index": len(self.records), "t": t_end}
            self.trajectory.append((t_end, result.fitness))

        record = {
            "index": len(self.records), "tag": tag, "source": source, "t_start": t_start, "t_end": t_end,
            "config": config, "x": x.tolist(), "fidelity": fidelity, "seed": seed,
            "fitness": result.fitness, "status": result.status, "cost_s": result.cost_s,
            "epochs_run": result.epochs_run, "best_epoch": result.best_epoch,
            "new_incumbent": improved, "incumbent_fitness": self.best if self.incumbent else None,
        }
        self.records.append(record)
        if self.on_record:
            self.on_record(record)
        return result
