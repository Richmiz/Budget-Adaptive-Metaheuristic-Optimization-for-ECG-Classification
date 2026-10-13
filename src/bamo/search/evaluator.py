"""Candidate evaluation interface, the shared initial-design cache, and a synthetic objective.

A real evaluator (``bamo.training.evaluator.TorchEvaluator``) trains the 1-D CNN and returns
search-validation macro AUROC. The ``SyntheticEvaluator`` mimics its structure on a simulated
clock so that every search method can be verified cheaply before GPU time is spent
(protocol §3.2.2: "Algorithm and evaluator implementations will first be checked against
inexpensive objectives").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..utils import derive_seed, read_json, stable_hash, write_json
from .budget import BudgetClock, BudgetExhausted, SimulatedTime

VALID_STATUSES = ("ok", "diverged")


@dataclass
class EvalResult:
    config: dict
    fidelity: int
    seed: int
    fitness: float = float("nan")
    cost_s: float = 0.0
    status: str = "ok"  # ok | diverged | infeasible | failed
    epochs_run: int = 0
    best_epoch: int = -1
    info: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status in VALID_STATUSES and math.isfinite(self.fitness)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(**d)


class Evaluator(Protocol):
    def evaluate(self, config: dict, fidelity: int, seed: int, clock: BudgetClock) -> EvalResult:
        """Train/evaluate one configuration. Must call ``clock.check()`` often enough to be
        interruptible at the cap, letting ``BudgetExhausted`` propagate."""


class SharedInitCache:
    """Results of the shared initial design for one search seed and partition.

    The first method to need an initial configuration evaluates it; later methods reuse the
    result and are *charged* its recorded cost (protocol §3.5.2, decision D09).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = read_json(self.path) if self.path.exists() else {}

    def get(self, key: str) -> EvalResult | None:
        d = self._data.get(key)
        return EvalResult.from_dict(d) if d is not None else None

    def put(self, key: str, result: EvalResult) -> None:
        self._data[key] = result.to_dict()
        write_json(self.path, self._data)


class SyntheticEvaluator:
    """Cheap stand-in for CNN training with block structure and cross-block interactions.

    * Block A: optimal learning rate shifts with batch size (within-block interaction).
    * Block B: weak weight-decay and dropout effects.
    * Block C: capacity helps up to a point and costs time.
    * Cross-block: large dilated kernels only help with low dropout (A–C/B–C interactions),
      so block-restricted search can stall where joint search (block J) does not.
    * Fitness rises with fidelity; additive noise is deterministic per (config, seed, fidelity).
    """

    def __init__(self, time_source: SimulatedTime, base_cost_s: float = 60.0, noise_sd: float = 0.002,
                 max_fidelity: int = 20, infeasible_prob: float = 0.0):
        self.time = time_source
        self.base_cost_s = base_cost_s
        self.noise_sd = noise_sd
        self.max_fidelity = max_fidelity
        self.infeasible_prob = infeasible_prob

    @staticmethod
    def quality(c: dict) -> float:
        lr_opt = -3.2 + 0.3 * math.log2(c["batch_size"] / 64)
        q = 0.915
        q -= 0.012 * (math.log10(c["learning_rate"]) - lr_opt) ** 2
        q -= 0.001 * (math.log10(c["weight_decay"]) + 4.0) ** 2
        q -= 0.03 * (c["dropout"] - 0.15) ** 2
        q += {0.5: -0.006, 1.0: 0.0, 1.5: 0.002}[c["width_multiplier"]]
        q += {1: -0.004, 2: 0.0, 3: 0.001}[c["blocks_per_group"]]
        q -= 0.002 * abs(c["kernel_size"] - 9) / 6
        if c["dilation"] == 2 and c["kernel_size"] == 15:
            q += 0.006 if c["dropout"] < 0.2 else -0.006
        if c["stem_stride"] == 1:
            q += 0.001
        return q

    def cost(self, c: dict, fidelity: int, rng: np.random.Generator) -> float:
        capacity = c["width_multiplier"] ** 2 * c["blocks_per_group"] / 2.0
        cost = self.base_cost_s * (fidelity / self.max_fidelity)
        cost *= (0.6 + 0.4 * capacity) * (1.6 if c["stem_stride"] == 1 else 1.0)
        cost *= (64.0 / c["batch_size"]) ** 0.3
        return cost * rng.uniform(0.9, 1.1)

    def evaluate(self, config: dict, fidelity: int, seed: int, clock: BudgetClock) -> EvalResult:
        rng = np.random.default_rng(derive_seed("synthetic", stable_hash(config), seed, fidelity))
        cost = self.cost(config, fidelity, rng)
        if clock.remaining() < cost:
            self.time.advance(max(clock.remaining(), 0.0))
            raise BudgetExhausted("evaluation interrupted at the cap")
        self.time.advance(cost)
        if rng.random() < self.infeasible_prob:
            return EvalResult(config, fidelity, seed, cost_s=cost, status="infeasible")
        progress = 1.0 - 0.04 * (1.0 - fidelity / self.max_fidelity) ** 1.5
        fit = self.quality(config) * progress + rng.normal(0.0, self.noise_sd)
        return EvalResult(config, fidelity, seed, fitness=float(min(fit, 1.0)), cost_s=cost,
                          epochs_run=fidelity, best_epoch=fidelity - 1)
