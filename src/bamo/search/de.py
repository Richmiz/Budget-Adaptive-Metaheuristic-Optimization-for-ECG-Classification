"""DE/rand/1/bin on [0, 1]^d with variation optionally restricted to a block (protocol §3.2.1).

Coordinates outside the active block are inherited from the target individual. Fitness is
maximized (search-validation macro AUROC); invalid evaluations carry ``-inf``.
"""

from __future__ import annotations

import math

import numpy as np


def fitness_or_ninf(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else -math.inf


class DifferentialEvolution:
    def __init__(self, dim: int, pop_size: int = 8, F: float = 0.5, CR: float = 0.7,
                 rng: np.random.Generator | None = None):
        if pop_size < 4:
            raise ValueError("DE/rand/1 needs at least four individuals")
        self.dim, self.pop_size, self.F, self.CR = dim, pop_size, F, CR
        self.rng = rng or np.random.default_rng()
        self.X: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self._next_target = 0

    def initialize(self, X: np.ndarray, fitness) -> None:
        X = np.asarray(X, dtype=float)
        if X.shape != (self.pop_size, self.dim):
            raise ValueError(f"population must have shape {(self.pop_size, self.dim)}, got {X.shape}")
        self.X = X.copy()
        self.fitness = np.array([fitness_or_ninf(f) for f in fitness], dtype=float)

    def next_target(self) -> int:
        """Targets are visited cyclically through the population."""
        i = self._next_target
        self._next_target = (i + 1) % self.pop_size
        return i

    def make_trial(self, target: int, active: np.ndarray | None = None) -> np.ndarray:
        active = np.arange(self.dim) if active is None else np.asarray(active, dtype=int)
        others = [i for i in range(self.pop_size) if i != target]
        r1, r2, r3 = self.rng.choice(others, size=3, replace=False)
        x = self.X[target]
        mutant = self.X[r1, active] + self.F * (self.X[r2, active] - self.X[r3, active])
        # Bounce-back: a violated bound is replaced by the midpoint of target value and bound.
        low, high = mutant < 0.0, mutant > 1.0
        mutant[low] = x[active][low] / 2.0
        mutant[high] = (x[active][high] + 1.0) / 2.0
        cross = self.rng.random(active.size) < self.CR
        cross[self.rng.integers(active.size)] = True
        trial = x.copy()
        trial[active[cross]] = mutant[cross]
        return trial

    def select(self, target: int, trial: np.ndarray, fitness: float | None) -> bool:
        """Greedy one-to-one replacement; ties favour the trial (standard DE)."""
        f = fitness_or_ninf(fitness)
        if f >= self.fitness[target]:
            self.X[target] = trial
            self.fitness[target] = f
            return True
        return False
