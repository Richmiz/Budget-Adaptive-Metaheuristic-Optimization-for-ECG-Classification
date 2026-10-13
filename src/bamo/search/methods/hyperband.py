"""B4 Hyperband and B5 BOHB, time-capped, sharing one successive-halving scheduler.

Fidelities are epochs (default 5, 10, 20 with eta = 2). Each rung retrains from scratch
(no checkpoint continuation, D25). Only maximum-fidelity results can become the incumbent.

BOHB here is an in-house reimplementation of Falkner et al. (2018): Hyperband scheduling with
a TPE-style ratio of two Gaussian product KDEs fitted on the largest fidelity with enough
observations. Unlike hpbandster it models every coordinate as continuous in the encoded
[0, 1] space (ordinal choices included). It must be checked against the official
implementation on a cheap benchmark before confirmatory use (decision D26).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .base import SearchContext, SearchMethod


class RandomSampler:
    def __init__(self, dim: int):
        self.dim = dim

    def observe(self, x: np.ndarray, fidelity: int, fitness: float | None) -> None:
        pass

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.dim)


class KDESampler:
    """BOHB configuration sampler (defaults follow hpbandster's BOHB)."""

    def __init__(self, dim: int, min_points_in_model: int | None = None, top_n_percent: int = 15,
                 num_samples: int = 64, random_fraction: float = 1 / 3, bandwidth_factor: float = 3.0,
                 min_bandwidth: float = 1e-3):
        self.dim = dim
        self.min_points = min_points_in_model or dim + 1
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.obs: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        self.model_samples = 0

    def observe(self, x: np.ndarray, fidelity: int, fitness: float | None) -> None:
        f = fitness if fitness is not None and math.isfinite(fitness) else -math.inf
        self.obs[fidelity].append((np.asarray(x, dtype=float), f))

    def _bandwidth(self, data: np.ndarray) -> np.ndarray:
        n, d = data.shape
        sd = data.std(axis=0, ddof=1) if n > 1 else np.zeros(d)
        return np.maximum(1.06 * sd * n ** (-1.0 / (4 + d)), self.min_bandwidth)

    @staticmethod
    def _log_kde(x: np.ndarray, data: np.ndarray, bw: np.ndarray) -> float:
        z = (x[None, :] - data) / bw[None, :]
        log_k = -0.5 * np.sum(z ** 2, axis=1) - np.sum(np.log(bw)) - 0.5 * data.shape[1] * math.log(2 * math.pi)
        m = log_k.max()
        return float(m + math.log(np.exp(log_k - m).sum()) - math.log(len(data)))

    def _model_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        for fid in sorted(self.obs, reverse=True):
            items = self.obs[fid]
            n = len(items)
            n_good = max(self.min_points, (self.top_n_percent * n) // 100)
            n_bad = max(self.min_points, ((100 - self.top_n_percent) * n) // 100)
            if n < n_good + self.min_points:
                continue
            order = np.argsort([-f for _, f in items], kind="stable")
            X = np.stack([x for x, _ in items])
            good = X[order[:n_good]]
            bad = X[order[n_good:n_good + n_bad]]
            if len(bad) >= self.min_points:
                return good, bad
        return None

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.random_fraction:
            return rng.random(self.dim)
        data = self._model_data()
        if data is None:
            return rng.random(self.dim)
        good, bad = data
        bw_good, bw_bad = self._bandwidth(good), self._bandwidth(bad)
        sample_bw = np.maximum(bw_good * self.bandwidth_factor, self.min_bandwidth)
        best_x, best_score = None, -math.inf
        for _ in range(self.num_samples):
            centre = good[rng.integers(len(good))]
            cand = centre + sample_bw * rng.standard_normal(self.dim)
            for _ in range(20):  # truncated normal by resampling out-of-range coordinates
                out = (cand < 0) | (cand > 1)
                if not out.any():
                    break
                cand[out] = centre[out] + sample_bw[out] * rng.standard_normal(out.sum())
            cand = np.clip(cand, 0.0, 1.0)
            score = self._log_kde(cand, good, bw_good) - self._log_kde(cand, bad, bw_bad)
            if score > best_score:
                best_x, best_score = cand, score
        self.model_samples += 1
        return best_x


class Hyperband(SearchMethod):
    name = "hyperband"
    multi_fidelity = True

    def __init__(self, eta: int = 2):
        self.eta = eta
        self.brackets_started = 0

    def make_sampler(self, ctx: SearchContext):
        return RandomSampler(ctx.space.dim)

    def run(self, ctx: SearchContext) -> str:
        fids = tuple(ctx.fidelities)
        for lo, hi in zip(fids, fids[1:]):
            if hi != lo * self.eta:
                raise ValueError(f"fidelities {fids} are not a geometric ladder with eta={self.eta}")
        sampler = self.make_sampler(ctx)
        for x, r in zip(ctx.initial_X, self.evaluate_initial(ctx)):
            sampler.observe(x, ctx.max_fidelity, r.fitness if r.valid else None)
        s_max = len(fids) - 1
        while True:
            for s in range(s_max, -1, -1):
                self.brackets_started += 1
                n = math.ceil((s_max + 1) / (s + 1) * self.eta ** s)
                rungs = fids[s_max - s:]
                configs: list[np.ndarray] = []
                scores: list[float] = []
                for i, fid in enumerate(rungs):
                    if i == 0:
                        for _ in range(n):  # sample lazily so the model sees earlier results
                            x = sampler.sample(ctx.rng)
                            configs.append(x)
                            scores.append(self._eval(ctx, sampler, x, fid, s))
                    else:
                        scores = [self._eval(ctx, sampler, x, fid, s) for x in configs]
                    if i == len(rungs) - 1:
                        break
                    keep = max(1, len(configs) // self.eta)
                    order = np.argsort(-np.asarray(scores), kind="stable")[:keep]
                    configs = [configs[j] for j in order]

    def _eval(self, ctx: SearchContext, sampler, x: np.ndarray, fidelity: int, bracket: int) -> float:
        r = ctx.tracker.evaluate(x=x, fidelity=fidelity, tag=f"{self.name}_b{bracket}")
        fit = r.fitness if r.valid else None
        sampler.observe(x, fidelity, fit)
        return fit if fit is not None else -math.inf

    def summary_extra(self) -> dict:
        return {"brackets_started": self.brackets_started}


class BOHB(Hyperband):
    name = "bohb"

    def __init__(self, eta: int = 2, **sampler_kwargs):
        super().__init__(eta)
        self.sampler_kwargs = sampler_kwargs
        self._sampler: KDESampler | None = None

    def make_sampler(self, ctx: SearchContext) -> KDESampler:
        self._sampler = KDESampler(ctx.space.dim, **self.sampler_kwargs)
        return self._sampler

    def summary_extra(self) -> dict:
        extra = super().summary_extra()
        extra["model_based_samples"] = self._sampler.model_samples if self._sampler else 0
        return extra
