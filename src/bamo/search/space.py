"""Nine-dimensional search space encoded in [0, 1]^9 (protocol §3.2).

Log-scaled variables are interpolated on the log scale; discrete variables use equal-width
bins over [0, 1] and encode to bin centres. Blocks A–C partition the coordinates; block J
covers all of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..utils import stable_hash


@dataclass(frozen=True)
class Param:
    name: str
    block: str
    kind: str  # "log" | "float" | "choice"
    low: float | None = None
    high: float | None = None
    choices: tuple = ()

    def __post_init__(self) -> None:
        if self.kind not in ("log", "float", "choice"):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == "choice":
            if len(self.choices) < 2:
                raise ValueError(f"{self.name}: a choice parameter needs at least two choices")
        else:
            if self.low is None or self.high is None or not self.low < self.high:
                raise ValueError(f"{self.name}: requires low < high")
            if self.kind == "log" and self.low <= 0:
                raise ValueError(f"{self.name}: log scale requires low > 0")

    def decode(self, u: float) -> Any:
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        if self.kind == "float":
            return float(self.low + u * (self.high - self.low))
        return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]

    def encode(self, value: Any) -> float:
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return (math.log(value) - lo) / (hi - lo)
        if self.kind == "float":
            return (float(value) - self.low) / (self.high - self.low)
        return (self.choices.index(value) + 0.5) / len(self.choices)

    def canonical(self, value: Any) -> Any:
        return float(f"{float(value):.10g}") if self.kind != "choice" else value


class SearchSpace:
    def __init__(self, params: Iterable[Param], joint_block: str | None = "J"):
        self.params = tuple(params)
        self.names = tuple(p.name for p in self.params)
        self.dim = len(self.params)
        blocks: dict[str, list[int]] = {}
        for i, p in enumerate(self.params):
            blocks.setdefault(p.block, []).append(i)
        if joint_block:
            if joint_block in blocks:
                raise ValueError(f"joint block name {joint_block!r} collides with a parameter block")
            blocks[joint_block] = list(range(self.dim))
        self.blocks = {k: np.asarray(v, dtype=int) for k, v in blocks.items()}
        self.joint_block = joint_block

    @classmethod
    def from_config(cls, cfg: dict) -> "SearchSpace":
        params = [
            Param(
                name=p["name"],
                block=p["block"],
                kind=p["kind"],
                low=p.get("low"),
                high=p.get("high"),
                choices=tuple(p.get("choices", ())),
            )
            for p in cfg["parameters"]
        ]
        return cls(params, joint_block=cfg.get("joint_block", "J"))

    def decode(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"expected vector of length {self.dim}, got shape {x.shape}")
        return {p.name: p.decode(u) for p, u in zip(self.params, x)}

    def encode(self, config: dict) -> np.ndarray:
        return np.array([p.encode(config[p.name]) for p in self.params], dtype=float)

    def block_indices(self, block: str) -> np.ndarray:
        return self.blocks[block]

    def config_key(self, config: dict) -> str:
        """Identity of a decoded configuration (used for evaluation caching)."""
        return stable_hash({p.name: p.canonical(config[p.name]) for p in self.params})


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random Latin hypercube sample in [0, 1]^dim (used for the shared initial design)."""
    u = (np.arange(n)[:, None] + rng.random((n, dim))) / n
    for j in range(dim):
        u[:, j] = u[rng.permutation(n), j]
    return u
