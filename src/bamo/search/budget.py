"""Search-time accounting (protocol §3.3.1, §3.5.2).

The clock measures wall time from ``start()`` plus any *charged* time. Charged time is
used when a shared initial-design evaluation is served from cache: the recorded cost of
that evaluation is billed to every method that uses it, without re-running it.
"""

from __future__ import annotations

import time


class BudgetExhausted(RuntimeError):
    """The search-time cap was reached. The in-flight evaluation must be discarded."""


class WallTime:
    def now(self) -> float:
        return time.perf_counter()


class SimulatedTime:
    """Manually advanced time source for synthetic objectives and tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot run backwards")
        self.t += seconds


class BudgetClock:
    def __init__(self, limit_s: float, time_source: WallTime | SimulatedTime | None = None):
        if limit_s <= 0:
            raise ValueError("budget must be positive")
        self.limit_s = float(limit_s)
        self.time = time_source or WallTime()
        self.charged_s = 0.0
        self._t0: float | None = None

    def start(self) -> "BudgetClock":
        self._t0 = self.time.now()
        return self

    def elapsed(self) -> float:
        if self._t0 is None:
            raise RuntimeError("clock not started")
        return self.time.now() - self._t0 + self.charged_s

    def remaining(self) -> float:
        return self.limit_s - self.elapsed()

    def exhausted(self) -> bool:
        return self.remaining() <= 0.0

    def check(self) -> None:
        if self.exhausted():
            raise BudgetExhausted(f"search-time cap of {self.limit_s:.1f}s reached")

    def charge(self, seconds: float) -> None:
        """Bill ``seconds`` without wall time passing. If the charge does not fit, the
        clock is filled to the cap and the evaluation counts as not completed."""
        remaining = self.remaining()
        if seconds > remaining:
            self.charged_s += max(remaining, 0.0)
            raise BudgetExhausted("charged evaluation does not fit in the remaining budget")
        self.charged_s += seconds
