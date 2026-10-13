"""B1 random search, B3 standard (full-dimensional) DE, and the naive-stopping DE baseline."""

from __future__ import annotations

from ..de import DifferentialEvolution
from .base import SearchContext, SearchMethod


class RandomSearch(SearchMethod):
    name = "random"

    def run(self, ctx: SearchContext) -> str:
        self.evaluate_initial(ctx)
        while True:
            ctx.tracker.evaluate(x=ctx.rng.random(ctx.space.dim), tag="random")


class StandardDE(SearchMethod):
    """DE/rand/1/bin over all nine coordinates, same engine and settings as the controller."""

    name = "de"

    def __init__(self, pop_size: int = 8, F: float = 0.5, CR: float = 0.7):
        self.pop_size, self.F, self.CR = pop_size, F, CR

    def run(self, ctx: SearchContext) -> str:
        init = self.evaluate_initial(ctx)
        de = DifferentialEvolution(ctx.space.dim, self.pop_size, self.F, self.CR, rng=ctx.rng)
        de.initialize(ctx.initial_X, [r.fitness if r.valid else None for r in init])
        while True:
            target = de.next_target()
            trial = de.make_trial(target)
            result = ctx.tracker.evaluate(x=trial, tag="de")
            de.select(target, trial, result.fitness if result.valid else None)


class PatienceDE(StandardDE):
    """Naive-stopping baseline (D44): standard DE that stops once the incumbent has not improved
    for ``patience_evals`` consecutive evaluations, but never before ``min_budget_fraction`` of the
    cap (the same floor as the controller). No utility signal, block allocation or confirmation,
    so it isolates what the controller adds beyond "stop when progress flattens"."""

    name = "de_patience"

    def __init__(self, pop_size: int = 8, F: float = 0.5, CR: float = 0.7, patience_evals: int = 16,
                 min_budget_fraction: float = 0.4):
        super().__init__(pop_size, F, CR)
        self.patience_evals = patience_evals
        self.min_budget_fraction = min_budget_fraction
        self.stopped_at: float | None = None

    def run(self, ctx: SearchContext) -> str:
        init = self.evaluate_initial(ctx)
        de = DifferentialEvolution(ctx.space.dim, self.pop_size, self.F, self.CR, rng=ctx.rng)
        de.initialize(ctx.initial_X, [r.fitness if r.valid else None for r in init])
        best, since = ctx.tracker.best, 0
        while True:
            target = de.next_target()
            trial = de.make_trial(target)
            result = ctx.tracker.evaluate(x=trial, tag="de_patience")
            de.select(target, trial, result.fitness if result.valid else None)
            if ctx.tracker.best > best:
                best, since = ctx.tracker.best, 0
            else:
                since += 1
            if since >= self.patience_evals and ctx.clock.elapsed() >= self.min_budget_fraction * ctx.clock.limit_s:
                self.stopped_at = ctx.clock.elapsed()
                return "patience_stop"

    def summary_extra(self) -> dict:
        return {"patience_evals": self.patience_evals, "min_budget_fraction": self.min_budget_fraction,
                "stopped_at_s": self.stopped_at}
