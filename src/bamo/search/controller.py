"""Marginal-utility block-allocation controller on top of DE (protocol §3.3–3.4).

``MarginalUtilityPolicy`` is the pure decision rule (rewards, priority scores, continue /
switch / probe choices, stop proposal). ``MarginalUtilitySearch`` executes it: it runs DE
rounds inside the chosen block, measures cost, and handles confirmation and termination.
See docs/04_controller_specification.md for the formal specification and the decision
register entries (D14–D24) that resolve ambiguities in the proposal.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .de import DifferentialEvolution, fitness_or_ninf
from .methods.base import SearchContext, SearchMethod


@dataclass
class ControllerSettings:
    blocks: tuple[str, ...] = ("A", "B", "C", "J")
    trials_per_round: int = 2
    warmup_rounds_per_block: int = 3
    window: int = 8                    # rolling window, per block (D15)
    kappa: float = 1.0
    sigma_floor: float = 1e-4          # AUROC per normalized cost unit
    tau: float = 5e-4                  # utility threshold, AUROC per normalized cost unit
    hysteresis_factor: float = 0.1     # h = factor * tau
    probe_gap: int = 8                 # mandatory probe after this many decisions without allocation
    stop_consecutive: int = 3
    min_rounds_per_block: int = 3
    min_budget_fraction: float = 0.4
    reeval_seeds: int = 2
    reeval_tolerance: float | None = None  # frozen after the pilot (D21)
    epsilon: float = 1e-9
    # Ablation switches (protocol §3.6.1)
    adaptive: bool = True              # False: round-robin allocation
    global_termination: bool = True
    cost_normalization: bool = True    # False: delta_c = number of evaluations in the round (D23)
    mandatory_probes: bool = True
    confirmation: bool = True
    stop_rule: str = "utility"         # matched controls: utility or simple evaluation patience
    patience_evals: int = 16
    confirmation_rule: str = "utility"  # legacy; matched controls all use any-improvement veto

    def __post_init__(self) -> None:
        self.blocks = tuple(self.blocks)
        if self.trials_per_round < 1 or self.window < 1:
            raise ValueError("trials_per_round and window must be positive")
        if self.stop_rule not in ("utility", "patience") or self.confirmation_rule not in ("utility", "improvement"):
            raise ValueError("unknown stopping or confirmation rule")
        if self.patience_evals < 1:
            raise ValueError("patience_evals must be positive")

    @property
    def hysteresis(self) -> float:
        return self.hysteresis_factor * self.tau


class MarginalUtilityPolicy:
    def __init__(self, settings: ControllerSettings):
        self.s = settings
        self.blocks = settings.blocks
        self.rewards = {k: deque(maxlen=settings.window) for k in self.blocks}
        self.rounds = {k: 0 for k in self.blocks}
        self.last_round: dict[str, int | None] = {k: None for k in self.blocks}
        self.t = 0                      # completed allocation rounds (global)
        self.current: str | None = None
        self.streak = 0                 # consecutive post-warm-up decisions with all U_k < tau
        self._rr = 0

    @property
    def in_warmup(self) -> bool:
        return any(self.rounds[k] < self.s.warmup_rounds_per_block for k in self.blocks)

    def delta_cost(self, round_seconds: float, n_evals: int, t_ref: float) -> float:
        return round_seconds / t_ref if self.s.cost_normalization else float(n_evals)

    def reward(self, delta_m: float, round_seconds: float, n_evals: int, t_ref: float) -> tuple[float, float]:
        """r_t = max(0, dM_t) / max(dc_t, eps); returns (r_t, dc_t)."""
        dc = self.delta_cost(round_seconds, n_evals, t_ref)
        return max(0.0, delta_m) / max(dc, self.s.epsilon), dc

    def score(self, block: str) -> float:
        """U_k = mean(r_k) + kappa * sigma_k * sqrt(2 ln(1 + t) / (1 + n_k))."""
        r = self.rewards[block]
        n = len(r)
        if n == 0:
            return math.inf
        sd = statistics.stdev(r) if n >= 2 else 0.0
        sigma = max(sd, self.s.sigma_floor)
        return sum(r) / n + self.s.kappa * sigma * math.sqrt(2.0 * math.log(1.0 + self.t) / (1.0 + n))

    def scores(self) -> dict[str, float]:
        return {k: self.score(k) for k in self.blocks}

    def stale_blocks(self) -> list[str]:
        """Blocks with no allocation in the last ``probe_gap`` decisions."""
        return [k for k in self.blocks
                if self.last_round[k] is None or self.t - self.last_round[k] > self.s.probe_gap]

    def choose(self) -> tuple[str, str, dict[str, float]]:
        scores = self.scores()
        order = {k: i for i, k in enumerate(self.blocks)}
        if self.in_warmup:  # interleaved warm-up A, B, C, J, A, ... (D14)
            return "warmup", min(self.blocks, key=lambda b: (self.rounds[b], order[b])), scores
        if not self.s.adaptive:
            block = self.blocks[self._rr % len(self.blocks)]
            self._rr += 1
            return "round_robin", block, scores
        if self.s.mandatory_probes:
            stale = self.stale_blocks()
            if stale:
                longest = min(stale, key=lambda b: (-1 if self.last_round[b] is None else self.last_round[b], order[b]))
                return "probe", longest, scores
        best = max(self.blocks, key=lambda b: (scores[b], -order[b]))
        if self.current is not None and scores[self.current] >= scores[best] - self.s.hysteresis:
            return "continue", self.current, scores
        return "switch", best, scores

    def record(self, block: str, reward: float, count_for_stop: bool = True) -> None:
        was_warmup = self.in_warmup
        self.rewards[block].append(reward)
        self.rounds[block] += 1
        self.last_round[block] = self.t
        self.t += 1
        self.current = block
        if count_for_stop and not was_warmup:  # D19
            below = all(u < self.s.tau for u in self.scores().values())
            self.streak = self.streak + 1 if below else 0

    def stop_proposed(self, elapsed_s: float, budget_s: float) -> bool:
        return (self.streak >= self.s.stop_consecutive
                and all(self.rounds[k] >= self.s.min_rounds_per_block for k in self.blocks)
                and elapsed_s >= self.s.min_budget_fraction * budget_s)

    def reset_streak(self) -> None:
        self.streak = 0


@dataclass
class _RoundOutcome:
    block: str
    reward: float
    delta_m: float
    seconds: float
    delta_c: float
    n_evals: int
    improved_targets: int = 0
    duplicates_repaired: int = 0


class MarginalUtilitySearch(SearchMethod):
    """Proposed method P and its ablation variants (selected through ``ControllerSettings``)."""

    name = "mu"

    def __init__(self, pop_size: int = 8, F: float = 0.5, CR: float = 0.7, max_duplicate_retries: int = 10,
                 **controller_settings):
        self.pop_size, self.F, self.CR = pop_size, F, CR
        self.max_duplicate_retries = max_duplicate_retries
        self.settings = ControllerSettings(**controller_settings)
        self.policy: MarginalUtilityPolicy | None = None
        self.de: DifferentialEvolution | None = None
        self.round_seconds: list[float] = []
        self.shadow_stop: dict | None = None
        self.unstable_incumbent: str | None = None
        self.confirmations: list[dict] = []
        self.confirmation_infeasible_at: float | None = None
        self.action_counts: dict[str, int] = {}
        self.evals_since_improvement = 0

    # ------------------------------------------------------------------ helpers
    def _trial(self, ctx: SearchContext, target: int, active: np.ndarray) -> tuple[np.ndarray, bool]:
        """Generate a trial whose decoded configuration differs from its target (D11)."""
        target_key = ctx.space.config_key(ctx.space.decode(self.de.X[target]))
        for _ in range(self.max_duplicate_retries):
            trial = self.de.make_trial(target, active)
            if ctx.space.config_key(ctx.space.decode(trial)) != target_key:
                return trial, False
        # Force one active coordinate into a different value/bin.
        trial = self.de.X[target].copy()
        j = int(ctx.rng.choice(active))
        p = ctx.space.params[j]
        if p.kind == "choice":
            current = p.choices.index(p.decode(trial[j]))
            other = int(ctx.rng.choice([i for i in range(len(p.choices)) if i != current]))
            trial[j] = (other + ctx.rng.random()) / len(p.choices)
        else:
            trial[j] = ctx.rng.random()
        return trial, True

    def _round(self, ctx: SearchContext, block: str, since: float, tag: str) -> _RoundOutcome:
        s = self.settings
        active = ctx.space.block_indices(block)
        m_before = ctx.tracker.best
        improved = repaired = 0
        for _ in range(s.trials_per_round):
            target = self.de.next_target()
            trial, forced = self._trial(ctx, target, active)
            repaired += forced
            result = ctx.tracker.evaluate(x=trial, tag=f"{tag}:{block}")
            improved += self.de.select(target, trial, result.fitness if result.valid else None)
            # Count completed distinct search evaluations, not cached duplicates or confirmation.
            if tag != "confirm" and ctx.tracker.records[-1]["source"] != "run_cache":
                if ctx.tracker.records[-1]["new_incumbent"]:
                    self.evals_since_improvement = 0
                else:
                    self.evals_since_improvement += 1
        seconds = ctx.clock.elapsed() - since
        m_after = ctx.tracker.best
        delta_m = 0.0 if not math.isfinite(m_before) else max(0.0, m_after - m_before)
        reward, delta_c = self.policy.reward(delta_m, seconds, s.trials_per_round, ctx.t_ref)
        self.round_seconds.append(seconds)
        return _RoundOutcome(block, reward, delta_m, seconds, delta_c, s.trials_per_round, improved, repaired)

    def _log(self, ctx: SearchContext, phase: str, action: str, scores: dict, out: _RoundOutcome | None,
             **extra) -> None:
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        entry = {
            "round": self.policy.t, "elapsed_s": ctx.clock.elapsed(), "phase": phase, "action": action,
            "scores": {k: (None if math.isinf(v) else v) for k, v in scores.items()},
            "incumbent": ctx.tracker.best if ctx.tracker.incumbent else None, "streak": self.policy.streak,
        }
        if out is not None:
            entry.update(block=out.block, reward=out.reward, delta_m=out.delta_m, round_s=out.seconds,
                         delta_c=out.delta_c, improved_targets=out.improved_targets,
                         duplicates_repaired=out.duplicates_repaired)
        entry.update(extra)
        ctx.log_decision(entry)

    def _incumbent_key(self, ctx: SearchContext) -> str | None:
        inc = ctx.tracker.incumbent
        return ctx.space.config_key(inc["config"]) if inc else None

    def _confirmation_feasible(self, ctx: SearchContext) -> bool:
        s = self.settings
        est_round = statistics.median(self.round_seconds) if self.round_seconds else ctx.t_ref * s.trials_per_round
        costs = [r["cost_s"] for r in ctx.tracker.records
                 if r["source"] == "evaluated" and r["fidelity"] == ctx.max_fidelity and r["status"] == "ok"]
        est_eval = statistics.median(costs) if costs else ctx.t_ref
        need = len(s.blocks) * est_round + s.reeval_seeds * est_eval
        return ctx.clock.remaining() >= need

    def _confirm(self, ctx: SearchContext) -> bool:
        """Probe every block once more, then re-evaluate the incumbent with extra seeds."""
        s = self.settings
        attempt = {"started_s": ctx.clock.elapsed(), "probes": [], "reeval": None, "outcome": None}
        self.confirmations.append(attempt)
        for block in s.blocks:
            scores = self.policy.scores()
            out = self._round(ctx, block, ctx.clock.elapsed(), tag="confirm")
            self.policy.record(block, out.reward, count_for_stop=False)
            attempt["probes"].append({"block": block, "reward": out.reward})
            self._log(ctx, "confirm", "confirm_probe", scores, out)
            veto = out.reward >= s.tau if s.confirmation_rule == "utility" else out.delta_m > 0
            if veto:
                attempt["outcome"] = f"resume_probe_{block}"
                return False
        inc = ctx.tracker.incumbent
        values = []
        for seed in ctx.reeval_seeds[: s.reeval_seeds]:
            r = ctx.tracker.evaluate(config=inc["config"], seed=seed, tag="reeval")
            values.append(r.fitness if r.valid else math.nan)
        deviation = max((abs(v - inc["fitness"]) if math.isfinite(v) else math.inf) for v in values)
        stable = deviation <= s.reeval_tolerance
        attempt["reeval"] = {"incumbent": inc["fitness"], "values": values, "max_deviation": deviation,
                             "tolerance": s.reeval_tolerance, "stable": stable}
        self._log(ctx, "confirm", "reevaluate", self.policy.scores(), None, reeval=attempt["reeval"])
        if not stable:
            self.unstable_incumbent = self._incumbent_key(ctx)  # D21: block stopping until the incumbent changes
            attempt["outcome"] = "resume_unstable_incumbent"
            return False
        attempt["outcome"] = "confirmed"
        return True

    def _stop_proposed(self, ctx: SearchContext) -> bool:
        s = self.settings
        if s.stop_rule == "utility":
            return self.policy.stop_proposed(ctx.clock.elapsed(), ctx.clock.limit_s)
        return (not self.policy.in_warmup
                and all(n >= s.min_rounds_per_block for n in self.policy.rounds.values())
                and self.evals_since_improvement >= s.patience_evals
                and ctx.clock.elapsed() >= s.min_budget_fraction * ctx.clock.limit_s)

    # --------------------------------------------------------------------- main
    def budget_reason(self) -> str:
        return "budget_during_warmup" if self.policy is None or self.policy.in_warmup else "budget"

    def run(self, ctx: SearchContext) -> str:
        s = self.settings
        if s.global_termination and s.confirmation and s.reeval_tolerance is None:
            raise ValueError("reeval_tolerance must be frozen (pilot) before running with confirmation")
        if len(ctx.reeval_seeds) < s.reeval_seeds:
            raise ValueError("not enough re-evaluation seeds supplied")
        unknown = [b for b in s.blocks if b not in ctx.space.blocks]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")

        init = self.evaluate_initial(ctx)
        self.de = DifferentialEvolution(ctx.space.dim, self.pop_size, self.F, self.CR, rng=ctx.rng)
        self.de.initialize(ctx.initial_X, [r.fitness if r.valid else None for r in init])
        self.policy = MarginalUtilityPolicy(s)
        last_end = ctx.clock.elapsed()

        while True:
            action, block, scores = self.policy.choose()
            phase = "warmup" if action == "warmup" else "search"
            out = self._round(ctx, block, last_end, tag=phase)  # round time includes decision overhead (D18)
            last_end = ctx.clock.elapsed()
            self.policy.record(block, out.reward)
            self._log(ctx, phase, action, scores, out)

            if not self._stop_proposed(ctx):
                continue
            if not s.global_termination:
                if self.shadow_stop is None:  # D24: hypothetical stop for the premature-termination analysis
                    self.shadow_stop = {"elapsed_s": ctx.clock.elapsed(), "round": self.policy.t,
                                        "incumbent": ctx.tracker.best,
                                        "incumbent_config": dict(ctx.tracker.incumbent["config"])}
                continue
            if not s.confirmation:
                self._log(ctx, "stop", "stop_unconfirmed", self.policy.scores(), None)
                return f"{s.stop_rule}_stop_unconfirmed"
            if self.unstable_incumbent is not None and self.unstable_incumbent == self._incumbent_key(ctx):
                continue
            if not self._confirmation_feasible(ctx):  # D22: finish as budget exhaustion
                if self.confirmation_infeasible_at is None:
                    self.confirmation_infeasible_at = ctx.clock.elapsed()
                    self._log(ctx, "stop", "confirmation_infeasible", self.policy.scores(), None)
                continue
            self._log(ctx, "stop", "propose_stop", self.policy.scores(), None)
            confirmed = self._confirm(ctx)
            last_end = ctx.clock.elapsed()
            if confirmed:
                self._log(ctx, "stop", "confirm_stop", self.policy.scores(), None)
                return f"{s.stop_rule}_stop_confirmed"
            self.policy.reset_streak()
            self.evals_since_improvement = 0

    def summary_extra(self) -> dict:
        p = self.policy
        return {
            "settings": asdict(self.settings),
            "rounds_per_block": dict(p.rounds) if p else {},
            "action_counts": dict(self.action_counts),
            "shadow_stop": self.shadow_stop,
            "confirmations": self.confirmations,
            "confirmation_infeasible_at": self.confirmation_infeasible_at,
            "evals_since_improvement": self.evals_since_improvement,
        }
