"""Adapters for comparators with official library implementations.

* B2 TPE  -> Optuna ``TPESampler`` through the ask/tell interface.
* B6 DEHB -> the ``dehb`` package (Awad et al., 2021) through its ask/tell interface.

Both run inside the shared tracker, so the time cap, caching and incumbent rules match the
in-house methods. The DEHB adapter targets the ask/tell API of ``dehb`` >= 0.1; check it
against the installed, pinned version before the pilot (decision D27).
"""

from __future__ import annotations

import math

import numpy as np

from .base import SearchContext, SearchMethod


class TPE(SearchMethod):
    name = "tpe"

    def __init__(self, n_startup_trials: int = 8, multivariate: bool = False):
        self.n_startup_trials = n_startup_trials
        self.multivariate = multivariate

    @staticmethod
    def distributions(space):
        from optuna.distributions import CategoricalDistribution, FloatDistribution

        out = {}
        for p in space.params:
            if p.kind == "log":
                out[p.name] = FloatDistribution(p.low, p.high, log=True)
            elif p.kind == "float":
                out[p.name] = FloatDistribution(p.low, p.high)
            else:
                out[p.name] = CategoricalDistribution(list(p.choices))
        return out

    def run(self, ctx: SearchContext) -> str:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=int(ctx.rng.integers(2**31)),
                                             n_startup_trials=self.n_startup_trials,
                                             multivariate=self.multivariate)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for x in ctx.initial_X:  # shared design is TPE's start-up set (D10)
            study.enqueue_trial(ctx.space.decode(x))
        dists = self.distributions(ctx.space)
        n_init = len(ctx.initial_X)
        i = 0
        while True:
            trial = study.ask(dists)
            config = {name: trial.params[name] for name in ctx.space.names}
            tag = "init" if i < n_init else "tpe"
            result = ctx.tracker.evaluate(config=config, tag=tag)
            if result.valid:
                study.tell(trial, result.fitness)
            else:
                study.tell(trial, state=TrialState.FAIL)
            i += 1


class DEHBAdapter(SearchMethod):
    name = "dehb"
    multi_fidelity = True

    def __init__(self, eta: int = 2, output_dir: str | None = None, shared_initial: bool = True):
        self.eta = eta
        self.output_dir = output_dir
        self.shared_initial = shared_initial

    @staticmethod
    def configspace(space, seed: int):
        import ConfigSpace as CS

        cs = CS.ConfigurationSpace(seed=seed)
        hps = []
        for p in space.params:
            if p.kind == "log":
                hps.append(CS.UniformFloatHyperparameter(p.name, p.low, p.high, log=True))
            elif p.kind == "float":
                hps.append(CS.UniformFloatHyperparameter(p.name, p.low, p.high))
            else:
                hps.append(CS.OrdinalHyperparameter(p.name, list(p.choices)))
        if hasattr(cs, "add"):
            cs.add(hps)
        else:
            cs.add_hyperparameters(hps)
        return cs

    def run(self, ctx: SearchContext) -> str:
        from dehb import DEHB  # import before any evaluation is spent
        from loguru import logger

        logger.disable("dehb")  # per-step logging would be charged to DEHB's budget
        seed = int(ctx.rng.integers(2**31))
        cs = self.configspace(ctx.space, seed)
        # The shared initial design is evaluated and charged first; DEHB cannot ingest it,
        # so it only contributes to the incumbent (asymmetry recorded in D10).
        if self.shared_initial:
            self.evaluate_initial(ctx)
        out_dir = self.output_dir or (str(ctx.run_dir / "dehb") if ctx.run_dir else "dehb_output")
        dehb = DEHB(cs=cs, dimensions=len(ctx.space.params), min_fidelity=ctx.fidelities[0],
                    max_fidelity=ctx.max_fidelity, eta=self.eta, n_workers=1, output_path=out_dir, seed=seed,
                    save_freq="end")  # "end" only saves inside DEHB.run(); ask/tell never writes state to disk
        while True:
            job = dehb.ask()
            config = job["config"]
            if isinstance(config, np.ndarray):
                config = dehb.vector_to_configspace(config)
            config = {k: config[k] for k in ctx.space.names}
            fidelity = int(round(job["fidelity"]))
            result = ctx.tracker.evaluate(config=config, fidelity=fidelity, tag="dehb")
            loss = -result.fitness if result.valid else math.inf  # DEHB minimizes
            dehb.tell(job, {"fitness": loss, "cost": result.cost_s})
