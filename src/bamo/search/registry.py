"""Builds search methods from ``configs/methods.yaml`` (and ``configs/controller.yaml`` for P)."""

from __future__ import annotations

from .controller import MarginalUtilitySearch
from .methods.base import SearchMethod
from .methods.hyperband import BOHB, Hyperband
from .methods.library import TPE, DEHBAdapter
from .methods.simple import PatienceDE, RandomSearch, StandardDE

CLASSES = {
    "random": RandomSearch,
    "standard_de": StandardDE,
    "de_patience": PatienceDE,
    "hyperband": Hyperband,
    "bohb": BOHB,
    "tpe": TPE,
    "dehb": DEHBAdapter,
    "mu": MarginalUtilitySearch,
}


def build_method(name: str, cfg: dict) -> SearchMethod:
    spec = dict(cfg["methods"][name])
    cls = CLASSES[spec.pop("class")]
    if cls is MarginalUtilitySearch:
        kwargs = dict(cfg["controller"])
        kwargs.update(cfg["de"])
        kwargs.update(spec)  # ablation overrides
    elif cls in (StandardDE, PatienceDE):
        kwargs = {**cfg["de"], **spec}
    else:
        kwargs = spec
    method = cls(**kwargs)
    method.name = name
    return method


def is_multi_fidelity(name: str, cfg: dict) -> bool:
    return CLASSES[cfg["methods"][name]["class"]].multi_fidelity
