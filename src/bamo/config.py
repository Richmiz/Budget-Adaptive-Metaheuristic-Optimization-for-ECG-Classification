"""Configuration loading. Configs are plain YAML merged in order; the merged dict is hashed
so every run records exactly which frozen settings it used."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(experiment: str | None = None, overrides: dict | None = None) -> dict:
    """Merge base, search-space, controller and method configs, then an experiment file."""
    cfg: dict[str, Any] = {}
    for name in ("base.yaml", "search_space.yaml", "controller.yaml", "methods.yaml"):
        cfg = deep_merge(cfg, load_yaml(CONFIG_DIR / name))
    if experiment:
        cfg = deep_merge(cfg, {"experiment": load_yaml(CONFIG_DIR / "experiments" / f"{experiment}.yaml")})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg)


def frozen_status() -> str:
    """'not_frozen', 'match' or 'mismatch' between configs/ and protocol/frozen_protocol.yaml."""
    path = PROJECT_ROOT / "protocol" / "frozen_protocol.yaml"
    if not path.exists():
        return "not_frozen"
    frozen = load_yaml(path)
    return "match" if frozen["_freeze"]["config_hash"] == config_hash(load_config()) else "mismatch"
