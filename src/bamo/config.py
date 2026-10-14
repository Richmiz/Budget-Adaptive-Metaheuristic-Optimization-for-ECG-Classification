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


def task_names() -> list[str]:
    return list(load_yaml(CONFIG_DIR / "tasks.yaml")["tasks"])


def load_config(experiment: str | None = None, overrides: dict | None = None, task: str | None = None) -> dict:
    """Merge base, search-space, controller and method configs, then a task, then an experiment file.

    A task (configs/tasks.yaml) replaces the data directory and classes, T_ref and the controller's
    re-evaluation tolerance, and records its default configurations under ``cfg["task"]``.
    """
    cfg: dict[str, Any] = {}
    for name in ("base.yaml", "search_space.yaml", "controller.yaml", "methods.yaml"):
        cfg = deep_merge(cfg, load_yaml(CONFIG_DIR / name))
    if task:
        tasks = load_yaml(CONFIG_DIR / "tasks.yaml")["tasks"]
        if task not in tasks:
            raise ValueError(f"unknown task {task!r}; choose from {list(tasks)}")
        spec = copy.deepcopy(tasks[task])
        defaults = spec.pop("defaults")
        cfg = deep_merge(cfg, spec)
        reference = cfg["search_space"]["reference_config"]
        cfg["task"] = {"name": task, "defaults": {name: {**reference, **override} for name, override in defaults.items()}}
    if experiment:
        cfg = deep_merge(cfg, {"experiment": load_yaml(CONFIG_DIR / "experiments" / f"{experiment}.yaml")})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg)


def frozen_status(task: str | None = None) -> str:
    """'not_frozen', 'match' or 'mismatch' between configs/ and protocol/frozen_protocol.yaml.

    With ``task``, the task-merged configuration is compared with the hash recorded for that task.
    """
    path = PROJECT_ROOT / "protocol" / "frozen_protocol.yaml"
    if not path.exists():
        return "not_frozen"
    stamp = load_yaml(path)["_freeze"]
    if task is None:
        return "match" if stamp["config_hash"] == config_hash(load_config()) else "mismatch"
    expected = stamp.get("task_config_hashes", {}).get(task)
    return "match" if expected == config_hash(load_config(task=task)) else "mismatch"
