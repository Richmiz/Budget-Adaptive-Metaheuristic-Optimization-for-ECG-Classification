"""Freeze the protocol and, later, lock the analysis.

    python scripts/lock_protocol.py freeze      # after the pilot: snapshot merged config -> protocol/frozen_protocol.yaml
    python scripts/lock_protocol.py analysis    # after all search + final training: unlocks fold 10

``freeze`` refuses while any FREEZE value is still null. ``analysis`` records hashes of the
frozen protocol and the analysis plan; the test evaluation refuses to run without it.
"""

import _paths  # noqa: F401

import argparse
import time
from pathlib import Path

import yaml

from bamo.config import PROJECT_ROOT, config_hash, load_config
from bamo.utils import read_json, sha256_file, write_json

PROTOCOL = PROJECT_ROOT / "protocol"
REQUIRED = [("budgets", "t_ref_seconds"), ("controller", "reeval_tolerance"), ("hardware", "required_gpu")]


def freeze() -> None:
    cfg = load_config()
    missing = [".".join(k) for k in REQUIRED if cfg[k[0]][k[1]] is None]
    if missing:
        raise SystemExit(f"cannot freeze, still null: {missing}")
    path = PROTOCOL / "frozen_protocol.yaml"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen protocol is never overwritten (amend via docs/08 instead)")
    PROTOCOL.mkdir(exist_ok=True)
    stamp = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": config_hash(cfg)}
    path.write_text(yaml.safe_dump({"_freeze": stamp, **cfg}, sort_keys=False), encoding="utf-8")
    print(f"frozen protocol written: {path} (config hash {stamp['config_hash']})")


def lock_analysis() -> None:
    frozen = PROTOCOL / "frozen_protocol.yaml"
    if not frozen.exists():
        raise SystemExit("freeze the protocol first")
    lock = PROTOCOL / "analysis_lock.json"
    if lock.exists():
        raise SystemExit(f"analysis already locked: {read_json(lock)}")
    write_json(lock, {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "frozen_protocol_sha256": sha256_file(frozen),
                      "analysis_plan_sha256": sha256_file(PROJECT_ROOT / "docs" / "07_analysis_plan.md")})
    print(f"analysis locked: {lock}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "analysis"])
    args = ap.parse_args()
    freeze() if args.action == "freeze" else lock_analysis()


if __name__ == "__main__":
    main()
