"""Development-only PTB-XL tasks using official SCP form/rhythm flags.

Vocabulary is selected from folds 1-6, never from held-out folds. Waveforms outside
folds 1-7 are never opened by this preparation route. A task is not silently reduced
when a retained label is absent from validation: that is a failed feasibility check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ptbxl import (CLASSES, audit_partition, label_matrix, load_metadata,
                    load_signals, map_superclasses, patient_fold_audit)
from ..utils import read_json, sha256_file, stable_hash, write_json


def task_metadata(df, scp, task: str, min_train_count: int = 50):
    df = df[df.strat_fold.isin(range(1, 8))].copy()
    if task == "superclass":
        df["task_labels"] = map_superclasses(df, scp)["superclasses"]
        classes = CLASSES
    elif task in ("form", "rhythm"):
        eligible = set(scp.index[scp[task] == 1])
        df["task_labels"] = df.scp_codes.apply(lambda codes: sorted(set(codes) & eligible))
        counts = {c: sum(c in labels for labels in df.loc[df.strat_fold <= 6, "task_labels"])
                  for c in sorted(eligible)}
        classes = tuple(c for c, n in counts.items() if n > min_train_count)
        df["task_labels"] = df.task_labels.apply(lambda labels: sorted(set(labels) & set(classes)))
    else:
        raise ValueError(f"unknown PTB-XL task: {task}")
    if not classes:
        raise ValueError(f"{task}: no eligible training labels")
    df = df[df.task_labels.apply(len) > 0].sort_index()
    Y = label_matrix(df.task_labels, classes)
    folds = df.strat_fold.to_numpy()
    train_counts = Y[folds <= 6].sum(0)
    val_counts = Y[folds == 7].sum(0)
    n_val = int((folds == 7).sum())
    if n_val == 0 or np.any((val_counts == 0) | (val_counts == n_val)):
        raise ValueError(f"{task}: a fixed training-derived label lacks validation positives or negatives")
    audit = patient_fold_audit(df.patient_id.to_numpy(), folds)
    if audit["patients_in_multiple_folds"]:
        raise ValueError("patient overlap in development folds")
    rows = {"train": np.flatnonzero(folds <= 6), "val": np.flatnonzero(folds == 7)}
    return df, Y, tuple(classes), {
        "task": task, "classes": list(classes), "min_train_count_exclusive": min_train_count,
        "folds": {"train": [1, 2, 3, 4, 5, 6], "val": [7]},
        "train_positive_counts": dict(zip(classes, map(int, train_counts))),
        "val_positive_counts": dict(zip(classes, map(int, val_counts))),
        "patient_fold_audit": audit, "partition": audit_partition(df.patient_id.to_numpy(), rows),
        "source_ecg_ids_hash": stable_hash(list(map(int, df.index))),
        "excluded_folds": [8, 9, 10],
        "vocabulary_rule": "SCP task flag; >50 positives on train folds only; no likelihood threshold",
    }


def prepare_pilot_task(raw_root: str | Path, out_root: str | Path, task: str,
                       min_train_count: int = 50) -> Path:
    out = Path(out_root) / task
    raw_root = Path(raw_root)
    df, scp = load_metadata(raw_root)
    df, Y, classes, report = task_metadata(df, scp, task, min_train_count)
    report["raw_metadata_sha256"] = {
        name: sha256_file(raw_root / name) for name in ("ptbxl_database.csv", "scp_statements.csv")}
    spec_hash = stable_hash(report)
    if (out / "data_report.json").exists():
        old = read_json(out / "data_report.json")
        if old.get("spec_hash") != spec_hash:
            raise ValueError(f"{out}: preparation specification changed; choose a new output directory")
        for name, digest in old["array_sha256"].items():
            if sha256_file(out / name) != digest:
                raise ValueError(f"{out / name}: cached array hash mismatch")
        return out
    out.mkdir(parents=True, exist_ok=True)
    X, issues = load_signals(df, raw_root, 100)
    if issues:
        raise ValueError(f"{task}: waveform issues, investigate before training: {issues}")
    records = df.reset_index()[["ecg_id", "patient_id", "strat_fold", "filename_lr"]].copy()
    for j, name in enumerate(classes):
        records[f"y_{name}"] = Y[:, j]
    records.to_csv(out / "records.csv", index=False)
    np.save(out / "signals_100hz.npy", X)
    np.save(out / "labels.npy", Y)
    report.update(spec_hash=spec_hash, n_records=len(df), n_patients=int(df.patient_id.nunique()),
                  array_sha256={name: sha256_file(out / name) for name in ("signals_100hz.npy", "labels.npy")})
    write_json(out / "data_report.json", report)
    return out
