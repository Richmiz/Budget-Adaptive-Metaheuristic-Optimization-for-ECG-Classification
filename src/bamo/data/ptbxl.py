"""PTB-XL v1.0.3 preparation (protocol §3.1).

Label mapping follows the official benchmark (Strodthoff et al., 2021): every SCP statement
in a record that is a diagnostic statement in ``scp_statements.csv`` contributes its
``diagnostic_class``; likelihood values are not thresholded. Records without any eligible
superclass are excluded and counted (decision D01).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def load_metadata(raw_root: str | Path):
    import pandas as pd

    raw_root = Path(raw_root)
    df = pd.read_csv(raw_root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(raw_root / "scp_statements.csv", index_col=0)
    return df, scp


def map_superclasses(df, scp):
    diagnostic = scp[scp["diagnostic"] == 1]
    classes = diagnostic["diagnostic_class"].to_dict()
    out = df.copy()
    out["superclasses"] = out["scp_codes"].apply(lambda codes: sorted({classes[c] for c in codes if c in classes}))
    return out


def task_label_lists(df, scp, task: str, classes) -> list[list[str]]:
    """Per-record labels for a study task. Superclass: benchmark mapping above. Rhythm/form: the
    record's SCP statements of that type, restricted to a frozen class list (no likelihood threshold)."""
    if task == "superclass":
        return map_superclasses(df, scp)["superclasses"].tolist()
    if task in ("rhythm", "form"):
        eligible = set(scp.index[scp[task] == 1]) & set(classes)
        if eligible != set(classes):
            raise ValueError(f"{task}: classes not flagged as {task} statements: {sorted(set(classes) - eligible)}")
        return [sorted(set(codes) & eligible) for codes in df["scp_codes"]]
    raise ValueError(f"unknown PTB-XL task: {task}")


def label_matrix(superclasses, classes=CLASSES) -> np.ndarray:
    Y = np.zeros((len(superclasses), len(classes)), dtype=np.uint8)
    index = {c: j for j, c in enumerate(classes)}
    for i, labels in enumerate(superclasses):
        for c in labels:
            Y[i, index[c]] = 1
    return Y


def load_signals(df, raw_root: str | Path, sampling_rate: int = 100, n_leads: int = 12):
    """Read waveforms with wfdb. Returns (X[N, leads, samples] float32, issues per row)."""
    import wfdb

    raw_root = Path(raw_root)
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    n_samples = 10 * sampling_rate
    X = np.zeros((len(df), n_leads, n_samples), dtype=np.float32)
    issues: dict[int, list[str]] = {}
    for i, filename in enumerate(df[column]):
        signal, _ = wfdb.rdsamp(str(raw_root / filename))
        problems = []
        if signal.ndim != 2 or signal.shape[1] != n_leads:
            problems.append(f"lead_count:{signal.shape}")
        elif signal.shape[0] != n_samples:
            problems.append(f"length:{signal.shape[0]}")
        else:
            if not np.isfinite(signal).all():
                problems.append("nonfinite")
            X[i] = signal.T.astype(np.float32)
        if problems:
            issues[i] = problems
    return X, issues


def duplicate_groups(X: np.ndarray) -> list[list[int]]:
    """Groups of rows whose waveforms are byte-identical."""
    import hashlib

    seen: dict[str, list[int]] = {}
    for i in range(len(X)):
        seen.setdefault(hashlib.sha1(X[i].tobytes()).hexdigest(), []).append(i)
    return [rows for rows in seen.values() if len(rows) > 1]


def patient_fold_audit(patient_ids: np.ndarray, folds: np.ndarray) -> dict:
    """Every patient must belong to exactly one strat_fold."""
    folds_per_patient: dict[int, set] = {}
    for p, f in zip(patient_ids, folds):
        folds_per_patient.setdefault(int(p), set()).add(int(f))
    leaking = {p: sorted(fs) for p, fs in folds_per_patient.items() if len(fs) > 1}
    return {"n_patients": len(folds_per_patient), "patients_in_multiple_folds": len(leaking),
            "examples": dict(list(leaking.items())[:10])}


def partition_rows(folds: np.ndarray, partition_cfg: dict) -> dict[str, np.ndarray]:
    """Row indices for each role (train/val/test) of a partition definition."""
    return {role: np.flatnonzero(np.isin(folds, fold_list)) for role, fold_list in partition_cfg.items()}


def audit_partition(patient_ids: np.ndarray, rows: dict[str, np.ndarray]) -> dict:
    roles = list(rows)
    patients = {r: set(map(int, patient_ids[idx])) for r, idx in rows.items()}
    overlaps = {f"{a}&{b}": len(patients[a] & patients[b])
                for i, a in enumerate(roles) for b in roles[i + 1:]}
    return {"n_records": {r: int(len(idx)) for r, idx in rows.items()},
            "n_patients": {r: len(p) for r, p in patients.items()},
            "patient_overlap": overlaps, "ok": all(v == 0 for v in overlaps.values())}
