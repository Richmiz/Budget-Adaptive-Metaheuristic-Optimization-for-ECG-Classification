"""Build the analysis dataset from raw PTB-XL (protocol §3.1, §3.1.1).

    python scripts/prepare_data.py                  # diagnostic superclasses -> data/processed
    python scripts/prepare_data.py --task rhythm    # frozen rhythm labels -> data/processed_rhythm (configs/tasks.yaml)

Outputs in the task's processed directory:
    signals_100hz.npy   (N, 12, 1000) float32, raw millivolts (normalization happens per partition)
    labels.npy          (N, n_classes) uint8 multi-hot task labels
    records.csv         row -> ecg_id, patient_id, strat_fold, superclasses, quality flags
    data_report.json    exclusions by reason, class counts per fold, duplicates, patient audit
    split_manifest.json per partition: folds, record/patient counts, overlap audit, id hashes
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from bamo.config import load_config
from bamo.data.ptbxl import (audit_partition, duplicate_groups, label_matrix, load_metadata, load_signals,
                             partition_rows, patient_fold_audit, task_label_lists)
from bamo.utils import sha256_file, stable_hash, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="superclass")
    args = ap.parse_args()
    cfg = load_config(task=args.task)
    classes = tuple(cfg["data"]["classes"])
    raw_root = Path(args.raw_root or cfg["data"]["raw_root"])
    out = Path(args.out or cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df, scp = load_metadata(raw_root)
    label_column = "superclasses" if args.task == "superclass" else "labels"
    df[label_column] = task_label_lists(df, scp, args.task, classes)
    report = {"task": args.task, "classes": list(classes),
              "source_records": int(len(df)), "source_patients": int(df["patient_id"].nunique()), "exclusions": {}}

    no_label = df[label_column].apply(len) == 0
    report["exclusions"]["no_diagnostic_superclass" if args.task == "superclass" else "no_task_label"] = int(no_label.sum())
    df = df[~no_label]

    X, issues = load_signals(df, raw_root, cfg["data"]["sampling_rate"])
    bad = np.zeros(len(df), dtype=bool)
    for i in issues:
        bad[i] = True
    report["exclusions"]["signal_issues"] = {str(int(df.index[i])): v for i, v in issues.items()}
    df, X = df[~bad], X[~bad]

    dups = duplicate_groups(X)
    report["duplicate_waveform_groups"] = [[int(df.index[i]) for i in g] for g in dups]  # reported, not excluded

    Y = label_matrix(df[label_column].tolist(), classes)
    patient_ids = df["patient_id"].astype(int).to_numpy()
    folds = df["strat_fold"].astype(int).to_numpy()
    report["patient_fold_audit"] = patient_fold_audit(patient_ids, folds)
    report["final_records"] = int(len(df))
    report["final_patients"] = int(len(set(patient_ids)))
    report["class_counts_by_fold"] = {int(f): dict(zip(classes, Y[folds == f].sum(0).tolist())) for f in range(1, 11)}
    absent = {int(f): [c for c, n in zip(classes, Y[folds == f].sum(0)) if n == 0] for f in range(1, 11)}
    if any(absent.values()):
        raise SystemExit(f"a frozen class has no positives in a fold: { {f: a for f, a in absent.items() if a} }")
    report["label_cardinality"] = {int(k): int(v) for k, v in zip(*np.unique(Y.sum(1), return_counts=True))}

    records = df.reset_index()[["ecg_id", "patient_id", "strat_fold", "validated_by_human", "filename_lr"]].copy()
    records["patient_id"] = patient_ids
    records[label_column] = ["|".join(s) for s in df[label_column]]
    for j, c in enumerate(classes):
        records[f"y_{c}"] = Y[:, j]
    records.to_csv(out / "records.csv", index=False)
    np.save(out / "signals_100hz.npy", X)
    np.save(out / "labels.npy", Y)

    manifest = {}
    for name, part in cfg["partitions"].items():
        rows = partition_rows(folds, part)
        manifest[name] = {"folds": part, **audit_partition(patient_ids, rows),
                          "ecg_id_hash": {r: stable_hash(sorted(map(int, records["ecg_id"].to_numpy()[idx])))
                                          for r, idx in rows.items()}}
    report["array_sha256"] = {"signals_100hz.npy": sha256_file(out / "signals_100hz.npy"),
                              "labels.npy": sha256_file(out / "labels.npy")}
    write_json(out / "data_report.json", report)
    write_json(out / "split_manifest.json", manifest)

    print(f"{args.task} records: {report['source_records']} -> {report['final_records']}  "
          f"(no label: {no_label.sum()}, signal issues: {len(issues)})")
    print(f"patients in more than one fold: {report['patient_fold_audit']['patients_in_multiple_folds']}")
    for name, m in manifest.items():
        print(f"{name:7s} records {m['n_records']}  patient overlap ok: {m['ok']}")
    if not all(m["ok"] for m in manifest.values()):
        raise SystemExit("patient overlap detected between partition roles")


if __name__ == "__main__":
    main()
