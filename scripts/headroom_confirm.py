"""Confirm the headroom verdict with repeats (doc 14, R1 follow-up).

    python scripts/headroom_confirm.py [--out-root runs/local_dev] [--extra-seeds 3]

Reads <out-root>/pilot/headroom.json, retrains six of its settings with extra seeds:
ranks 1-3 (best), ranks 6-7 (around Q75) and rank 13 (median). Each setting then has
1 + extra_seeds runs. Compares mean(best) - mean(Q75 settings) with the pooled seed SD
(same thresholds: >= 3x go, <= 1.5x redesign). Pilot data only (folds 1-6 -> 7).
"""

import _paths  # noqa: F401

import argparse
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bamo.config import load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.hardware import fingerprint
from bamo.training.trainer import train_model
from bamo.utils import derive_seed, read_json, write_json

GROUPS = {"best": [1, 2, 3], "q75": [6, 7], "median": [13]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="runs/local_dev")
    ap.add_argument("--extra-seeds", type=int, default=3)
    ap.add_argument("--reuse-log", default=None, help="log of an interrupted run; finished trainings are reused")
    args = ap.parse_args()
    reused = {}
    if args.reuse_log and Path(args.reuse_log).exists():
        pattern = re.compile(r"\] \w+\s+rank\s+\d+ (\w+)\s+seed (\d+): AUROC ([0-9.]+)")
        for line in Path(args.reuse_log).read_text(encoding="utf-8").splitlines():
            m = pattern.search(line)
            if m:
                reused[(m.group(1), int(m.group(2)))] = float(m.group(3))
        print(f"reusing {len(reused)} finished trainings from {args.reuse_log}", flush=True)
    out = Path(args.out_root) / "pilot"
    first = read_json(out / "headroom.json")
    ranked = sorted(first["spread"], key=lambda r: -r["val_auroc"])
    cfg = load_config("pilot")
    data = load_prepared(cfg["data"]["processed_dir"])
    tensors, _ = build_tensors(data, cfg["partitions"]["pilot"], torch.device("cuda"))
    ev = cfg["evaluator"]["single_fidelity"]
    seeds = [derive_seed("headroom_confirm", i) for i in range(args.extra_seeds)]

    def train(config, seed):
        t0 = time.perf_counter()
        o = train_model(config, cfg["model"], tensors["train"], tensors["val"], max_epochs=ev["max_epochs"],
                        patience=ev["patience"], seed=seed, amp=cfg["training"]["amp"])
        return {"seed": seed, "val_auroc": o.best_score, "epochs_run": o.epochs_run, "seconds": time.perf_counter() - t0}

    train(cfg["search_space"]["reference_config"], 0)  # untimed GPU warm-up
    total = sum(len(v) for v in GROUPS.values()) * args.extra_seeds
    done, settings = 0, []
    for group, ranks in GROUPS.items():
        for rank in ranks:
            base = ranked[rank - 1]
            runs = [{"seed": base["seed"], "val_auroc": base["val_auroc"], "original": True}]
            for s in seeds:
                done += 1
                if (base["name"], s) in reused:
                    runs.append({"seed": s, "val_auroc": reused[(base["name"], s)], "reused_from_log": True})
                    continue
                r = train(base["config"], s)
                runs.append(r)
                print(f"[{done}/{total}] {group:6s} rank {rank:2d} {base['name']:9s} seed {s}: AUROC {r['val_auroc']:.4f} "
                      f"({r['seconds']:.0f}s)", flush=True)
            vals = [r["val_auroc"] for r in runs]
            settings.append({"group": group, "rank": rank, "name": base["name"], "config": base["config"], "runs": runs,
                             "mean": statistics.mean(vals), "sd": statistics.stdev(vals)})

    noise_sd = float(np.sqrt(np.mean([s["sd"] ** 2 for s in settings])))
    by = {g: [s for s in settings if s["group"] == g] for g in GROUPS}
    best_mean = max(s["mean"] for s in by["best"])
    q75_mean = statistics.mean(s["mean"] for s in by["q75"])
    median_mean = by["median"][0]["mean"]
    gap = best_mean - q75_mean
    ratio = gap / noise_sd
    verdict = "go" if ratio >= 3 else ("redesign" if ratio <= 1.5 else "borderline")
    summary = {"best_setting_mean": best_mean, "q75_settings_mean": q75_mean, "median_setting_mean": median_mean,
               "confirmed_top_gap": gap, "noise_sd_pooled_6_settings": noise_sd, "gap_over_noise": ratio,
               "verdict": verdict, "runs_per_setting": 1 + args.extra_seeds}
    write_json(out / "headroom_confirm.json", {"hardware": fingerprint(), "summary": summary, "settings": settings})

    print("\nCONFIRMATION SUMMARY")
    for s in settings:
        print(f"  {s['group']:6s} rank {s['rank']:2d} {s['name']:9s} first {s['runs'][0]['val_auroc']:.4f} -> "
              f"mean of {len(s['runs'])} {s['mean']:.4f} (SD {s['sd']:.4f})")
    print(f"noise SD pooled over 6 settings: {noise_sd:.4f}")
    print(f"confirmed top gap (best mean - Q75 mean): {gap:.4f} = {ratio:.1f}x noise | best vs median: {best_mean - median_mean:.4f}")
    print(f"VERDICT: {verdict.upper()}")


if __name__ == "__main__":
    main()
