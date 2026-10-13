"""Headroom check (doc 14, R1): does tuning move validation AUROC by more than seed noise does?

    python scripts/headroom_check.py [--out-root runs/local_dev] [--n-configs 24] [--n-seeds 5]

Pilot data only (train folds 1-6, validate on fold 7). Two parts:
  A. spread: n random (Latin hypercube) configurations + the reference, one training seed each;
  B. noise: the reference and one random configuration retrained with n_seeds seeds in total.
Verdict compares the top-end gap (best - 75th percentile) with the seed SD:
  >= 3x -> go;  <= 1.5x -> redesign;  otherwise borderline.
(For pure noise, best - Q75 of ~25 draws is about 1.3 SD, so 1.5x is "indistinguishable from noise".)
"""

import _paths  # noqa: F401

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bamo.config import load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.hardware import fingerprint
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.training.trainer import train_model
from bamo.utils import derive_seed, write_json


def short(c: dict) -> str:
    return (f"lr {c['learning_rate']:.1e} bs{c['batch_size']} wd {c['weight_decay']:.0e} do {c['dropout']:.2f} "
            f"w{c['width_multiplier']} b{c['blocks_per_group']} k{c['kernel_size']} d{c['dilation']} s{c['stem_stride']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="runs/local_dev")
    ap.add_argument("--n-configs", type=int, default=24)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--design-seed", type=int, default=9300)
    args = ap.parse_args()
    cfg = load_config("pilot")
    out = Path(args.out_root) / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    space = SearchSpace.from_config(cfg["search_space"])
    data = load_prepared(cfg["data"]["processed_dir"])
    tensors, _ = build_tensors(data, cfg["partitions"]["pilot"], torch.device("cuda"))
    ev = cfg["evaluator"]["single_fidelity"]

    def train(config, seed):
        t0 = time.perf_counter()
        o = train_model(config, cfg["model"], tensors["train"], tensors["val"], max_epochs=ev["max_epochs"],
                        patience=ev["patience"], seed=seed, amp=cfg["training"]["amp"])
        return {"config": config, "seed": seed, "val_auroc": o.best_score, "status": o.status,
                "epochs_run": o.epochs_run, "best_epoch": o.best_epoch, "seconds": time.perf_counter() - t0}

    train(cfg["search_space"]["reference_config"], 0)  # untimed GPU warm-up (result discarded)
    seeds = [derive_seed("headroom", i) for i in range(args.n_seeds)]
    configs = [("reference", cfg["search_space"]["reference_config"])]
    X = latin_hypercube(args.n_configs, space.dim, np.random.default_rng(args.design_seed))
    configs += [(f"lhs{i:02d}", space.decode(x)) for i, x in enumerate(X)]
    total = len(configs) + 2 * (args.n_seeds - 1)
    done = 0

    spread = []
    for name, c in configs:
        r = train(c, seeds[0])
        r["name"] = name
        spread.append(r)
        done += 1
        print(f"[{done}/{total}] spread {name:9s} AUROC {r['val_auroc']:.4f}  {r['epochs_run']:2d} ep  "
              f"{r['seconds']:5.0f}s  {short(c)}", flush=True)

    noise = {}
    for name, c in (configs[0], configs[1]):
        runs = [next(r for r in spread if r["name"] == name)]
        for s in seeds[1:]:
            r = train(c, s)
            r["name"] = name
            runs.append(r)
            done += 1
            print(f"[{done}/{total}] noise  {name:9s} seed {s}: AUROC {r['val_auroc']:.4f}", flush=True)
        noise[name] = runs

    vals = np.array([r["val_auroc"] for r in spread if np.isfinite(r["val_auroc"])])
    sds = [statistics.stdev([r["val_auroc"] for r in runs]) for runs in noise.values()]
    noise_sd = float(np.sqrt(np.mean(np.square(sds))))
    best, q75, median, worst = float(vals.max()), float(np.quantile(vals, 0.75)), float(np.median(vals)), float(vals.min())
    top_gap = best - q75
    between_sd = float(np.sqrt(max(vals.var(ddof=1) - noise_sd ** 2, 0.0)))
    ratio = top_gap / noise_sd if noise_sd > 0 else float("inf")
    verdict = "go" if ratio >= 3 else ("redesign" if ratio <= 1.5 else "borderline")
    summary = {"n_configs": len(vals), "best": best, "q75": q75, "median": median, "worst": worst,
               "top_gap_best_minus_q75": top_gap, "best_minus_median": best - median,
               "seed_sd_per_config": dict(zip(noise, sds)), "noise_sd_pooled": noise_sd,
               "between_config_sd_noise_corrected": between_sd, "top_gap_over_noise": ratio, "verdict": verdict}
    write_json(out / "headroom.json", {"hardware": fingerprint(), "summary": summary, "spread": spread, "noise": noise})

    print("\nHEADROOM SUMMARY")
    print(f"validation macro AUROC over {len(vals)} settings: best {best:.4f} | Q75 {q75:.4f} | median {median:.4f} | worst {worst:.4f}")
    print(f"seed noise SD: {noise_sd:.4f} (per config: " + ", ".join(f"{k} {v:.4f}" for k, v in zip(noise, sds)) + ")")
    print(f"top-end gap (best - Q75): {top_gap:.4f} = {ratio:.1f}x noise SD | noise-corrected spread SD {between_sd:.4f}")
    print(f"VERDICT: {verdict.upper()}")


if __name__ == "__main__":
    main()
