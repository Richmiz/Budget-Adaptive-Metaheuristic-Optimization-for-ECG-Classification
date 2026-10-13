# 01 — Study plan

Six-month primary study, starting **Monday 12 October 2026**. It follows the proposal's roadmap (§3.8–3.9), broken down into work packages (WPs), weekly tasks, acceptance criteria and go/no-go gates. **R** = researcher, **C** = Claude-assisted (code, analysis scripts, drafting). Every scientific decision stays with R.

## 1.1 Work packages

### WP0 — Project infrastructure (W1) ✅ scaffolded 8 Oct 2026
| Task | Output | Acceptance | Who |
|---|---|---|---|
| Repository, package, configs, tests | `src/bamo`, `configs/`, `tests/` | 49 unit and synthetic tests pass | C ✅ |
| Synthetic dry run of the full matrix | `runs_synthetic/` | All in-house methods respect the cap; logs complete | C ✅ |
| Local environment (RTX 3050) | conda env `D:\envs\bamo` with CUDA PyTorch | `pytest` passes; `torch.cuda.is_available()` | C ✅ |
| Colab environment (VS Code + Colab extension, G4) | `notebooks/COLAB_G4_SETUP.ipynb` runs end to end | Drive mounted; data identical; tests pass on the G4 | R+C |
| Pin library versions | `requirements-lock.txt` | DEHB/Optuna/ConfigSpace versions recorded (D27) | C ✅ |

### WP1 — Evidence review and novelty assessment (W1–W4, then continuous)
| Task | Output | Acceptance |
|---|---|---|
| Structured search (doc 11) in Scopus, WoS, IEEE Xplore, arXiv | Search log, screening table | Strings and dates recorded; ≥ 2 passes |
| Full-text reading: Hyperband, BOHB, DEHB, HABO, AOS/bandit hyper-heuristics, cost-aware HPO, automatic HPO termination | Differentiation matrix | Every overlapping mechanism has a stated difference or a narrowed claim |
| Contribution map | `docs/11_novelty_assessment.md` §11.5 filled | Claims restricted to what the matrix supports |

### WP2 — Data preparation (W1–W3)
| Task | Output | Acceptance |
|---|---|---|
| Download PTB-XL 1.0.3, record SHA-256 | `data/raw/` | Hash logged ✅ |
| Label mapping, exclusions, quality checks | `data/processed/*`, `data_report.json` | Counts reported; no unexplained exclusions ✅ |
| Patient audit and split manifest | `split_manifest.json` | Zero patient overlap in every partition ✅ |
| Rebuild data on Colab | done by the notebook | Arrays byte-identical to local ✅ (verified locally from the bundle) |

### WP3 — Baseline model and evaluator (W2–W4)
| Task | Output | Acceptance |
|---|---|---|
| Reference CNN on the pilot partition | Training curve, val AUROC | Trains stably; macro AUROC in a plausible range for PTB-XL superclasses ✅ (local P0) |
| Throughput optimization (GPU-resident data, AMP) | Seconds/epoch per GPU | Evaluation time dominated by GPU compute |
| Evaluator interruptibility and failure handling | Tests on GPU | Cap interrupts mid-epoch; OOM → `infeasible` |

### WP4 — Controller and comparator verification (W4–W6)
| Task | Output | Acceptance |
|---|---|---|
| Controller decision-rule tests | `tests/test_controller.py` ✅ | Matches doc 04 |
| TPE and DEHB adapters on synthetic and real data | Short runs | Cap respected; DEHB fidelity ladder 5/10/20 |
| BOHB check against the official hpbandster on a cheap objective (D26) | Comparison note | Comparable anytime curves, or switch to the official implementation |
| Decision-log review | Annotated example run | Every action traceable to scores |

### WP5 — Pilot calibration and protocol freeze (W5–W8)
See [06_pilot_calibration.md](06_pilot_calibration.md). Outputs: chosen Colab GPU type, T_ref, re-evaluation tolerance, τ/κ, final domains, seed count, a feasible budget plan, and `protocol/frozen_protocol.yaml`, git tag `protocol-frozen`.

### WP6 — Main benchmark (W9–W13): 210 runs
### WP7 — Ablation and failure analysis (W14–W17): 60 runs + shadow-stop final models
### WP8 — Final training and confirmatory evaluation (W18–W21)
Final models → analysis lock → **one** fold-10 evaluation → primary and secondary analyses.
### WP9 — Synthesis and dissemination (W22–W26)
Manuscript, code release, reproducibility package.

## 1.2 Weekly timeline

| Wk | Dates (2026–27) | Phase | Key tasks | Milestone |
|---|---|---|---|---|
| W1 | 12–18 Oct | M1 Evidence & data | Environments (local + Colab); download PTB-XL; start novelty search | — |
| W2 | 19–25 Oct | M1 | `prepare_data.py`; audit report; Drive copy | **M1a** dataset frozen |
| W3 | 26 Oct–1 Nov | M1 | Reference CNN on pilot partition; throughput tuning | — |
| W4 | 2–8 Nov | M1 | Evaluator GPU tests; novelty matrix v1 | **G1** data & baseline gate |
| W5 | 9–15 Nov | M2 Controller & pilot | P1 feasibility on candidate Colab GPUs; choose GPU type | — |
| W6 | 16–22 Nov | M2 | P2 T_ref (2 sessions); P3 noise; comparator verification | — |
| W7 | 23–29 Nov | M2 | P4 τ/κ sweep; P5 budget feasibility; seed-count check | — |
| W8 | 30 Nov–6 Dec | M2 | Fill FREEZE values; `lock_protocol.py freeze`; tag | **G2** protocol frozen |
| W9–W13 | 7 Dec–10 Jan | M3 Benchmark | 210 main runs (sharded); weekly progress review (validation only) | **M3** all main logs complete |
| W14–W16 | 11–31 Jan | M4 Ablation | 60 ablation runs; decision-log analysis; overhead audit | — |
| W17 | 1–7 Feb | M4 | Premature-termination (shadow) analysis on validation data | **G3** search phase closed |
| W18–W19 | 8–21 Feb | M5 Final & confirmatory | Final training of all incumbents + reference + shadow models; deployment profiling | — |
| W20 | 22–28 Feb | M5 | `lock_protocol.py analysis`; `evaluate_test.py` once | **G4** analysis locked |
| W21 | 1–7 Mar | M5 | Primary and secondary analyses; figures | **M5** results fixed |
| W22–W24 | 8–28 Mar | M6 Synthesis | Manuscript draft; internal review | — |
| W25–W26 | 29 Mar–11 Apr | M6 | Code and reproducibility release; submission | **M6** submitted |

Slack: W11–W12 (holidays) are assumed to run at ~50% capacity. If compute falls behind, the order of priority is: the primary contrast (P and DEHB at B_mid), then the rest of B_mid, then B_low and B_high, then the ablations (keeping `rr_nostop` and `mu_nostop` first, as the proposal says).

## 1.3 Gates (go / no-go)

| Gate | When | Pass criteria | If it fails |
|---|---|---|---|
| **G1** Data & baseline | End W4 | Zero patient overlap; exclusions explained; reference CNN trains stably; evaluator interrupts at cap | Fix before any pilot timing |
| **G2** Protocol freeze | End W8 | GPU type fixed; T_ref measured in ≥ 2 sessions with ≤ 10% drift; warm-up + confirmation fit in B_low; projected GPU-hours ≤ available (doc 10); comparators verified; all FREEZE fields filled | Apply the mitigation ladder in doc 10 §10.4; never start the benchmark with null FREEZE fields |
| **G3** Search closed | End W17 | Every scheduled job has `summary.json` or a documented failure; no config hash mismatch | Rerun failed jobs (D39); document the rest |
| **G4** Analysis lock | W20 | All final models trained; analysis plan unchanged since freeze, or amendments logged | No test access until it passes |

## 1.4 Compute budget (summary; details in doc 10)

**Updated 13 Oct 2026 with the measured T_ref = 9.79 s (Colab G4).** Estimates, not guaranteed bounds (drift check pending; reruns add time); methods that stop early use less:

| Block | T_ref units | G4 hours |
|---|---|---|
| Main benchmark (210 runs) | 39,200 | 107 |
| Ablations (60 runs) | 9,600 | 26 |
| Stop-rule checks (30 runs, D43–D44) | ≤ 4,800 | ≤ 13 |
| Final training (~300 models) | ~730 | ~2 |
| **Total remaining** | **≤ 54,330** | **≤ 148 h ≈ 1,320 CU ≈ $130** |

The pilot itself is done (2 h 30 min on G4). The pre-pilot planning table (T_ref 20–60 s → 285–855 GPU-h) is superseded.

## 1.5 Risk register

| # | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|---|
| R1 | Colab GPU type unavailable or varies between sessions | High | High: budgets are wall-clock | Runner GPU guard; shard the work so any session on the right GPU can take it; choose a commonly available type at G2 |
| R2 | Session disconnects mid-run | High | Low: one job lost | Jobs ≤ a few hours; summary-based resume; restart-from-scratch policy (D39) |
| R3 | Timing noise (shared VM, thermal, Drive I/O) distorts rewards | Medium | Medium | GPU-resident data; local-disk data copy; T_ref drift check; randomized job order; log per-evaluation durations |
| R4 | Compute exceeds plan | Medium | High | Mitigation ladder (doc 10 §10.4); priority order in §1.2 |
| R5 | Controller never stops (τ too low) or always stops early (τ too high) | Medium | Medium | P4 sweep with a pre-specified selection rule; ablations quantify it |
| R6 | DEHB/BOHB implementation differences bias comparisons | Medium | High (primary contrast) | Official DEHB package pinned; BOHB checked against hpbandster (D26); report fidelity differences as method characteristics |
| R7 | Novelty overlap (HABO, adaptive operator selection, cost-aware HPO) | Medium | High | WP1 matrix; claims limited to the specific controller and evidence |
| R8 | Label mapping or leakage error | Low | High | Benchmark mapping (D01); patient audit; fold-10 lock |
| R9 | Noise in single-seed fitness makes stopping erratic | Medium | Medium | Confirmation re-evaluation; P3 noise estimate; D21 rule |
| R10 | Ten seeds too few for non-inferiority precision | Medium | Medium | Seed planning from pilot variance (doc 07 §7.8) before the freeze |
| R11 | Windows/Colab environment drift (library versions) | Medium | Low | Lock file; record versions in every summary |
