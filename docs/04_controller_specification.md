# 04 — Controller specification

The formal definition of method P, implemented in [controller.py](../src/bamo/search/controller.py) (`MarginalUtilityPolicy` = decision rule, `MarginalUtilitySearch` = execution). Proposal §3.3–3.4 is authoritative. Where it left room for interpretation, the decision is cited (`Dxx`).

## 4.1 Notation

| Symbol | Meaning | Default |
|---|---|---|
| K | blocks {A, B, C, J} | — |
| t | number of completed allocation rounds (global) | — |
| M*_t | best search-validation macro AUROC (incumbent) after round t | — |
| ΔT_t | wall time of round t, including decision overhead since the previous round ended (D18) | measured |
| T_ref | median single-fidelity evaluation duration on the frozen GPU (pilot P2) | FREEZE |
| Δc_t | ΔT_t / T_ref (dimensionless cost) | — |
| r_t | max(0, M*_t − M*_{t−1}) / max(Δc_t, ε) | — |
| W | per-block rolling window of rewards (D15) | 8 rounds |
| n_k | number of rounds of block k inside its window | ≤ 8 |
| κ | exploration coefficient | 1 (sweep 0.5, 1, 2) |
| σ_floor | minimum dispersion | 1e-4 |
| τ | utility threshold | 5e-4 (sweep 1e-4, 5e-4, 1e-3) |
| h | hysteresis, 0.1 τ | 5e-5 |
| G | probe gap (decisions without allocation that trigger a mandatory probe) | 8 |

**Units.** r and τ are in *AUROC per T_ref of compute*. τ = 5e-4 means: stop when even the optimistic estimate says one more evaluation's worth of compute buys less than 0.0005 macro AUROC. For scale, the non-inferiority margin of 0.005 equals ten such evaluation-equivalents at τ.

## 4.2 Reward (proposal §3.3.2)

```
ΔM_t = max(0, M*_t − M*_{t−1})        # improvement of the global incumbent (D17)
Δc_t = ΔT_t / T_ref                   # or n_evals in the round when cost_normalization = false (D23)
r_t  = ΔM_t / max(Δc_t, ε),  ε = 1e-9
```
* A trial that beats its DE target but not the global incumbent earns zero reward.
* If no valid incumbent exists yet, ΔM_t = 0.
* Rounds whose trials are infeasible or failed still cost time and earn no reward.

## 4.3 Block priority score (proposal §3.3.3)

```
U_k(t) = mean(r_k) + κ · σ_k · sqrt( 2 ln(1 + t) / (1 + n_k) )
σ_k    = max( SD(r_k), σ_floor )       # SD with ddof = 1; for n_k < 2, SD = 0 (D16)
U_k    = +∞ if block k has no rounds yet
```
This is a bandit-inspired optimistic score, not a confidence bound with guaranteed coverage. Rewards are non-stationary and adaptively sampled.

## 4.4 Action rule (evaluated before each round)

In priority order:
1. **Warm-up** (until every block has 3 rounds): interleaved order A, B, C, J, A, B, C, J, … (D14).
2. **Round-robin** (only when `adaptive = false`): cycle A → B → C → J.
3. **Probe** (if `mandatory_probes`): any block with no allocation in the last G decisions (t − last_k > G). The longest-unvisited block is chosen first.
4. **Continue** the current block if U_current ≥ max_k U_k − h.
5. Otherwise **switch** to argmax_k U_k (ties → block order).

## 4.5 A round

A round evaluates `trials_per_round = 2` DE trials sequentially in the chosen block:
* the target index cycles through the population (D13)
* each trial is DE/rand/1/bin on the block's coordinates only
* other coordinates are inherited from the target
* bounce-back at the bounds (D12)
* duplicate repair (D11)
* greedy replacement (ties go to the trial)

Block J varies all nine coordinates.

## 4.6 Termination state machine

```
            ┌───────────── search ◄─────────────────────────────┐
            │  streak ≥ 3  ∧  every block ≥ 3 rounds            │
            │  ∧ elapsed ≥ 0.4·B                                │
            ▼                                                   │
     stop proposed ──(global_termination = false)──► record shadow stop (D24), continue
            │
            ├─(confirmation = false)──► STOP  "utility_stop_unconfirmed"
            ├─(incumbent flagged unstable, unchanged)──► continue (D21)
            ├─(not enough time for confirmation)──► continue; ends as "budget" (D22)
            ▼
     confirmation:
       for k in A, B, C, J:  one round in k  →  if r ≥ τ: resume ──────────────────┘
       re-evaluate incumbent with 2 fresh training seeds (max fidelity)
       stable = max_i |m_i − M*| ≤ tolerance (frozen after pilot P3)
       if not stable: flag incumbent unstable; resume ──────────────────────────────┘
       else STOP "utility_stop_confirmed"

  at any time: cap reached → STOP "budget" (or "budget_during_warmup")
```
* **streak** counts consecutive post-warm-up decisions after which *all* U_k < τ (D19). It resets after a failed confirmation.
* **Confirmation feasibility:** remaining ≥ |K| × median round time + 2 × median max-fidelity evaluation time.
* **Confirmation probe rounds** update block statistics like normal rounds, but do not count toward the streak (D20).
* **All confirmation costs are charged to B.** Re-evaluations never change the incumbent: they use other seeds, and the incumbent is defined on the run's training seed.

## 4.7 Configuration selection

The configuration returned is the incumbent: the best valid search-validation result at maximum fidelity with the run's training seed. Selection uses search results only. The final model is trained separately (doc 02 §2.10).

## 4.8 Ablation switches

| Variant | Settings changed | Isolates |
|---|---|---|
| `mu_nostop` | global_termination = false | allocation gains without stopping savings (H1) |
| `rr_nostop` | adaptive = false, global_termination = false | adaptive vs fixed allocation (same blocks, DE, evaluator) |
| `mu_nocost` | cost_normalization = false (Δc = evaluations per round) | value of measured-cost information |
| `mu_noprobe` | mandatory_probes = false | abandonment of temporarily unproductive blocks (H3) |
| `mu_nojoint` | blocks = [A, B, C] | cross-block interactions (H3) |
| `mu_noconfirm` | confirmation = false | premature stopping vs safeguard cost |

## 4.9 Worked example (illustrative, not results)

T_ref = 40 s. After warm-up (t = 12), block C's window holds r = {0, 0.0012, 0}. Its mean is 0.0004, SD 0.00069 → σ_C = 0.00069, and the bonus is 1 × 0.00069 × sqrt(2 ln 13 / 4) = 0.00078, so U_C = 0.00118. Blocks A, B and J hold only zeros, so U = σ_floor × 1.13 = 0.000113. C has the highest score, and the controller continues or switches to C.

Suppose C then gives three more zero-reward rounds: its window becomes {0, 0.0012, 0, 0, 0, 0}, mean 0.0002, SD 0.00049, and the bonus shrinks with n_C = 6 (t = 15) to 0.00049 × sqrt(2 ln 16 / 7) = 0.00049 × 0.89 = 0.00044. So U_C ≈ 0.00064 > τ, and no stop yet. After a few more zeros, all U_k < τ, and three such decisions in a row propose a stop.

## 4.10 Decision log (`decisions.jsonl`)

One line per round or controller event:

| Field | Meaning |
|---|---|
| `round` | t after the event |
| `elapsed_s` | clock time |
| `phase` | warmup, search, confirm or stop |
| `action` | warmup, round_robin, probe, continue, switch, confirm_probe, reevaluate, propose_stop, confirm_stop, stop_unconfirmed, confirmation_infeasible |
| `scores` | U_k before the decision (null = ∞) |
| `block`, `reward`, `delta_m`, `round_s`, `delta_c` | round outcome |
| `improved_targets`, `duplicates_repaired` | DE bookkeeping |
| `incumbent`, `streak` | state after the round |
| `reeval` | values, max deviation, tolerance, stable (re-evaluation events only) |

`summary.json → method_extra` adds: settings, rounds per block, action counts, shadow stop, every confirmation attempt, and when confirmation became infeasible. These feed the controller-behaviour outcomes (doc 07 §7.6).

## 4.11 Verification status

* Unit tests (`tests/test_controller.py`) cover:
  * reward normalization
  * the proposal's §3.3.4 example
  * interleaved warm-up
  * the score formula with σ_floor
  * hysteresis
  * mandatory probes
  * the stop-proposal conditions
  * round-robin
* End-to-end synthetic runs (`tests/test_methods_synthetic.py`) cover:
  * all variants respect the cap
  * logs are complete
  * determinism
  * refusal to run with an unfrozen re-evaluation tolerance
* Still to do on GPU: a decision-log review of one real pilot run (WP4).
