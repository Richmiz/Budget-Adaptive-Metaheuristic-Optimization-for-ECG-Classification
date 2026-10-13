# 05 — Comparators and fairness

All methods run through the same `SearchTracker` (same cap, clock, caching rules, incumbent rule and logs), on the same GPU type, the same data partition and the same training seed per search seed.

## 5.1 Methods

| ID | Method | Implementation | Fidelity | Shared initial design (D10) | Key settings |
|---|---|---|---|---|---|
| Ref | Untuned CNN | `reference_config`, final training only | — | — | D05 |
| B1 | Random search | in-house [simple.py](../src/bamo/search/methods/simple.py) | single (20 ep) | first 8 evaluations | uniform in encoded space |
| B2 | TPE | Optuna `TPESampler` via ask/tell [library.py](../src/bamo/search/methods/library.py) | single | enqueued as the 8 start-up trials | n_startup_trials = 8, univariate (D28) |
| B3 | Standard DE | in-house, same engine as P | single | initial population | DE/rand/1/bin, NP = 8, F = 0.5, CR = 0.7, all 9 coordinates |
| B4 | Hyperband | in-house [hyperband.py](../src/bamo/search/methods/hyperband.py) | 5 / 10 / 20, η = 2 | evaluated at 20 ep; incumbent candidates | brackets s = 2, 1, 0 cycled until the cap |
| B5 | BOHB | in-house reimplementation (D26) | 5 / 10 / 20, η = 2 | added as 20-epoch observations to the KDE model | top 15%, 64 samples, random fraction 1/3, bandwidth factor 3, min bandwidth 1e-3 |
| B6 | DEHB | official `dehb` package via ask/tell (D27) | 5 / 10 / 20, η = 2 | evaluated first; **not** ingested by DEHB | package defaults otherwise |
| P | DE + marginal-utility controller | in-house [controller.py](../src/bamo/search/controller.py) | single | initial population | doc 04 |

Hyperband bracket structure with fidelities 5/10/20 and η = 2:

| Bracket s | Rung 1 | Rung 2 | Rung 3 |
|---|---|---|---|
| 2 | 4 configs × 5 ep | 2 × 10 ep | 1 × 20 ep |
| 1 | 3 × 10 ep | 1 × 20 ep | — |
| 0 | 3 × 20 ep | — | — |

## 5.2 Matched across methods

* dataset, labels, partitions
* hardware type
* loss and optimizer
* full search domain
* maximum fidelity (20 epochs)
* early-stopping patience (5)
* time caps
* incumbent rule (20-epoch results only)
* final-training procedure
* shared initial design and its cost

## 5.3 Method characteristics, reported rather than equalized

* **Multi-fidelity methods** spend part of the budget at 5 or 10 epochs. This is their mechanism, not an unfairness.
* **Model-fitting time** of TPE, BOHB and DEHB is charged as overhead, like the controller's own overhead.
* **DEHB cannot ingest external initial configurations.** It pays for the shared design (fair cost) but only gains a potential incumbent from it. This asymmetry is reported and is conservative toward DEHB's anytime curve.
* **TPE** uses its native distributions (log-float, categorical). The DE family and BOHB operate in the encoded [0, 1]⁹ space.

## 5.4 Verification before G2

| Check | Method | Pass |
|---|---|---|
| Cap respected; logs complete | all methods, synthetic objective | ✅ in-house (tests); TPE/DEHB after install |
| DEHB fidelity values are 5/10/20 and integer epochs are used | DEHB, short real run | all `fidelity` values in the log are in {5, 10, 20} |
| DEHB ask/tell API matches the pinned version | DEHB | adapter runs without modification, or the adapter is updated and noted |
| BOHB behaviour vs official hpbandster | both on the synthetic objective, 10 seeds, equal budget | similar median anytime curves; if not, switch B5 to hpbandster wrapped in the tracker |
| TPE enqueue order | TPE | the first 8 logged configurations equal the shared design |
| Overhead plausibility | all | overhead_s / elapsed_s reported; no method above 5% without explanation |

## 5.5 Interpretation guardrails

* The **primary** comparison is P vs DEHB at B_mid (both evolutionary). Comparisons with Hyperband and BOHB are secondary. Comparisons with random search and TPE are context.
* "P vs DEHB" compares two *complete systems* (single-fidelity + controller vs multi-fidelity DE). Which component causes a difference is answered by the ablations, which share the evaluator and the DE engine.
* Portability to PSO or GA is a later extension (proposal §3.8.2), not part of this comparison.
