# 11 — Novelty assessment protocol

The proposal is explicit (§2.1, §2.3): adaptive allocation and early stopping are established, and a new algorithm name does not establish originality. The contribution has to be located precisely and backed by the controlled evidence. This document plans that assessment (WP1).

## 11.1 Claim under test

> An auditable continue / switch / stop controller that allocates DE search time across **hyperparameter blocks** using **observed validation improvement per measured compute cost**, with mandatory exploration, a joint block, and confirmed termination, evaluated with patient-separated ECG benchmarking under matched time caps.

## 11.2 Search protocol

| Element | Plan |
|---|---|
| Databases | Scopus, Web of Science, IEEE Xplore, ACM DL, arXiv (cs.LG, cs.NE), plus the AutoML Conf, GECCO, NeurIPS/ICML/ICLR proceedings |
| Period | No lower bound for the mechanism searches; 2015– for ECG-specific searches |
| Concept groups | (1) hyperparameter optimization / AutoML; (2) adaptive allocation / multi-armed bandit / credit assignment; (3) cost-aware / budget / anytime / termination; (4) subspace / block / coordinate / grouped hyperparameters; (5) DE / evolutionary / metaheuristic; (6) ECG / PTB-XL (applied searches only) |
| Example string | `("hyperparameter optimi*" OR AutoML) AND (bandit OR "adaptive allocation" OR "credit assignment") AND (subspace OR block OR coordinate OR group*)` |
| Example string | `("hyperparameter optimi*" OR "algorithm configuration") AND ("cost-aware" OR "per unit time" OR "per second" OR "budget-aware") AND (terminat* OR "early stopping of the search" OR "stopping criterion")` |
| Example string | `("differential evolution" OR evolutionary) AND ("adaptive operator selection" OR "hyper-heuristic") AND bandit` |
| Snowballing | Backward and forward citations of Hyperband, BOHB, DEHB, HABO and the closest hits |
| Logging | Date, database, string, hit count, screened, included (keep the export files) |

## 11.3 Mechanism areas and candidate references (verify each before citing)

The references marked *(verify)* come from memory and must be checked for existence, venue and content before any use.

| Area | Why it matters | Candidate starting points |
|---|---|---|
| Multi-fidelity bandit HPO | Allocation over candidates | Hyperband (Li et al., 2018) [R3]; BOHB (Falkner et al., 2018) [R4]; DEHB (Awad et al., 2021) [R5] |
| Bandits over hyperparameter subsets | Closest to "allocation across blocks" | HABO preprint (Karroum & Mazhar, 2025) [R6] |
| Adaptive operator selection in EAs | Bandit credit assignment inside DE/EAs | Fialho et al., dynamic multi-armed bandits for AOS, GECCO 2008 *(verify)* |
| Non-stationary bandits | Justifies windowed statistics | Garivier & Moulines, sliding-window UCB, ALT 2011 *(verify)*; Audibert et al., UCB-V, TCS 2009 *(verify)* |
| Cost-aware Bayesian optimization | "Improvement per second" | Snoek et al., "expected improvement per second", NeurIPS 2012 *(verify)* |
| Automatic termination of HPO | Competes with the stopping rule | Makarova et al., "Automatic termination for hyperparameter optimization", AutoML Conf 2022 *(verify)* |
| Learning-curve / freeze-thaw allocation | Allocation by expected progress | Swersky et al., freeze-thaw BO, 2014 *(verify)* |
| Hyperparameter importance | Block structure justification | fANOVA-type importance analyses *(verify)* |
| Metaheuristics for ECG deep learning | Application context | Source SLR [S1] and its included studies |

## 11.4 Screening questions for each close paper

1. Does it allocate search effort across **groups of hyperparameters** (not just across candidates or fidelities)?
2. Is the allocation signal **improvement per measured cost** (not per evaluation)?
3. Does it include a **termination** decision based on that signal, with a **confirmation** safeguard?
4. Does it address **cross-group interactions** (a joint arm)?
5. Is it evaluated under **matched wall-clock caps** with search, training and inference costs separated?
6. Is it evaluated on **biomedical signals** with **patient-level** separation?

## 11.5 Differentiation matrix (fill during WP1)

| Paper | Q1 groups | Q2 per-cost | Q3 stop + confirm | Q4 joint | Q5 matched caps | Q6 patient-separated | Our difference / narrowed claim |
|---|---|---|---|---|---|---|---|
| Hyperband | ✗ (candidates) | ✗ | ✗ | — | ✓ | ✗ | |
| BOHB | ✗ | ✗ | ✗ | — | ✓ | ✗ | |
| DEHB | ✗ | ✗ | ✗ | — | ✓ | ✗ | |
| HABO | ? | ? | ? | ? | ? | ✗ | |
| … | | | | | | | |

The ✓/✗ values for Hyperband, BOHB and DEHB are first impressions from the proposal's summary. Confirm them from the full texts.

## 11.6 Outcome rule

If a prior method satisfies Q1–Q3 together, the novelty claim is narrowed to the combination that remains (for example, the confirmation safeguard and the patient-separated, compute-matched evidence). The study is still valuable as a controlled evaluation. The Introduction and Related Work wording follows this matrix, not the proposal draft.
