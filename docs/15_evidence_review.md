# 15 — Evidence review: is Article 1 worth pursuing? (9 Oct 2026)

Requested by the researcher: verify the earlier recommendation (doc 14) with evidence rather than opinion.

**Sources:**
* scite literature search (peer-reviewed papers and arXiv records, with citation counts)
* the official PTB-XL benchmark repository
* arXiv abstract pages
* **our own experiments** on PTB-XL (docs 13, 14)

Each claim below is marked **Verified**, **Partly verified** or **Not verified**.

## 15.1 Claim-by-claim verification

| # | Claim made earlier | Evidence | Verdict |
|---|---|---|---|
| 1 | On PTB-XL diagnostic superclasses, strong models score within ~0.01 macro AUROC of each other | Official benchmark leaderboard: six deep models range **0.921–0.930** (spread 0.009) (Strodthoff et al., 2021) | **Verified** |
| 2 | Tuning headroom on our task and search space is small | **Our data** (fold 7, 4–5 seeds per setting): untuned reference **0.9351** (mean of 5) vs best tuned setting **0.9387** (mean of 4), a +0.0036 gain; seed SD 0.0016; best − top-quarter gap 0.0032 = 2.0 × SD (doc 13) | **Verified** (our measurement, one task, 25 settings) |
| 3 | Defaults are often nearly as good as tuning | Probst et al. (2018): tunability measured on 38 datasets; Weerts et al. (2020), 59 datasets: leaving some hyperparameters at defaults is **non-inferior**, and sometimes beats limited random search; Sipper (2022), 26 algorithms × 250 datasets, 28.9 M runs: "we should not expect considerable gains from hyperparameter tuning on average" | **Partly verified.** Strong for classical ML (SVM, RF, …). **No equivalent study found for deep ECG models**, so our measurement (row 2) is the direct evidence |
| 4 | Harder PTB-XL tasks may offer more room | Benchmark leaderboard (same six deep models): superclasses **0.009** spread; all statements **0.018**; rhythm **0.026**; **form 0.048** (0.851–0.899) | **Partly verified.** The larger spread is *between architectures*; headroom from *hyperparameters* on those tasks is untested |
| 5 | Searches on this task are cheap | **Our G4 measurement:** ~14 s per evaluation, so a 160-evaluation search ≈ 37 min. For contrast, a PSO-tuned CNN for MIT-BIH arrhythmia reports **17.68 h** of search (Santander-Baños et al., 2023) | **Verified** for our setup. Cost depends heavily on model and data |
| 6 | The controller's core idea already exists | **Allocating effort across variable groups by contribution instead of round-robin:** Omidvar et al. (2011), contribution-based cooperative coevolution, which is exactly our "adaptive vs round-robin" contrast; extended in IEEE TEVC by Jia et al. (2019). **A bandit choosing which variable group to evolve next from recent progress:** De Rainville et al. (2013). **Bandit-based adaptive operator selection from fitness gains:** Fialho et al. (2009). **Bandits over hyperparameters in HPO:** Karroum & Mazhar (2025, preprint; arms = individual hyperparameters). **Improvement per unit time:** "expected improvement per second" (Snoek et al., 2012). **Automatic stopping of HPO:** Makarova et al. (2022); a dynamic early-stopping rule for random search (Florea & Andonie, 2018) | **Verified.** Each component of the controller has published precedent; only the *combination* applied to ECG HPO appears untested |
| 7 | ECG metaheuristic-tuning papers rarely report gain-for-cost | Your SLR [S1]: of 44 studies, 10 favourable, 15 mixed, **19 uncertain** trade-offs. Spot check of two 2023 ECG papers: one reports accuracy only (He et al., 2023); the other reports 17.68 h of search but, in its abstract, no comparison with an untuned model (Santander-Baños et al., 2023) | **Partly verified.** The SLR is the evidence; the spot check is illustrative only (abstracts, n = 2) |
| 8 | Nobody has measured tuning gain vs cost for deep ECG models on PTB-XL | Targeted search (514 matches, top 6 inspected): benchmark, self-supervised and architecture papers; none measured gain over defaults with compute accounting | **Not verified.** It suggests a gap, but a systematic search is needed before claiming it |

## 15.2 What the evidence implies for Article 1 as proposed

1. **Novelty.** Reviewers can point to Omidvar et al. (2011), De Rainville et al. (2013), Snoek et al. (2012) and Makarova et al. (2022) for every part of the controller. The paper could only claim the *integration* and its *application* to ECG HPO, which the proposal itself anticipated (§2.1, "the contribution must be demonstrated through the specific controller design and controlled experiments").
2. **Measurable effect.** On the proposed task (superclasses), our data show at most ~0.004 AUROC to gain over an untuned default, and only ~0.003 separating the best settings from ordinary good ones (rows 1–2). Final-accuracy comparisons between search methods will mostly be ties.
3. **Practical value of saving search time.** A full search costs ~37 GPU-minutes on G4 (row 5), so even a large relative saving is small in absolute terms on this task.
4. **One supportive point.** Cooperative-coevolution work shows adaptive allocation helps when variable groups contribute *unequally* (Omidvar et al., 2011). Our data show that kind of imbalance (learning rate dominates: lr < 1e-4 averages 0.921 vs 0.936), so the mechanism has something to exploit. The open question is whether it beats simpler baselines by a margin worth reporting.

## 15.3 Evidence-backed options

| Option | Supported by | Main risk |
|---|---|---|
| **A. Article 1 as proposed** (controller on PTB-XL superclasses) | Row 6 (mechanism plausible) | Rows 1, 2, 5, 6: small effect, cheap search, weak novelty |
| **B. Reframe Article 1 as a cost–benefit ("tunability") study of metaheuristic HPO for deep ECG classification**, across PTB-XL tasks of different difficulty (superclasses, rhythm, form): default vs random vs DE vs DEHB vs the controller, reporting gain over default, gain per GPU-hour and time-to-target | Rows 3, 4, 7, 8: tunability studies exist for classical ML but none was found for deep ECG; the SLR shows cost/gain is rarely clear; task difficulty varies 5× on the leaderboard | Row 8 is unverified (needs a systematic search); hyperparameter headroom on harder tasks is untested |
| C. Switch to Article 2 (robust selection) | Not evaluated in this review | Needs its own evidence check |

**Recommendation, based on the rows above:** Option **B**. It keeps the controller as one tested candidate (benchmarked honestly against a round-robin, CBCC-style allocation and a simple stopping rule), reuses everything built, and answers the question your SLR left open. This depends on two checks:
1. **Headroom on the PTB-XL form task**, our harder-task test of row 4: ~45 min on the local GPU, free.
2. **A systematic search confirming row 8** (Scopus/WoS/IEEE, per doc 11): a few days.

If (1) shows no more headroom than the superclass task and (2) finds existing tunability studies for deep ECG, option B weakens too, and the honest conclusion would be to prioritize Article 2.

## 15.4 Limits of this review

* The literature search was targeted, not systematic. Abstracts were read for most items; full texts were not.
* The tunability evidence (row 3) comes from classical ML on OpenML tabular data, not deep networks on ECG.
* Our headroom data cover one task, 25 settings and one validation fold.
* The PTB-XL leaderboard spread (row 4) compares architectures, not hyperparameter settings.

## 15.5 References

De Rainville, F.-M., Sebag, M., Gagné, C., Schoenauer, M., & Laurendeau, D. (2013). *Sustainable cooperative coevolution with a multi-armed bandit*. GECCO 2013 / arXiv. https://doi.org/10.48550/arxiv.1304.3138

Fialho, Á., Da Costa, L., Schoenauer, M., et al. (2009). Dynamic multi-armed bandits and extreme value-based rewards for adaptive operator selection in evolutionary algorithms (pp. 176–190). Springer. https://doi.org/10.1007/978-3-642-11169-3_13

Florea, A. C., & Andonie, R. (2018). A dynamic early stopping criterion for random search in SVM hyperparameter optimization (pp. 168–180). Springer. https://doi.org/10.1007/978-3-319-92007-8_15

He, X., Shan, W., Zhang, R., et al. (2023). Improved colony predation algorithm optimized convolutional neural networks for electrocardiogram signal classification. *Biomimetics, 8*(3), 268. https://doi.org/10.3390/biomimetics8030268

Jia, Y.-H., Chen, W.-N., Gu, T., et al. (2019). Distributed cooperative co-evolution with adaptive computing resource allocation for large scale optimization. *IEEE Transactions on Evolutionary Computation, 23*(2), 188–202. https://doi.org/10.1109/tevc.2018.2817889

Karroum, S., & Mazhar, S. (2025). *HyperArm bandit optimization: A novel approach to hyperparameter optimization and an analysis of bandit algorithms in stochastic and adversarial settings*. arXiv. https://doi.org/10.48550/arxiv.2503.10282

Makarova, A., Shen, H., Perrone, V., et al. (2022). *Automatic termination for hyperparameter optimization*. AutoML Conference 2022 / arXiv. https://doi.org/10.48550/arxiv.2104.08166

Omidvar, M. N., Li, X., & Yao, X. (2011). Smart use of computational resources based on contribution for cooperative co-evolutionary algorithms (pp. 1115–1122). ACM. https://doi.org/10.1145/2001576.2001727

Probst, P., Bischl, B., & Boulesteix, A.-L. (2018). *Tunability: Importance of hyperparameters of machine learning algorithms*. arXiv. https://doi.org/10.48550/arxiv.1802.09596

Santander-Baños, F., Hernández-Romero, N., Seck-Tuoh-Mora, J. C., et al. (2023). A novel hybrid model based on convolutional neural network with particle swarm optimization algorithm for classification of cardiac arrhythmias. *IEEE Access, 11*, 55515–55532. https://doi.org/10.1109/access.2023.3282315

Sipper, M. (2022). High per parameter: A large-scale study of hyperparameter tuning for machine learning algorithms. *Algorithms, 15*(9), 315. https://doi.org/10.3390/a15090315

Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms*. arXiv. https://doi.org/10.48550/arxiv.1206.2944

Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics, 25*(5), 1519–1528. https://doi.org/10.1109/jbhi.2020.3022989. Leaderboard: https://github.com/helme/ecg_ptbxl_benchmarking

Weerts, H., Mueller, A., & Vanschoren, J. (2020). *Importance of tuning hyperparameters of machine learning algorithms*. arXiv. https://doi.org/10.48550/arxiv.2007.07588
