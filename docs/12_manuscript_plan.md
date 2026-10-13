# 12 — Manuscript plan

Structure follows proposal §3.9.4. Abstract numbers are written **only after** the locked analysis.

| Section | Content | Source |
|---|---|---|
| Introduction | Cost-opaque metaheuristic gains in biomedical DL (SLR evidence: 10 favourable, 15 mixed, 19 uncertain trade-offs in 44 studies); the allocation and stopping problem; contributions as narrowed by doc 11 | Proposal Ch. 1, [S1] |
| Related work | Multi-fidelity HPO, bandit allocation, AOS, cost-aware and terminating HPO, metaheuristics for ECG | Doc 11 matrix |
| Problem formulation | max M_val(x) s.t. C_search ≤ B; cost accounting; what is and is not charged | Doc 02 §2.7 |
| Marginal-utility controller | Reward, score, actions, state machine, safeguards; complexity and overhead | Doc 04 |
| ECG experimental protocol | PTB-XL, labels, partitions, model, domain, evaluators, comparators, budgets, seeds, GPU | Docs 02, 03, 05 |
| Results | T3–T4, F2–F3, F5; deployment T6 | Doc 07 |
| Ablation and failure analysis | T5, F4, F6; premature termination; when short-term gains mislead | Doc 07 §7.5, §7.7 |
| Discussion | When block allocation helps; cost of safeguards; limits (single dataset, one CNN family, one GPU type, internal validation only) | — |
| Conclusion | Claims matched to evidence, including negative findings | — |

## Claim discipline

* "Efficient" is used only if the doc 07 §7.3 criterion is met.
* No clinical-utility or external-validity language. Fold 10 is internal, patient-separated generalization.
* Report search GPU-hours, final-training cost and inference cost separately, with the GPU type.
* Report every pre-specified outcome, including the ones that do not support the hypotheses, and list deviations (doc 07 §7.10).

## Reporting checklist (internal)

- [ ] Data version, label mapping and exclusions with counts
- [ ] Patient-separation audit statement
- [ ] Full search space and the frozen controller values (τ, κ, T_ref, tolerance)
- [ ] Comparator implementations and versions
- [ ] Hardware, software versions, determinism caveat
- [ ] Budget definitions in T_ref and seconds
- [ ] Per-seed results in the supplement
- [ ] Code and reproducibility package link
