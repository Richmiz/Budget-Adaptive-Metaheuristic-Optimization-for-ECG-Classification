# 24 — Response to the assessor's G4 statement (doc 23)

14 October 2026. All figures are recomputed from `runs/colab/decision_2026-10-14/` (227 hash-verified files) and the local rhythm arrays. Those arrays have the same SHA-256 as the Colab audit. Development evidence only; fold 7 was already used for task selection.

## 24.1 Accepted

| Doc 23 point | Check | Verdict |
|---|---|---|
| Confirmation is active (13 attempts: 7 confirmed, 3 probe vetoes, 3 stability rejections) | `decisions.jsonl`: 13 `propose_stop`, 7 `confirm_stop`, 3 re-proposals after a probe improvement, 3 `stable: false` | **Correct.** Doc 22's original statement ("cannot veto a stop") was wrong. Small precision: the 3 stability rejections are one configuration, re-evaluated identically in three smoke arms of seed 9402 (shared design and RNG; values 0.9314 and 0.9106 each time). The same holds for the 2 confirmed smoke stops of seed 9401. Independent events: 5 + 1 confirmed, 3 vetoes, 1 rejection. |
| Equal-cost quality needs refits, not validation ties | Incumbent re-evaluations in the five P runs | **Correct, and the evidence is stronger than stated.** 9 of 10 re-trainings of a selected incumbent scored below its search score, with a mean drop of 0.0070 AUROC (selection optimism). This is as large as the P−DEHB gap itself. |
| No re-timing to fix T_ref | `calibration_definition_v1.json` | Agreed. The definition is recorded, and doc 19 already uses 11.17 s and 12.95 s. |
| Freeze seven rhythm labels | Rhythm label counts | Verified: AFLT has 44 positives in folds 1–6 and 51 in folds 1–7; BIGU has 49 and 57. |
| Five-pair projection is a diagnostic | — | Agreed. "Infeasible" in the first draft of doc 22 was too strong. |
| Test count, notebook integrity | `pytest`; `verify_colab_bundle.py` | 76 pass; source bytes match. |

## 24.2 Not addressed by doc 23: the rhythm gain is mostly a receptive-field default

**Evidence 1 — every rhythm search chose the largest kernel.**

| | kernel 5 | kernel 9 | kernel 15 (max) | dilation 2 (max) |
|---|---|---|---|---|
| Rhythm incumbents (P and DEHB, 10 runs) | 0 | 0 | **10** | 7 |
| Superclass pilot incumbents (17 runs) | 6 | 11 | 0 | 0 |
| Untuned reference | | ✓ | | (dilation 1) |

**Evidence 2 — the gain is one class.** Per-class fold-7 AUROC, mean of 5 seeds:

| Class (positives in fold 7) | Reference | P | DEHB |
|---|---|---|---|
| SARRH, sinus arrhythmia (78) | **0.751** | 0.904 | 0.917 |
| SR (1,676) | 0.936 | 0.961 | 0.965 |
| SVARR (16) | 0.908 | **0.857** | **0.889** |
| AFIB, PACE, SBRAD, STACH | 0.96–0.99 | within 0.005 of reference | within 0.007 of reference |

Sinus arrhythmia is defined by beat-to-beat interval variation over the 10-second strip. The reference CNN was set up for diagnostic morphology, and its short field of view fits that reading of the data. **Implications:**

1. **The untuned baseline is weak for rhythm.** A reviewer can argue that the +0.02 "tuning gain" is mostly "use a longer kernel for rhythm", which an expert default gets without any search. The cost–benefit paper must answer this, or its rhythm result is a strawman.
   - **Proposal:** add a second, pre-declared reference for rhythm: a task-informed default (kernel 15, dilation 2, other settings as the reference).
   - Disclose that it was chosen after seeing development data.
   - Its role is to measure what search adds beyond a sensible default. That is the question a practitioner actually faces.
2. **Search-space boundary.**
   - All 10 rhythm incumbents sit on the upper kernel bound, and 7 on the upper dilation bound.
   - The best rhythm settings may lie outside the space. Widen it for rhythm (for example, kernel 21/25 or dilation 4), or justify the bound before G2.
   - Spending more G4 hours searching a space that is about to change buys evidence for a design we will not freeze.
3. **The P−DEHB gap is fragile.**
   - SVARR, with 16 positive ECGs, contributes about 60% of the −0.0073 macro gap: −0.032 / 7 ≈ −0.0046.
   - On that class, both tuners scored below the untuned model.
   - Per-class results, and a statement of the minimum positive count, should be pre-specified for every rhythm contrast.

## 24.3 On the matched follow-up notebook

The design is sound and correctly scoped (development only, pinned inputs, refits before assessment). But its parts differ in value:

| Part | Cost | Value |
|---|---|---|
| DEHB refits at P's stopping cost (5), fixed-40% prefix refits (≤ 10) | Minutes | **High.** Answers "equal cost, held-out quality", which §24.1 shows validation cannot. |
| 15 new searches (rr_utility, adaptive_patience, rr_patience) | ≤ 5.81 h search cap, roughly 50 CU with refits | **Moderate.** With 5 pairs and a paired seed SD around 0.009, AUROC mechanism contrasts will very likely be inconclusive (doc 23 §"Decision rules" accepts this). The useful outputs are a full-budget rehearsal of the four arms and search-time contrasts, which are far less noisy than AUROC. |

**Recommendation:** settle §24.2 first, because it decides what the rhythm arm of the main study is:
1. the task-informed reference (5 refits; minutes on G4, or about 15 minutes locally, since accuracy rather than timing is measured);
2. the kernel/dilation bound.

Then run the follow-up, in the revised space if it changes. If the space does not change, run the follow-up as it stands.

## 24.3a Result of the task-default check (run locally, 14 Oct)

`scripts/task_default_check.py`, plan `55f4b41e3cf9`, registered before training. Outputs are in `runs/local_dev/task_default_rhythm_2026-10-14/` and [evidence/task_default_rhythm.json](evidence/task_default_rhythm.json). Setup:
- Same refit partition (1–5/6, assessment fold 7), refit seeds and 50-epoch/patience-8 rule as the G4 precision pilot.
- Local RTX 3050; accuracy only.

**Platform control.** The plain reference refit locally scored 0.9300 against 0.9326 on G4. The difference is −0.0026 (CI [−0.0137, +0.0078]), so there is no systematic shift. Same-seed runs are not reproduced run-for-run across GPUs, though: per-seed differences reach 0.014. The same-platform contrast below was therefore added after registration and is labelled as such.

| Model | Search cost | Mean fold-7 macro AUROC | SARRH | SVARR (16 pos.) | PACE (29 pos.) |
|---|---|---|---|---|---|
| Reference, local | none | 0.9300 | 0.740 | 0.903 | 0.963 |
| **Task default (kernel 15, dilation 2), local** | **none** | **0.9497** | **0.918** | 0.847 | 0.953 |
| P (G4) | 57% of DEHB | 0.9518 | 0.904 | 0.857 | 0.968 |
| DEHB (G4) | 160 T_ref | 0.9591 | 0.917 | 0.889 | 0.970 |

| Contrast | Difference | Hierarchical 95% CI |
|---|---|---|
| Task default − reference, same platform (added) | **+0.0197** (all 5 seeds positive: +0.014 to +0.025) | [+0.0041, +0.0322] |
| Task default − reference (G4), registered | +0.0171 | [+0.0011, +0.0311] |
| Task default − P (G4) | −0.0020 | [−0.0184, +0.0138] |
| Task default − DEHB (G4) | −0.0094 | [−0.0214, +0.0016] |

**Reading.**
- Changing two default settings, with no search, recovers about all of P's gain and about 65–75% of DEHB's.
- It matches DEHB on sinus arrhythmia, the class that drove the tuning gain.
- What DEHB's search adds beyond it (−0.0094, with a CI that includes 0) sits mostly in the two rarest classes. SVARR, with 16 positives, contributes about 0.006 of it, and PACE, with 29, about 0.0024.

**Limits.** Development fold 7 has been reused. The default was chosen after seeing the tuned configurations, which favours the default. The comparison has 5 seeds and spans two GPUs.

**Consequence for the design.** For rhythm, the untuned reference must not be the only baseline. The informative question is what search adds beyond a task-appropriate default, and this check suggests the answer is small and concentrated in rare classes. This should be settled, together with the kernel/dilation bound, before more G4 hours are spent on rhythm searches.

## 24.4 Decisions requested

- [ ] Add a task-informed reference for rhythm (proposed: kernel 15, dilation 2), declared as data-informed.
- [ ] Widen the rhythm receptive-field choices, or record a justification for the current bound.
- [ ] Pre-specify per-class reporting and a minimum-positives statement for rhythm contrasts.
- [ ] Run the matched follow-up after the above (or now, if the space stays unchanged).
