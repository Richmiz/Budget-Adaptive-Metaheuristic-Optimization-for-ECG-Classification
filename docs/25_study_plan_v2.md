# 25 — Study plan v2: is tuning worth its compute? (one page)

**Chosen 14 Oct 2026 (route A). This page is the current plan.** Docs 00–24 are the history that led here.

**Working title.** *When is hyperparameter tuning worth its compute? A cost–benefit study of CNN-based ECG classification on PTB-XL.*

## Questions
1. **Gain:** how much does tuning improve held-out accuracy over (a) the original default and (b) a sensible task default?
2. **Cost:** how does the gain grow with search compute (two budget levels)?
3. **Stopping:** how much compute does our tuner's early stopping save, and how much accuracy does it give up?

## What we use
| | |
|---|---|
| Data | PTB-XL v1.0.3, 12-lead, 100 Hz. Official patient-separated folds. |
| Tasks | **Diagnosis**: 5 classes (NORM, MI, STTC, CD, HYP). **Rhythm**: 7 classes, frozen (AFIB, PACE, SARRH, SBRAD, SR, STACH, SVARR). |
| Splits | Search: train folds 1–7, validate on 8. Final models: train 1–8, early-stop on 9. **Test: fold 10, opened once, at the end.** |
| Model | Standard 1-D ResNet with 9 settings to tune. Not a new architecture. |
| Baselines (no search) | **D0**: original default (kernel 9, dilation 1). **D1**: sensible task default. For rhythm, kernel 15 / dilation 2. For diagnosis, D1 = D0, because the longer view hurts diagnosis (step 1). D1 is disclosed as chosen from development data. |
| Tuners | **Random search** (simple standard), **DEHB** (strong published method), **P** (our DE tuner with early stopping). |
| Budgets | 80 and 160 time units per search (unit = median training time: 11.2 s for diagnosis, 13.0 s for rhythm on G4). P may stop early. |
| Repeats | 10 per tuner × budget × task. |

## What we measure
- **Main result:** test AUROC gain over D0 and over D1, per task, tuner and budget, with uncertainty over patients and repeats. Shown as a **cost–quality chart**: accuracy against G4 hours.
- **Also reported:** results per class (with the number of positive cases), P's stopping times, and the compute P saved.
- **Not tested:** a "no worse than DEHB" verdict. The noise makes it unreliable (docs 22–24).

## Step 1: two local checks (done 14 Oct; rules registered before training, plan `55252745161e`)
Evidence: [evidence/default_range_check.json](evidence/default_range_check.json). Development fold 7, 5 repeats, local RTX 3050.
1. **Wider rhythm view: no gain, so the search range stays as it is** (kernel 5/9/15, dilation 1/2), and rhythm needs no re-timing.

   | Compared with D1 | Difference | 95% CI |
   |---|---|---|
   | Kernel 25 | −0.002 | [−0.018, +0.014] |
   | Dilation 4 | −0.010 | [−0.028, +0.008] |

2. **D1 on diagnosis: it hurts, so diagnosis keeps D0 as its task default.** The difference is −0.0097 (CI [−0.0146, −0.0052]), lower in all 5 repeats. This matches the diagnosis searches, which preferred kernel 5–9 and dilation 1. The right default depends on the task.

## Compute (G4)
| Item | Count | G4 hours |
|---|---|---|
| Main searches | 2 tasks × 3 tuners × 2 budgets × 10 = **120** | ≤ 48 (≈ 42 expected, because P stops early) |
| Final models | 120 tuned + 30 baseline (D0 for both tasks, D1 for rhythm) = 150 | ≈ 2 |
| Session setup (about 10 sessions) | | ≈ 3 |
| **Total** | | **≈ 45–53 h ≈ 400–470 compute units (≈ $40–47)** |

Optional add-on: P without stopping, at 160 units (20 runs, about +11 h). It shows directly what continuing would have gained.
Spent so far: about 70 units (pilot plus decision run).

## Timeline
| Weeks | Dates | Work | Who |
|---|---|---|---|
| W1–W2 | 14–25 Oct | ~~Step 1 checks~~ (done); code updates (task-aware runner, D1 baseline, new analysis, Colab notebook with live progress); full protocol; **lock (freeze + git tag)** | Claude; you approve the lock |
| W3–W5 | 26 Oct–15 Nov | **Main searches on Colab**, about 10 sessions of about 5 h; resumes after disconnects | You run; Claude watches and checks |
| W6 | 16–22 Nov | Final models; lock the analysis; **one-time test on fold 10** | You run; Claude checks |
| W7–W8 | 23 Nov–6 Dec | Analysis, figures, results write-up | Claude drafts; you review |
| W9–W16 | Dec–Jan | Manuscript | Together |
| Target | Feb–Mar 2027 | Submission (the original plan said Apr 2027) | |

## What the paper can say, whatever the numbers
- **Search clearly beats D1:** "tuning pays beyond a good default by X, at Y G4 hours."
- **Search ≈ D1:** "a task-appropriate default captures most of the gain; search adds at most X."
- **Either way:** P's compute saving and the accuracy it gives up, with honest uncertainty.

## Rules that protect the result
- Fold 10 is never opened before the analysis is locked, and then only once.
- The plan is locked (frozen protocol plus git tag) before any main run. Changes after that are logged as amendments.
- Development results (fold 7) are not reported as test results.
