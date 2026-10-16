# Publication figure captions — v2

All figures use ten paired seeds on official fold 10. They describe one dataset and one CNN
family. D0 is the original default. Rhythm D1 (kernel 15, dilation 2) was selected after inspecting
development tuning outcomes, checked locally and frozen before the main test. D1 has no per-run
search once fixed; its earlier selection involved development compute.

## Figure 1. Search compute and test discrimination

Mean test macro AUROC against mean accounted search minutes per run for random search, native
DEHB and P, at 80 and 160 time units. Error bars are pointwise 95% percentile intervals from
10,000 patient-cluster/paired-seed bootstrap draws. These absolute-score intervals were added
after test inspection for figure presentation. Horizontal lines show default means only.
Accounted time includes charged cached initialization; it is not billed runtime. Use Figure 2
for paired gains; overlapping absolute-score intervals do not establish equivalence.

## Figure 2. Paired gains over the task default

Mean paired macro AUROC differences versus diagnosis D0 and rhythm D1, with the locked
10,000-draw patient/seed bootstrap 95% intervals. Filled markers indicate Holm-adjusted
one-sided bootstrap p < 0.05 within the six-comparison primary family for that task.
The displayed intervals are pointwise, not multiplicity-adjusted. Hollow markers indicate
that a positive gain was not established; they do not establish equivalence to the default.
Circles denote 80 units and squares 160 units.

## Figure 3. P versus DEHB: discrimination and search time

Locked secondary estimates at each budget. Left: P minus DEHB macro AUROC and 95% intervals
from 10,000 patient/paired-seed draws. Right: geometric mean P/DEHB accounted search-time
ratios and 95% intervals from 5,000 paired-seed bootstrap draws. Quality differences are
uncertain despite lower search time. No non-inferiority, equivalence or component-attribution
claim is made from these complete-policy comparisons.

## Figure S1. Paired effect of increasing the search budget

Mean paired macro AUROC differences, 160 minus 80 units, with 95% patient/seed percentile
intervals from 10,000 draws. Positive values favor 160 units. This supplemental analysis
was specified after test inspection and uses existing predictions. It adds no confirmatory
hypothesis test and does not establish an absence of a budget effect.

## Figure S2. Per-class descriptive gains

Differences between mean per-class AUROC of each tuned arm and its task default, multiplied
by 1,000. Values come from locked secondary summaries and have no per-class uncertainty bars
or hypothesis tests. Labels give positive test-record counts, not positive patient counts.
Rhythm SVARR (14 positive records) and PACE (28) require particular caution. A common color
scale is used for both tasks. Rand. denotes random search.
