# Serial-slice quality threshold calibration

This document defines how a release policy obtains its thresholds. The values
are learned outside the runtime API and stored in a versioned policy JSON; they
are not universal constants or `SliceQCConfig` defaults.

Do not confuse the policy cutoffs with `SliceQCConfig.review_threshold` and
`SliceQCConfig.exclude_threshold`. The latter create the diagnostic detector's
internal recommendation and are only one input to the guarded policy; they do
not directly publish a keep or exclude action.

## Stage-1 anchors

Stage 1 searches score thresholds at 0.001 resolution. Each enabled direction
must have at least 1,000 publications per calibration dataset and a two-sided
95% Wilson error upper bound no greater than 1%. The broadest safe keep threshold
is selected. The smallest safe exclude threshold above keep is selected after
applying a manually locked conservative floor.

The current base policy used 18 calibration datasets with 45,000 randomized
plans (157,658 injected technical-defect trials and 156,399 matched controls),
then 59 untouched validation datasets with 147,500 plans (512,249 injected
trials and 491,809 matched controls). It selected:

The calibration pool comprised 11 Drosophila, four planarian, two mouse-heart,
and one human-embryo series. The untouched pool comprised 38 Drosophila, 13
planarian, six multi-species cerebellum, and two coordinate-perturbed series;
thus the 77 total series represent 75 biological datasets plus two artificial
coordinate controls.

- keep maximum `K = 0.129`;
- direct-exclude minimum `E = 0.700`.

The `0.700` floor was locked from E11.5 mouse-heart manual adjudication before
unseen validation. The highest unprotected watchlist score was `0.690986` and
was rounded upward to the next 0.01. Two confirmed exclusions scored `0.730391`
and `0.777534`.

The untouched validation run observed zero false exclusions; its 95% Wilson
exclude-error upper bound was `0.00141%`. It observed 434 errors among 398,112
published keeps (95% Wilson upper bound `0.1198%`). All 59 datasets certified
exclude, 55 of 59 certified keep, and all five validation folds passed.

## Stage-2 tier selection

Stage 2 covers every Stage-1 review score. Candidate score boundaries and
evidence requirements are selected only on the 18 calibration datasets. An
exclusion route is retained only when it is safe and adds at least one correctly
excluded technical-defect trial beyond existing routes. If no lower-score rule
has positive safe gain, that remaining interval is explicitly encoded as
`enable_exclude=false` and resolves to keep. Enabled lower-score tiers must have
strictly stronger evidence requirements. A candidate is safe only when all of
the following hold:

1. pooled false-exclude and exclude-error 95% Wilson upper bounds are at most 1%;
2. every dataset passes the false-exclude bound with at least 300 benign trials;
3. coherent taper, natural lumen, single connected-component shift and single
   expression/composition shift each pass the same bound;
4. all five dataset-level calibration folds pass;
5. detector candidacy, two-sided context, inactive anatomical protection,
   minimum score confidence and 3/5/7-window stability all pass.

The high tier is selected first, followed by successively lower bands. Within
the safe, positive-gain candidate set, the ranking maximizes correctly excluded
randomized technical defects. The complete policy is then frozen and evaluated once on
the 59 unseen datasets. Publication additionally requires every unseen dataset,
each benign-control class and all five validation folds to pass.

## Learned values versus design guardrails

The benchmark learns the Stage-1 policy cutoffs and the Stage-2 score boundary,
domain count, severity cutoff, and detector mode from declared candidate grids.
Other conditions are deliberately fixed before calibration:

| Quantity | Value | Status and rationale |
|---|---:|---|
| Evidence-domain support | `0.45` | Scoring definition for counting a substantial domain; not optimized in this benchmark. |
| Windows | `3/5/7` | Predeclared local scales for short, medium, and wider continuity. |
| Window stability | `1.00` | All available scales must pass; a conservative design guardrail. |
| Score confidence | `0.90` | Excludes low-context rows; terminal confidence is `0.72`, so endpoints cannot pass. |
| Two-sided context | required | Prevents a one-sided endpoint comparison from driving exclusion. |
| Anatomical protection | must be inactive | Prevents coherent taper or partial-section geometry from acting as technical-failure evidence. |
| Maximum Wilson error upper bound | `1%` | Predeclared release safety criterion, not a fitted value. |
| Benign trials per dataset | at least `300` | Predeclared minimum; the Wilson bound must also pass. |

These fixed values are validated with the frozen policy, but the present
experiment does not establish that they are uniquely optimal. Changing one
requires a new calibration version and a new untouched validation pool.

## Frozen Stage-2 result

- Calibration used 18 datasets, 9,000 randomized plans, 31,420 technical-
  defect trials, and 31,177 benign controls. It excluded 19,853 defects with
  zero false exclusions (63.19% recall); all five calibration folds passed.
- Frozen validation used 59 untouched series, 29,500 plans, 102,476 defect
  trials, and 100,468 benign controls. It excluded 65,632 defects with zero
  false exclusions (64.05% recall). The false-exclude and exclude-error 95%
  Wilson upper bounds were 0.00382% and 0.00585%; every dataset, benign-control
  class, and validation fold passed.
- `0.129 <= score < 0.540` is an explicit keep-only band. All 45 tested lower-
  score candidates were safe but added zero correctly excluded defects, so no
  exclusion rule was inferred for this interval.
- `score >= 0.540` is the enabled Stage-2 route: candidate-level detector
  support, at least two substantial domains, maximum domain score at least
  0.80, score confidence at least 0.90, two-sided unprotected context, and
  100% support across all available 3/5/7 windows. Floors 0.50 and 0.54 tied
  for maximum calibration gain; the conservative tie-break selected 0.54.
- Orthogonal manual checks were 23/23 correct (21 keep and two exclude).
  Count-matrix injections excluded 1/10 targets and changed 0/121 non-injected
  baseline keeps to exclude. This independently supports specificity but
  exposes limited sensitivity to isolated single-factor defects; it is not
  evidence of universal low-quality-slice recall.

Repeated perturbations are stress trials rather than independent biological
replicates. Dataset-level separation, per-dataset error gates, control-type
gates, fold checks, count-matrix experiments and manual labels are retained as
distinct evidence rather than pooled into one inflated sample count.
