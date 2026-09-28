# Serial-slice quality control

## Position in the workflow

Serial-slice quality control is an optional preprocessing step after spatial
I/O and before alignment:

```text
read spatial data -> optional serial-slice QC -> alignment -> downstream analysis
```

The implementation is non-destructive. It does not remove observations,
rewrite coordinates, or modify the input H5AD unless the caller explicitly
requests a small provenance summary in `adata.uns`.

## Supported inputs

The core API supports:

1. one AnnData object containing an ordered slice-id column;
2. one three-dimensional AnnData object with discrete z coordinates;
3. one H5AD file per slice (natural filename order by default; set
   `sort_inputs=False` for an explicitly ordered manifest);
4. multiple independent datasets through `scan_h5ad_collection`.

Independent datasets are evaluated separately. A local expectation from one
dataset is never used for a slice in another dataset.

The core implementation uses the existing Spateo runtime dependencies
(`anndata`, `numpy`, `pandas`, `scipy` and `h5py`) and introduces no additional
package requirement. It is compatible with the repository's supported Python
3.10--3.12 environment.

## Evidence domains

The detector combines four domains:

- density and tissue integrity, including location density, connected
  components, sparse-neighborhood tails and spatial holes;
- expression capture, including captured counts, detected genes, complexity
  and spatially concentrated low-capture observations;
- damage-associated evidence, including mitochondrial fraction when available;
- cross-slice continuity, including expression-profile and optional cell-type
  composition divergence.

Captured counts are an H5AD library-size or molecule-capture proxy. They are
not raw sequencing depth unless read-level metadata establishes that meaning.

## Parallel binary decision routes (tissue-loss v3)

Tissue-loss v3 is an **experimental policy**, not an independently certified
biological classifier. Public reports use `retain / exclude`; the engine retains
`keep` as the backward-compatible value for retain. Retain means insufficient
evidence for exclusion, not verified healthy tissue. No source cells are deleted.

### Bilateral physical-loss route

For each available 3/5/7-slice window, exclude the focal slice and calculate
separate left and right medians. For each measurement, use

```text
reference = min(median(left), median(right))
deficit = max(0, 1 - observed / reference)
```

Both sides must provide finite positive reference measurements. The geometry
route requires at least 20 reference locations by default. Using the lower
side requires a deficit relative to both neighboring sides, reducing sensitivity
to an ordinary monotonic taper. These deficits are relative observations, not
known amounts of missing tissue or simulated injection doses.

- **Geometry:** at least 50% location loss plus at least 35% area **or** density
  loss. Edge truncation can reduce area while leaving a dense remnant; random
  removal or internal tears can reduce density without destroying the hull.
- **Capture:** at least 50% loss of median captured counts plus at least 30%
  loss of median detected genes. This route requires explicit measured-capture
  availability; normalized proxies, declared annotation one-hot values, or
  missing provenance cannot substitute for raw count evidence.

The primary window must support a route, as must at least
`min(loss_min_confirming_windows, available_windows)` confirming windows
(`loss_min_confirming_windows=2`). No valid confirming windows means no pass.
The matching primary-window scale counts toward this total: with primary 3 and
available 3/5/7, primary 3 plus either 5 or 7 suffices, not two additional scales.
The route does **not** require a second anomaly domain or the legacy aggregate
score threshold. Location count, area and density are correlated physical
measurements, not statistically independent domains.

### Restricted anatomical protection

Normal expression alone no longer establishes that a geometrically small slice
is anatomically intact. A legacy geometry-looking protection candidate must
also show actual location count **and** area between its immediate neighbors,
normal molecular/damage/continuity evidence, and no independent loss evidence.
This protects supported local transitions, not arbitrary dense remnants.

Endpoints, singleton series, missing measurements, uniformly degraded series
and long runs of consecutive loss may remain undetectable by bilateral
comparison. Neither these heuristics nor synthetic experiments establish that
every tissue-loss type or severity is detectable. Real-data exclusion counts
are not accuracy estimates without independent labels.

### Retained legacy score route

The diagnostic detector first calculates an anomaly score and may produce an
internal `keep / review / exclude` recommendation. A released filtering policy
uses two additional stages:

1. **Threshold triage:** locked keep and exclude thresholds create a score-only
   `threshold_band`. Detector agreement, two-sided context and partial-structure
   protection can downgrade an unsafe high-score exclusion into the internal
   review queue.
2. **Review fine screen:** only review rows are tested for multi-domain
   corroboration, one severe domain, adequate score confidence and stable
   support across the configured 3/5/7-slice windows. Every condition must pass
   for exclusion; otherwise the complete operational action is keep.

The final action is exclude when **either** the legacy score route or the v3
loss route passes. Otherwise it is retain (`keep` in the engine). The binary audit preserves the
threshold band, guarded triage, review resolution, failed conditions and final
decision basis, together with `tissue_loss_exclusion_gate`, `tissue_loss_route`,
`tissue_loss_reason`, `tissue_loss_window_support`, the five loss fractions,
their references, and per-window candidates. Low aggregate scores can therefore
coexist with a justified physical-loss exclusion.

## Migration and validation scope

The native module and bundled QC skill runtime use the same scientific source.
This update includes previously packaged safeguards absent from the older
native baseline: explicit-zero sparse cleanup, selected raw counts taking
priority over stale observation summaries, one-hot capture unavailability,
complete review tiers, and honest experimental application scopes.

`SliceQCConfig(tissue_loss_enabled=True)` computes v3 evidence by default.
Set it false only for an explicitly labeled legacy replay. A legacy saved audit
does not become v3 simply by changing policy JSON: rescan the source or rescore
a complete metric cache with its measured-capture contract. Evidence rows and
all confirming windows must carry `loss_evidence_version='bilateral-v3'` and
`loss_evidence_config` matching the frozen policy parameters. Missing or
mismatched evidence must not be silently accepted.

The new policy deliberately retains false independent-validation metadata.
Historical joint-review-v2 results are not v3 results. Record new benchmark
outcomes separately with their code/config hashes, all doses and defect types,
negative controls, natural transitions and failures. Synthetic detection means
**final exclude / eligible injected targets**, not review-plus-exclude candidates.
Repeated injections into the same slice are not independent biological samples.

## Minimal API examples

One in-memory dataset:

```python
import spateo as st

metrics = st.pp.calculate_slice_quality(
    adata,
    slice_key="slice_id",
    spatial_key="spatial",
    layer="counts",
    config=st.pp.SliceQCConfig(window=3),
)
```

Multiple independent datasets:

```python
results = st.pp.scan_h5ad_collection(
    {
        "study_a": "study_a.h5ad",
        "study_b": ["b_01.h5ad", "b_02.h5ad", "b_03.h5ad"],
    },
    dataset_options={
        "study_a": {"slice_key": "slice_id", "spatial_key": "spatial", "layer": "counts"},
        "study_b": {"slice_key": "auto", "spatial_key": "spatial", "layer": "counts"},
    },
)
```

Write lightweight scientific artifacts:

```python
st.pp.write_slice_quality_collection_outputs(
    results,
    "qc_outputs",
    write_display_payload=True,
)
```

`write_display_payload=True` writes a bounded point sample for the companion
skill; it does not embed an HTML template in the package.

Apply the explicitly experimental v3 policy from the companion skill:

The versioned [tissue_loss_v3.json policy](https://github.com/gmhhhhhh-929/Spateo-skills/blob/main/skills/spatial-slice-quality-qc/policies/tissue_loss_v3.json)
is distributed with the companion skill. It is not bundled as native package
data; record the actual policy file SHA256 used for each application.

```python
import json
from pathlib import Path
from spateo.preprocessing.slice_quality import (
    SliceQCConfig,
    add_multiscale_exclusion_evidence,
    scan_h5ad_series,
    write_high_confidence_outputs,
    write_slice_quality_outputs,
)

# Policy and config values must match; these are the packaged v3 defaults.
config = SliceQCConfig(window=3, tissue_loss_enabled=True)
result = scan_h5ad_series(
    ["specimen.h5ad"], slice_key="slice_id", spatial_key="spatial",
    layer="counts", config=config,
)
write_slice_quality_outputs(result, "new_qc_run", write_display_payload=True)
evidence = add_multiscale_exclusion_evidence(result.metrics, config=config)
policy = json.loads(Path("/path/to/spatial-slice-quality-qc/policies/tissue_loss_v3.json").read_text())["policy"]
write_high_confidence_outputs(
    evidence, policy, "new_qc_run", application_scope="experimental_policy",
)
```

For the skill CLI, development consent is explicit:

```bash
python subskills/spatial-slice-quality-viewer/scripts/build_viewer.py \
  --input-dir /path/to/new_qc_run --output-dir /path/to/new_report \
  --policy policies/tissue_loss_v3.json --allow-unvalidated-policy \
  --application-scope experimental_policy
```

`--allow-unvalidated-policy` does not confer certification. Omit `--policy` to
render an existing complete binary audit unchanged. The canonical report shows
the actual loss fractions, confirming-window counts, route, limitations and
color scales; scientific decisions are not recomputed in JavaScript.

## Package and skill boundary

`spateo.preprocessing.slice_quality` contains the scientific runtime:

- input resolution and metric calculation;
- single- and multi-dataset scanning;
- controlled defect simulation and paired evaluation;
- multiscale evidence and two-stage binary policy application;
- CSV/JSON outputs and provenance.

The companion `spatial-slice-quality-qc` skill contains presentation and
workflow automation:

- single-dataset HTML reports;
- KDE density, absolute expression-capture and connected-component panels;
- grouped multi-dataset collection pages;
- publication tables and remote/local rendering orchestration.

This boundary keeps the conda runtime focused while preserving the complete
interactive report used for datasets such as the E9.5 mouse heart.
