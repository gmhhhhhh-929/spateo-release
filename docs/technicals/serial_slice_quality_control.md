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

## Two-stage binary decision

The diagnostic detector first calculates an anomaly score and may produce an
internal `keep / review / exclude` recommendation. A released filtering policy
uses two additional stages:

1. **Threshold triage:** locked keep and exclude thresholds create a score-only
   `threshold_band`. Detector agreement, two-sided context and partial-structure
   protection can downgrade an unsafe high-score exclusion into the internal
   review queue.
2. **Review fine screen:** contiguous calibrated score bands cover the complete
   review interval, so every review row is evaluated. A band is exclude-enabled
   only when calibration shows positive safe detection gain; unsupported lower
   scores form an explicit keep-only band. Enabled lower-score rules require
   stricter evidence and retain two-sided-context, anatomical-protection,
   confidence and 3/5/7-window-stability gates. Every enabled condition must
   pass for exclusion; otherwise the complete operational action is keep.

The public calls contain only `keep / exclude`. The binary audit preserves the
threshold band, guarded triage, review resolution, failed conditions and final
decision basis.

A frozen policy represents the fine screen explicitly:

```python
policy = st.pp.HighConfidencePolicy(
    keep_max_score=K,
    exclude_min_score=E,
    unresolved_action="keep",
    review_exclusion_tiers=(
        st.pp.ReviewEvidenceTier("lower_keep_only", K, H, enable_exclude=False),
        st.pp.ReviewEvidenceTier("high", H, None, 2, 0.85, 1.0, 0.90),
    ),
)
```

`K`, `H`, `E` and the evidence requirements are placeholders here. A
released policy must load values selected on a declared calibration pool and
then pass an unchanged evaluation on unseen datasets; they are not API
defaults. The tier intervals must be contiguous beginning at `K`. A keep-only
band is valid only when calibration found no safe positive-gain exclusion rule
for that interval.

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
