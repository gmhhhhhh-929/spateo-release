# Spatial IO and preprocessing

## Unified public interfaces

The maintained implementations now live directly under `spateo.io` and `spateo.preprocessing`. Historical flat IO modules remain as compatibility wrappers, so downstream code can migrate gradually without importing `protocol_io` or `protocol_pipeline`.

```python
import spateo as st

adata = st.io.read_auto_spatial("/path/to/dataset")
st.pp.preprocess_spatial(adata, recipe="auto")
```

The automatic reader detects Atera, Xenium, Visium, Visium HD, Stereo-seq/BGI, Slide-seq, MERFISH/MERSCOPE, CosMx, seqFISH, and STARmap+ layouts. It records the chosen technology, evidence, reader arguments, source path, and a bounded file manifest in `adata.uns["spateo_io"]`.

## Atera In Situ

The current public Atera whole-transcriptome preview is distributed in a layout that closely resembles Xenium Onboard Analysis v4. The dedicated reader validates Atera metadata or named stain channels before automatic dispatch and marks the result as `preview-xenium-v4`, because 10x states that the final Atera output format will change.

```python
adata = st.io.read_atera(
    "/path/to/extracted/outs",
    image_key="dapi",             # dapi, boundary, rna, stroma, or channel index
    load_boundaries=True,
    load_nucleus_boundaries=True,
    load_cell_groups=True,
)
```
The reader loads the cell-feature matrix, cell metadata and centroids, optional cell/nucleus boundaries, a bounded OME-TIFF pyramid level, optional cell-group annotations, and H&E alignment metadata. Full transcript tables and full-resolution image pyramids are inventoried but not loaded implicitly.

References: [10x Atera platform](https://www.10xgenomics.com/platforms/atera), [10x public Atera dataset](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer).

## Visium metadata behavior

`read_visium` always reads tissue positions and scale factors. `load_images=False` skips only pixel arrays, so spatial coordinates remain available. Current Parquet/CSV and legacy headerless positions files are supported; `obsm["spatial"]` stores image `(x, y)` as full-resolution column then row. Available diagnostic and source-image files are recorded separately from loaded arrays.

## Spatial preprocessing contract

`preprocess_spatial` preserves raw counts in `layers["counts"]`, writes normalized and log-normalized layers, computes spatial QC, optionally builds per-library spatial graphs, selects features, and runs memory-aware PCA. `recipe="auto"` uses `adata.uns["spateo_io"]["technology"]` when available.

Important defaults:

- count validation warns on non-integer, negative, or non-finite input and can be made strict with `validate_counts="error"`;
- upper-tail adaptive filtering is opt-in because high-count spatial regions may be real tissue structure;
- local spatial outliers are annotated but removed only with `filter_local_outliers=True`;
- mitochondrial genes are excluded from feature ranking by default, while ribosomal and hemoglobin genes are not silently discarded;
- sparse centered PCA densifies only below a memory cap and otherwise uses truncated SVD.

```python
processed = st.pp.preprocess_spatial(
    adata,
    recipe="auto",
    n_top_genes=3000,
    local_qc=True,
    filter_local_outliers=False,
    build_spatial_graph=True,
    inplace=False,
)
```
