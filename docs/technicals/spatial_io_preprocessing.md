# Spatial IO and preprocessing

## Unified public interfaces

The maintained implementations now live directly under `spateo.io` and `spateo.preprocessing`; the former duplicate IO modules and the temporary `protocol_io`/`protocol_pipeline` namespaces have been removed.

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

`preprocess_spatial` preserves raw counts in `layers["counts"]`, writes normalized and log-normalized layers, computes spatial QC, optionally builds per-library spatial graphs, selects features, and runs memory-aware PCA. Acquisition technology and analysis method are deliberately independent: `adata.uns["spateo_io"]["technology"]` only supplies safe graph hints (for example, six grid neighbors and `in_tissue` for Visium).

The maintained recipes are intentionally small:

- `auto` selects `standard` while retaining the detected technology in provenance;
- `standard` performs total-count normalization, `log1p`, mean-binned HVG/SVG selection and PCA;
- `pearson_residuals` uses a negative-binomial count model and residual variance for feature ranking and PCA;
- `raw` performs count validation, QC/filtering and an optional spatial graph, but leaves expression untransformed for downstream count models.

Technology-named recipes such as `visium`, `xenium`, `atera` and `generic` are deprecated aliases of `standard`. They no longer install hard-coded QC thresholds. `sctransform` is not exposed as a production recipe because the former sparse approximation discarded negative Pearson residuals; its experimental low-level module remains available for method development.

Important defaults:

- count validation warns on non-integer, negative, or non-finite input and can be made strict with `validate_counts="error"`;
- upper-tail adaptive filtering is opt-in because high-count spatial regions may be real tissue structure;
- no platform-specific count/gene cutoffs are inferred; inspect QC distributions and pass `min_counts`, `min_genes`, `max_pct_mt` or `adaptive_qc=True` deliberately;
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

### Replacing Dynamo size-factor normalization

For a conventional spatial count matrix, the former call

```python
dyn.pp.normalize_cell_expr_by_size_factors(
    adata=stage_adata,
    layers="X",
    X_total_layers=True,
    skip_log=True,
)
```

becomes the following Dynamo-free call with the same median-library-depth and no-log semantics:

```python
import spateo as st

st.pp.normalize_total(
    stage_adata,
    layer="X",
    out_layer="X",
    target_sum=None,
)
```

For a new workflow, preserve integer counts instead of replacing `X`:

```python
stage_adata.layers["counts"] = stage_adata.X.copy()
st.pp.normalize_total(
    stage_adata,
    layer="counts",
    out_layer="norm",
    target_sum=None,
)
```

`target_sum=None` uses the median positive library size; use `target_sum=1e4` for fixed counts per ten thousand. `skip_log=True` needs no replacement because `normalize_total` never logs. Add `st.pp.log1p_layer(...)` as an explicit next step when required. Dynamo's `X_total_layers=True` only changes behavior when total RNA is assembled from kinetic layers such as spliced/unspliced or new/old; it has no extra effect for a standard spatial `X`/`counts` matrix.

This separation follows the production patterns used by [Scanpy total-count normalization and HVG selection](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html), [Squidpy's spatial tutorials](https://squidpy.readthedocs.io/en/latest/notebooks/tutorials/tutorial_vizgen.html), [Seurat's spatial workflow](https://satijalab.org/seurat/articles/spatial_vignette), and [Giotto's normalization API](https://giottosuite.com/reference/normalizeGiotto.html). Advanced spatial-bias correction should remain an explicit method choice rather than a hidden platform default.

## Native numerical runtime

Spateo does not require Dynamo. Internal utilities now provide AnnData matrix selection, sparse-safe normalization, nearest-neighbor graphs, resolution-aware graph clustering, spatially balanced sampling, robust sparse Gaussian-kernel vector fields, analytical Jacobians, and trajectory integration. Existing public Spateo functions and stored keys such as `VecFld_morpho`, `X_ctrl`, `grid_V`, and `fate_morpho` remain available. The compatibility contracts were informed by the public [Dynamo preprocessing and vector-field APIs](https://github.com/aristoteleo/dynamo-release/tree/master/dynamo), but the maintained numerical implementation is local to Spateo.
