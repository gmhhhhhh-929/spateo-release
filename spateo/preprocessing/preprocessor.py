"""Recipe-based spatial transcriptomics preprocessing."""

from __future__ import annotations

from typing import Literal, Optional

from anndata import AnnData

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .external.pearson_residual_recipe import pearson_residuals
from .external.sctransform import sctransform as run_sctransform
from .feature import select_spatial_features
from .graph import expression_neighbors, spatial_neighbors
from .normalization import calculate_size_factors, normalize_total
from .pca import pca
from .qc import (
    calculate_spatial_qc,
    filter_genes_by_spatial_qc,
    filter_spots,
    flag_local_qc_outliers,
)
from .transform import log1p_layer
from .utils import standardize_spatial_adata

logger = LoggerManager.get_main_logger()

Recipe = Literal[
    "auto",
    "visium",
    "visium_hd",
    "generic",
    "stereoseq",
    "slide_seq",
    "slideseq",
    "seqfish",
    "merfish",
    "xenium",
    "atera",
    "cosmx",
    "imaging",
    "pearson_residuals",
    "sctransform",
]


def _infer_recipe(adata: AnnData, recipe: str) -> str:
    aliases = {
        "bgi": "stereoseq",
        "stereo": "stereoseq",
        "slideseq": "slide_seq",
        "visium_hd_bin": "visium_hd",
        "visium_hd_cellseg": "visium_hd",
        "nanostring": "cosmx",
        "starmap_plus": "imaging",
    }
    if recipe != "auto":
        return aliases.get(recipe, recipe)
    io_metadata = adata.uns.get("spateo_io", {})
    detected = str(io_metadata.get("technology") or io_metadata.get("type") or "").lower()
    detected = detected.removesuffix("_seg")
    return aliases.get(detected, detected) if detected else "generic"


def _recipe_defaults(recipe: str) -> dict[str, object]:
    valid = {
        "visium",
        "visium_hd",
        "generic",
        "stereoseq",
        "slide_seq",
        "seqfish",
        "merfish",
        "xenium",
        "atera",
        "cosmx",
        "imaging",
        "pearson_residuals",
        "sctransform",
    }
    if recipe not in valid:
        raise ValueError(f"Unknown spatial preprocessing recipe {recipe!r}; expected one of {sorted(valid)}.")
    defaults = {
        "use_in_tissue": False,
        "coord_type": "generic",
        "n_neighbors": 8,
        "min_cells": 3,
        "target_sum": 1e4,
        "feature_method": "hvg",
        "run_pca": True,
        "min_counts": None,
        "min_genes": None,
        "max_pct_mt": None,
        "adaptive_qc": False,
        "radius": None,
    }
    if recipe == "visium":
        defaults.update(
            {"use_in_tissue": True, "coord_type": "grid", "n_neighbors": 6, "min_counts": 100, "min_genes": 50}
        )
    elif recipe in {"visium_hd", "stereoseq"}:
        defaults.update({"min_counts": 50, "min_genes": 20})
    elif recipe == "slide_seq":
        defaults.update({"min_counts": 20, "min_genes": 10, "feature_method": "hvg_svg_union"})
    elif recipe in {"seqfish", "merfish", "xenium", "atera", "cosmx", "imaging"}:
        defaults.update({"min_cells": 1, "min_counts": 10, "min_genes": 5})
    elif recipe in {"pearson_residuals", "sctransform"}:
        defaults.update({"adaptive_qc": False})
    return defaults


class SpatialPreprocessor:
    """Recipe-based preprocessor for spatial transcriptomics AnnData objects.

    Examples:
        >>> import spateo as st
        >>> adata = st.read_h5ad("sample.h5ad")
        >>> st.pp.preprocess_spatial(
        ...     adata,
        ...     recipe="generic",
        ...     spatial_key="spatial",
        ...     counts_layer="counts",
        ...     n_top_genes=3000,
        ... )
        >>> st.pp.preprocess_spatial(
        ...     adata,
        ...     recipe="visium",
        ...     spatial_key="spatial",
        ...     in_tissue_key="in_tissue",
        ...     feature_method="hvg_svg_union",
        ... )
        >>> st.pp.preprocess_spatial(adata, recipe="xenium", spatial_key="spatial", feature_method="hvg")
    """

    def __init__(
        self,
        recipe: str = "auto",
        spatial_key: str = "spatial",
        layer: Optional[str] = None,
        counts_layer: str = "counts",
        normalized_layer: str = "norm",
        log1p_layer: str = "log1p_norm",
        library_key: Optional[str] = None,
        sample_key: Optional[str] = None,
        pp_key: str = "pp",
        copy_raw_counts: bool = True,
    ):
        self.recipe = recipe
        self.spatial_key = spatial_key
        self.layer = layer
        self.counts_layer = counts_layer
        self.normalized_layer = normalized_layer
        self.log1p_layer = log1p_layer
        self.library_key = library_key
        self.sample_key = sample_key
        self.pp_key = pp_key
        self.copy_raw_counts = copy_raw_counts

    def preprocess_adata(
        self,
        adata: AnnData,
        recipe: Recipe = "auto",
        spatial_key: str = "spatial",
        layer: Optional[str] = None,
        counts_layer: str = "counts",
        normalized_layer: str = "norm",
        log1p_layer: str = "log1p_norm",
        library_key: Optional[str] = None,
        sample_key: Optional[str] = None,
        in_tissue_key: str = "in_tissue",
        target_sum: float = 1e4,
        min_counts: Optional[int] = None,
        max_counts: Optional[int] = None,
        min_genes: Optional[int] = None,
        max_genes: Optional[int] = None,
        max_pct_mt: Optional[float] = None,
        min_cells: int = 3,
        min_gene_counts: Optional[int] = None,
        n_top_genes: int = 3000,
        feature_method: Literal["hvg", "svg", "hvg_svg_union", "hvg_svg_intersection", "all"] = "hvg",
        build_spatial_graph: bool = True,
        build_expression_graph: bool = False,
        run_pca: bool = True,
        n_pca_components: int = 50,
        keep_filtered: bool = False,
        local_qc: bool = False,
        filter_local_outliers: bool = False,
        adaptive_qc: Optional[bool] = None,
        adaptive_upper: bool = False,
        nmads: float = 3.0,
        spatial_n_neighbors: Optional[int] = None,
        spatial_radius: Optional[float] = None,
        spatial_delaunay: bool = False,
        validate_counts: Literal["warn", "error", "ignore"] = "warn",
        inplace: bool = True,
    ) -> Optional[AnnData]:
        """Preprocess a spatial AnnData object using a named recipe.

        Args:
            adata: Input AnnData object.
            recipe: Spatial recipe name.
            spatial_key: Key in ``adata.obsm`` containing coordinates.
            layer: Input layer; ``None`` means ``adata.X``.
            counts_layer: Raw counts layer.
            normalized_layer: Normalized output layer.
            log1p_layer: Log-normalized output layer.
            library_key: Optional library/slice key.
            sample_key: Optional sample key.
            in_tissue_key: Visium tissue flag key.
            target_sum: Target total counts for normalization.
            min_counts: Minimum counts per observation.
            max_counts: Maximum counts per observation.
            min_genes: Minimum detected genes per observation.
            max_genes: Maximum detected genes per observation.
            max_pct_mt: Maximum mitochondrial percentage.
            min_cells: Minimum observations per gene.
            min_gene_counts: Minimum counts per gene.
            n_top_genes: Number of features to select.
            feature_method: Feature selection method.
            build_spatial_graph: Whether to build a spatial graph.
            build_expression_graph: Whether to build an expression graph.
            run_pca: Whether to run PCA.
            n_pca_components: Requested PCA components.
            keep_filtered: Whether to annotate instead of subset filtered data.
            local_qc: Whether to flag local spatial QC outliers.
            inplace: If ``True``, modify ``adata`` in place.

        Returns:
            Updated AnnData when ``inplace=False``; otherwise ``None``.
        """
        if recipe == "auto" and self.recipe != "auto":
            recipe = self.recipe
        if spatial_key == "spatial" and self.spatial_key != "spatial":
            spatial_key = self.spatial_key
        layer = self.layer if layer is None else layer
        if counts_layer == "counts" and self.counts_layer != "counts":
            counts_layer = self.counts_layer
        if normalized_layer == "norm" and self.normalized_layer != "norm":
            normalized_layer = self.normalized_layer
        if log1p_layer == "log1p_norm" and self.log1p_layer != "log1p_norm":
            log1p_layer = self.log1p_layer
        library_key = library_key if library_key is not None else self.library_key
        sample_key = sample_key if sample_key is not None else self.sample_key
        recipe = _infer_recipe(adata, recipe)
        defaults = _recipe_defaults(recipe)

        use_in_tissue = defaults["use_in_tissue"]
        coord_type = defaults["coord_type"]
        n_neighbors = spatial_n_neighbors or defaults["n_neighbors"]
        radius = spatial_radius if spatial_radius is not None else defaults["radius"]
        if min_cells == 3:
            min_cells = defaults["min_cells"]
        if min_counts is None:
            min_counts = defaults["min_counts"]
        if min_genes is None:
            min_genes = defaults["min_genes"]
        if max_pct_mt is None:
            max_pct_mt = defaults["max_pct_mt"]
        if adaptive_qc is None:
            adaptive_qc = defaults["adaptive_qc"]
        if target_sum == 1e4:
            target_sum = defaults["target_sum"]
        if feature_method == "hvg":
            feature_method = defaults["feature_method"]
        if run_pca is True:
            run_pca = defaults["run_pca"]

        target = adata if inplace else adata.copy()
        n_obs_before, n_vars_before = target.n_obs, target.n_vars
        steps: list[str] = []
        logger.log_time()
        logger.info(f"Starting spatial preprocessing recipe `{recipe}`...")

        standardize_spatial_adata(
            target,
            spatial_key=spatial_key,
            layer=layer,
            counts_layer=counts_layer,
            library_key=library_key,
            sample_key=sample_key,
            copy_raw_counts=self.copy_raw_counts,
            validate_counts=validate_counts,
            inplace=True,
        )
        steps.append("standardize_spatial_adata")

        calculate_spatial_qc(target, layer=counts_layer, spatial_key=spatial_key, inplace=True)
        steps.append("calculate_spatial_qc")

        if local_qc:
            spatial_neighbors(
                target,
                spatial_key=spatial_key,
                library_key=library_key,
                coord_type=coord_type,
                n_neighbors=n_neighbors,
                radius=radius,
                delaunay=spatial_delaunay,
                inplace=True,
            )
            steps.append("spatial_neighbors")
        if local_qc:
            flag_local_qc_outliers(target)
            steps.append("flag_local_qc_outliers")

        filter_spots(
            target,
            min_counts=min_counts,
            max_counts=max_counts,
            min_genes=min_genes,
            max_genes=max_genes,
            max_pct_mt=max_pct_mt,
            use_in_tissue=use_in_tissue,
            in_tissue_key=in_tissue_key,
            library_key=library_key,
            adaptive=bool(adaptive_qc),
            adaptive_upper=adaptive_upper,
            nmads=nmads,
            exclude_local_outliers=filter_local_outliers,
            keep_filtered=keep_filtered,
            inplace=True,
        )
        steps.append("filter_spots")

        filter_genes_by_spatial_qc(
            target,
            layer=counts_layer,
            min_cells=min_cells,
            min_counts=min_gene_counts,
            library_key=library_key,
            keep_filtered=keep_filtered,
            inplace=True,
        )
        steps.append("filter_genes_by_spatial_qc")

        if build_spatial_graph:
            spatial_neighbors(
                target,
                spatial_key=spatial_key,
                library_key=library_key,
                coord_type=coord_type,
                n_neighbors=n_neighbors,
                radius=radius,
                delaunay=spatial_delaunay,
                inplace=True,
            )
            steps.append("spatial_neighbors")

        pca_layer = log1p_layer
        if recipe == "pearson_residuals":
            pearson_residuals(target, layer=counts_layer, out_layer="pearson_residuals", inplace=True)
            pca_layer = "pearson_residuals"
            steps.append("pearson_residuals")
        elif recipe == "sctransform":
            run_sctransform(target, layer=counts_layer, out_layer="sctransform", inplace=True)
            pca_layer = "sctransform"
            steps.append("sctransform")
        else:
            calculate_size_factors(
                target,
                layer=counts_layer,
                library_key=library_key,
                target_sum=target_sum,
                inplace=True,
            )
            steps.append("calculate_size_factors")
            normalize_total(
                target,
                layer=counts_layer,
                out_layer=normalized_layer,
                target_sum=target_sum,
                inplace=True,
            )
            steps.append("normalize_total")
            log1p_layer_fn = globals()["log1p_layer"]
            log1p_layer_fn(target, layer=normalized_layer, out_layer=log1p_layer, set_X=True, inplace=True)
            steps.append("log1p_layer")

        select_spatial_features(
            target,
            layer=pca_layer,
            method=feature_method,
            n_top_genes=n_top_genes,
            batch_key=sample_key or library_key,
            inplace=True,
        )
        steps.append("select_spatial_features")

        if run_pca:
            pca(target, layer=pca_layer, n_pca_components=n_pca_components, inplace=True)
            steps.append("pca")
        if build_expression_graph:
            expression_neighbors(target, basis=SKM.OBSM_X_PCA_KEY, inplace=True)
            steps.append("expression_neighbors")
        SKM.init_uns_pp_namespace(target)
        SKM.init_uns_spatial_namespace(target)
        target.uns[SKM.UNS_PP_KEY]["spatial_preprocess"] = {
            "recipe": recipe,
            "spatial_key": spatial_key,
            "counts_layer": counts_layer,
            "normalized_layer": normalized_layer,
            "log1p_layer": log1p_layer,
            "library_key": library_key,
            "sample_key": sample_key,
            "n_obs_before": n_obs_before,
            "n_obs_after": target.n_obs,
            "n_vars_before": n_vars_before,
            "n_vars_after": target.n_vars,
            "params": {
                "target_sum": target_sum,
                "min_counts": min_counts,
                "max_counts": max_counts,
                "min_genes": min_genes,
                "max_genes": max_genes,
                "max_pct_mt": max_pct_mt,
                "min_cells": min_cells,
                "min_gene_counts": min_gene_counts,
                "n_top_genes": n_top_genes,
                "feature_method": feature_method,
                "build_spatial_graph": build_spatial_graph,
                "build_expression_graph": build_expression_graph,
                "run_pca": run_pca,
                "n_pca_components": n_pca_components,
                "keep_filtered": keep_filtered,
                "local_qc": local_qc,
                "filter_local_outliers": filter_local_outliers,
                "adaptive_qc": adaptive_qc,
                "adaptive_upper": adaptive_upper,
                "nmads": nmads,
                "spatial_n_neighbors": n_neighbors,
                "spatial_radius": radius,
                "spatial_delaunay": spatial_delaunay,
                "validate_counts": validate_counts,
            },
            "steps": steps,
        }
        target.uns[SKM.UNS_SPATIAL_KEY].setdefault(SKM.UNS_SPATIAL_QC_KEY, {})
        target.uns[SKM.UNS_SPATIAL_KEY][SKM.UNS_SPATIAL_QC_KEY].update(
            {
                "metrics": [
                    SKM.OBS_TOTAL_COUNTS_KEY,
                    SKM.OBS_N_GENES_BY_COUNTS_KEY,
                    SKM.OBS_PCT_COUNTS_MT_KEY,
                ],
                "filter_key": SKM.OBS_PASS_SPATIAL_QC_KEY,
                "local_qc": local_qc,
            }
        )
        target.uns[SKM.UNS_SPATIAL_KEY]["neighbors"] = {
            "spatial_key": spatial_key,
            "library_key": library_key,
            "coord_type": coord_type,
            "n_neighbors": n_neighbors,
            "radius": radius,
            "delaunay": spatial_delaunay,
        }
        logger.finish_progress(progress_name="spatial preprocessing")
        return None if inplace else target


Preprocessor = SpatialPreprocessor


def preprocess_spatial(adata: AnnData, **kwargs: object) -> Optional[AnnData]:
    """Function interface for :class:`SpatialPreprocessor`.

    Examples:
        >>> import spateo as st
        >>> st.pp.preprocess_spatial(adata, recipe="generic", spatial_key="spatial")

    Args:
        adata: Input AnnData object.
        **kwargs: Passed to :meth:`SpatialPreprocessor.preprocess_adata`.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    call_kwargs = dict(kwargs)
    copy_raw_counts = call_kwargs.pop("copy_raw_counts", True)
    pp_key = call_kwargs.pop("pp_key", "pp")
    preprocessor = SpatialPreprocessor(
        recipe=kwargs.get("recipe", "auto"),
        spatial_key=kwargs.get("spatial_key", "spatial"),
        layer=kwargs.get("layer", None),
        counts_layer=kwargs.get("counts_layer", "counts"),
        normalized_layer=kwargs.get("normalized_layer", "norm"),
        log1p_layer=kwargs.get("log1p_layer", "log1p_norm"),
        library_key=kwargs.get("library_key", None),
        sample_key=kwargs.get("sample_key", None),
        pp_key=pp_key,
        copy_raw_counts=copy_raw_counts,
    )
    return preprocessor.preprocess_adata(adata, **call_kwargs)
