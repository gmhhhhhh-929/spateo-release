from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import spateo as st
from spateo.preprocessing.slice_quality import (
    HighConfidencePolicy,
    ReviewEvidenceTier,
    SliceQCConfig,
    _metric_anomaly,
    _score_metrics,
    add_multiscale_exclusion_evidence,
    apply_high_confidence_policy,
    calculate_slice_quality,
    evaluate_paired_simulation,
    scan_h5ad_collection,
    scan_h5ad_series,
    simulate_slice_quality_artifacts,
    write_high_confidence_outputs,
    write_slice_quality_collection_outputs,
    write_slice_quality_outputs,
)


def make_series(seed: int = 7, prefix: str = "S") -> ad.AnnData:
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    coordinates: list[np.ndarray] = []
    matrices: list[sparse.csr_matrix] = []
    categories = [f"{prefix}{index:02d}" for index in range(9)]
    for index, label in enumerate(categories):
        n_obs = 70 + index * 3
        theta = rng.uniform(0, 2 * np.pi, n_obs)
        radius = np.sqrt(rng.uniform(0, 1, n_obs))
        coordinates.append(np.column_stack([radius * np.cos(theta), 1.4 * radius * np.sin(theta)]))
        matrices.append(sparse.csr_matrix(rng.poisson(0.35 + 0.02 * index, size=(n_obs, 80))))
        labels.extend([label] * n_obs)
    matrix = sparse.vstack(matrices).tocsr()
    result = ad.AnnData(matrix)
    result.obs["slice_id"] = pd.Categorical(labels, categories=categories, ordered=True)
    result.obsm["spatial"] = np.vstack(coordinates)
    result.layers["counts"] = matrix.copy()
    return result


def test_public_preprocessing_exports() -> None:
    assert st.pp.SliceQCConfig is SliceQCConfig
    assert st.pp.ReviewEvidenceTier is ReviewEvidenceTier
    assert st.pp.calculate_slice_quality is calculate_slice_quality
    assert st.pp.scan_h5ad_collection is scan_h5ad_collection


def test_in_memory_scan_is_non_destructive() -> None:
    adata = make_series()
    original_coordinates = adata.obsm["spatial"].copy()
    original_obs_names = adata.obs_names.copy()
    metrics = calculate_slice_quality(
        adata,
        slice_key="slice_id",
        spatial_key="spatial",
        layer="counts",
        config=SliceQCConfig(window=3),
    )
    assert len(metrics) == 9
    assert set(metrics["recommendation"]) <= {"keep", "review", "exclude"}
    np.testing.assert_array_equal(adata.obsm["spatial"], original_coordinates)
    assert adata.obs_names.equals(original_obs_names)
    assert "slice_quality_qc" not in adata.uns


def test_scan_one_multislice_h5ad_and_natural_per_slice_files() -> None:
    adata = make_series()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        combined = root / "combined.h5ad"
        adata.write_h5ad(combined)
        one = scan_h5ad_series([combined], slice_key="slice_id", spatial_key="spatial", layer="counts")
        paths: list[Path] = []
        for slice_id in ["S00", "S01", "S02", "S03", "S04"]:
            subset = adata[adata.obs["slice_id"].astype(str) == slice_id].copy()
            subset.obs.drop(columns="slice_id", inplace=True)
            path = root / f"{slice_id}.h5ad"
            subset.write_h5ad(path)
            paths.append(path)
        many = scan_h5ad_series(
            list(reversed(paths)),
            slice_key="auto",
            spatial_key="spatial",
            layer="counts",
        )
        explicit = scan_h5ad_series(
            list(reversed(paths)),
            slice_key="auto",
            spatial_key="spatial",
            layer="counts",
            sort_inputs=False,
        )
    assert len(one.metrics) == 9
    assert many.metrics["slice_id"].tolist() == ["S00", "S01", "S02", "S03", "S04"]
    assert explicit.metrics["slice_id"].tolist() == ["S04", "S03", "S02", "S01", "S00"]


def test_collection_keeps_independent_dataset_boundaries() -> None:
    first = make_series(seed=2, prefix="A")
    second = make_series(seed=3, prefix="B")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_path, second_path = root / "first.h5ad", root / "second.h5ad"
        first.write_h5ad(first_path)
        second.write_h5ad(second_path)
        results = scan_h5ad_collection(
            {"study/first": first_path, "study/second": second_path},
            config=SliceQCConfig(window=3),
            dataset_options={
                "study/first": {
                    "slice_key": "slice_id",
                    "spatial_key": "spatial",
                    "layer": "counts",
                },
                "study/second": {
                    "slice_key": "slice_id",
                    "spatial_key": "spatial",
                    "layer": "counts",
                },
            },
        )
        outputs = write_slice_quality_collection_outputs(results, root / "outputs", write_display_payload=True)
        assert (
            Path(outputs["study/first"]["metrics"]).parent.resolve() == (root / "outputs" / "study" / "first").resolve()
        )
    assert list(results) == ["study/first", "study/second"]
    assert results["study/first"].metrics["slice_id"].str.startswith("A").all()
    assert results["study/second"].metrics["slice_id"].str.startswith("B").all()


def test_consecutive_expression_losses_use_wide_support() -> None:
    n_slices = 15
    counts = 150 + 6 * np.arange(n_slices, dtype=float)
    genes = 80 + 3 * np.arange(n_slices, dtype=float)
    counts[6:10] *= 0.12
    genes[6:10] *= 0.25
    metrics = pd.DataFrame(
        {
            "slice_id": [f"S{index:02d}" for index in range(n_slices)],
            "median_total_counts": counts,
            "median_n_genes": genes,
        }
    )
    scored = _score_metrics(metrics, None, {}, SliceQCConfig(window=3), 3)
    assert set(scored.loc[6:9, "recommendation"]) <= {"review", "exclude"}
    assert (scored.loc[6:9, "support_window_size"] == 9).all()


def test_small_absolute_fraction_is_not_amplified() -> None:
    values = np.asarray([0.0, 0.0, 0.004, 0.0, 0.0], dtype=float)
    score, _, _ = _metric_anomaly(values, 3, "high", 0.05, 0.32)
    assert float(score[2]) < 0.10


def test_multiscale_evidence_records_each_window() -> None:
    metrics = pd.DataFrame(
        {
            "slice_id": [f"S{index:02d}" for index in range(9)],
            "n_locations": [100, 104, 102, 99, 15, 103, 101, 105, 100],
            "cell_density": [1.0, 1.02, 1.01, 0.99, 0.12, 1.01, 1.0, 1.03, 1.0],
            "median_total_counts": [500, 510, 495, 505, 80, 500, 515, 505, 500],
            "median_n_genes": [250, 255, 248, 252, 60, 250, 258, 252, 250],
            "largest_component_fraction": [
                0.95,
                0.96,
                0.95,
                0.94,
                0.35,
                0.96,
                0.95,
                0.96,
                0.95,
            ],
            "fragmentation": [0.05, 0.04, 0.05, 0.06, 0.65, 0.04, 0.05, 0.04, 0.05],
            "knn_tail_ratio": [1.0, 1.0, 1.0, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0],
        }
    )
    enriched = add_multiscale_exclusion_evidence(metrics, windows=(3, 5, 7))
    assert enriched.loc[4, "adaptive_windows_tested"] == "3|5|7"
    assert [item["window"] for item in json.loads(enriched.loc[4, "adaptive_window_details"])] == [3, 5, 7]


def test_two_stage_policy_resolves_only_stable_multidomain_review() -> None:
    metrics = pd.DataFrame(
        {
            "slice_id": ["stable", "single-domain", "unstable"],
            "quality_anomaly_score": [0.677, 0.713, 0.680],
            "recommendation": ["exclude", "review", "exclude"],
            "window_context": ["two_sided"] * 3,
            "partial_structure_protection": [False] * 3,
            "corroborating_domains": [2, 1, 2],
            "score_confidence": [1.0] * 3,
            "adaptive_window_stability": [1.0, 1.0, 2 / 3],
            "adaptive_min_score_across_windows": [0.648, 0.690, 0.650],
            "density_domain_score": [0.933, 0.978, 0.920],
            "expression_domain_score": [0.620, 0.445, 0.610],
            "damage_domain_score": [0.326, 0.342, 0.270],
            "continuity_domain_score": [0.354, 0.150, 0.100],
        }
    )
    policy = HighConfidencePolicy(
        keep_max_score=0.129,
        exclude_min_score=0.700,
        adaptive_exclude_min_score=0.640,
        adaptive_min_corroborating_domains=2,
        adaptive_severe_domain_threshold=0.850,
        adaptive_min_window_stability=0.900,
        adaptive_min_score_confidence=0.900,
        unresolved_action="keep",
    )
    applied = apply_high_confidence_policy(metrics, policy).set_index("slice_id")
    assert applied.loc["stable", "threshold_band"] == "review"
    assert applied.loc["stable", "review_resolution"] == "exclude"
    assert applied.loc["stable", "final_call"] == "exclude"
    assert applied.loc["single-domain", "review_resolution"] == "keep"
    assert applied.loc["unstable", "final_call"] == "keep"


def test_tiered_review_resolver_covers_the_full_review_interval() -> None:
    rows = []
    examples = [
        ("low", 0.20, 4, 0.97),
        ("mid", 0.45, 3, 0.92),
        ("high", 0.65, 2, 0.87),
        ("low-insufficient", 0.22, 3, 0.97),
    ]
    for slice_id, score, domains, maximum in examples:
        details = [
            {
                "window": window,
                "score": score,
                "detector_call": "review",
                "corroborating_domains": domains,
                "maximum_domain_score": maximum,
                "window_context": "two_sided",
                "partial_structure_protection": False,
            }
            for window in (3, 5, 7)
        ]
        rows.append(
            {
                "slice_id": slice_id,
                "quality_anomaly_score": score,
                "recommendation": "review",
                "window_context": "two_sided",
                "partial_structure_protection": False,
                "corroborating_domains": domains,
                "score_confidence": 1.0,
                "adaptive_window_details": json.dumps(details),
                "density_domain_score": maximum,
                "expression_domain_score": 0.80 if domains >= 2 else 0.0,
                "damage_domain_score": 0.65 if domains >= 3 else 0.0,
                "continuity_domain_score": 0.55 if domains >= 4 else 0.0,
            }
        )
    policy = HighConfidencePolicy(
        keep_max_score=0.129,
        exclude_min_score=0.700,
        review_exclusion_tiers=(
            ReviewEvidenceTier("low", 0.129, 0.35, 4, 0.95, 1.0, 0.90, "candidate"),
            ReviewEvidenceTier("mid", 0.35, 0.60, 3, 0.90, 1.0, 0.90, "candidate"),
            ReviewEvidenceTier("high", 0.60, None, 2, 0.85, 1.0, 0.90, "candidate"),
        ),
        unresolved_action="keep",
    )
    applied = apply_high_confidence_policy(pd.DataFrame(rows), policy).set_index("slice_id")
    assert applied.loc["low", "final_call"] == "exclude"
    assert applied.loc["mid", "final_call"] == "exclude"
    assert applied.loc["high", "final_call"] == "exclude"
    assert applied.loc["low-insufficient", "final_call"] == "keep"
    assert applied.loc["low", "review_resolution_tier"] == "low"
    assert applied.loc["mid", "review_resolution_tier"] == "mid"
    assert applied.loc["high", "review_resolution_tier"] == "high"


def test_tiered_review_resolver_rejects_score_band_gaps() -> None:
    policy = HighConfidencePolicy(
        keep_max_score=0.129,
        exclude_min_score=0.700,
        review_exclusion_tiers=(
            ReviewEvidenceTier("low", 0.129, 0.35, 4, 0.95),
            ReviewEvidenceTier("high", 0.40, None, 2, 0.85),
        ),
        unresolved_action="keep",
    )
    metrics = pd.DataFrame(
        {
            "slice_id": ["S01"],
            "quality_anomaly_score": [0.5],
            "recommendation": ["review"],
            "corroborating_domains": [2],
            "score_confidence": [1.0],
            "adaptive_window_details": ["[]"],
            "density_domain_score": [0.9],
            "expression_domain_score": [0.5],
            "damage_domain_score": [0.0],
            "continuity_domain_score": [0.0],
        }
    )
    with pytest.raises(ValueError, match="contiguous score bands"):
        apply_high_confidence_policy(metrics, policy)


def test_tiered_review_resolver_rejects_weaker_low_score_evidence() -> None:
    policy = HighConfidencePolicy(
        keep_max_score=0.129,
        exclude_min_score=0.700,
        review_exclusion_tiers=(
            ReviewEvidenceTier("low", 0.129, 0.40, 2, 0.85),
            ReviewEvidenceTier("high", 0.40, None, 3, 0.90),
        ),
        unresolved_action="keep",
    )
    metrics = pd.DataFrame(
        {
            "slice_id": ["S01"],
            "quality_anomaly_score": [0.5],
            "recommendation": ["review"],
            "corroborating_domains": [3],
            "score_confidence": [1.0],
            "adaptive_window_details": ["[]"],
            "density_domain_score": [0.9],
            "expression_domain_score": [0.8],
            "damage_domain_score": [0.7],
            "continuity_domain_score": [0.0],
        }
    )
    with pytest.raises(ValueError, match="lower-score review tier"):
        apply_high_confidence_policy(metrics, policy)


def test_tiered_review_resolver_records_calibrated_keep_only_band() -> None:
    details = [
        {
            "window": window,
            "score": 0.30,
            "detector_call": "review",
            "corroborating_domains": 4,
            "maximum_domain_score": 1.0,
            "window_context": "two_sided",
            "partial_structure_protection": False,
        }
        for window in (3, 5, 7)
    ]
    metrics = pd.DataFrame(
        {
            "slice_id": ["lower-review"],
            "quality_anomaly_score": [0.30],
            "recommendation": ["review"],
            "window_context": ["two_sided"],
            "partial_structure_protection": [False],
            "corroborating_domains": [4],
            "score_confidence": [1.0],
            "adaptive_window_details": [json.dumps(details)],
            "density_domain_score": [1.0],
            "expression_domain_score": [1.0],
            "damage_domain_score": [1.0],
            "continuity_domain_score": [1.0],
        }
    )
    policy = HighConfidencePolicy(
        keep_max_score=0.129,
        exclude_min_score=0.700,
        review_exclusion_tiers=(
            ReviewEvidenceTier("mid_low", 0.129, 0.54, enable_exclude=False),
            ReviewEvidenceTier("high", 0.54, None, 2, 0.80),
        ),
        unresolved_action="keep",
    )
    applied = apply_high_confidence_policy(metrics, policy).iloc[0]
    assert applied["final_call"] == "keep"
    assert applied["review_resolution_tier"] == "mid_low"
    assert "calibrated keep-only" in applied["review_resolution_reason"]


def test_public_binary_output_hides_internal_review_fields() -> None:
    metrics = pd.DataFrame(
        {
            "slice_id": ["S00", "S01", "S02"],
            "quality_anomaly_score": [0.02, 0.40, 0.90],
            "recommendation": ["keep", "review", "exclude"],
            "window_context": ["two_sided"] * 3,
            "partial_structure_protection": [False] * 3,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_high_confidence_outputs(
            metrics,
            HighConfidencePolicy(keep_max_score=0.10, exclude_min_score=0.70, unresolved_action="keep"),
            tmp,
        )
        public = pd.read_csv(Path(tmp) / "slice_quality_binary_calls.csv")
        audit = pd.read_csv(Path(tmp) / "slice_quality_binary_audit.csv")
    assert set(public["final_call"]) == {"keep", "exclude"}
    assert not {"threshold_band", "threshold_triage_call", "review_resolution"} & set(public)
    assert {"threshold_band", "threshold_triage_call", "review_resolution"} <= set(audit)


def test_core_writer_emits_payload_but_no_html() -> None:
    adata = make_series()
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.h5ad"
        adata.write_h5ad(source)
        result = scan_h5ad_series([source], slice_key="slice_id", spatial_key="spatial", layer="counts")
        outputs = write_slice_quality_outputs(result, Path(tmp) / "run", write_display_payload=True)
        assert Path(outputs["display_payload"]).exists()
        assert not (Path(tmp) / "run" / "slice_quality_report.html").exists()


def test_depth_thinning_and_paired_evaluation() -> None:
    adata = make_series()
    simulated = simulate_slice_quality_artifacts(
        adata,
        [{"slice_id": "S04", "kind": "depth", "rate": 0.1}],
        slice_key="slice_id",
        spatial_key="spatial",
        layer="counts",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "simulated.h5ad"
        simulated.write_h5ad(path)
        result = scan_h5ad_series(
            [path],
            config=SliceQCConfig(window=3),
            slice_key="slice_id",
            spatial_key="spatial",
            layer="slice_qc_simulated_counts",
        )
    assert result.metrics.set_index("slice_id").loc["S04", "recommendation"] in {
        "review",
        "exclude",
    }
    summary, comparison = evaluate_paired_simulation(
        pd.DataFrame(
            {
                "slice_id": ["S00", "S01"],
                "quality_anomaly_score": [0.1, 0.2],
                "recommendation": ["keep", "keep"],
            }
        ),
        pd.DataFrame(
            {
                "slice_id": ["S00", "S01"],
                "quality_anomaly_score": [0.1, 0.8],
                "recommendation": ["keep", "exclude"],
            }
        ),
        [{"slice_id": "S01", "kind": "depth"}],
    )
    assert summary["injected_newly_flagged"] == 1
    assert bool(comparison.set_index("slice_id").loc["S01", "newly_flagged"])
