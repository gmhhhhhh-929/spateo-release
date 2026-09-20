#!/usr/bin/env python3
"""Reproducible cross-format contract benchmark; not population-level accuracy.

Creates small synthetic inputs in a temporary directory, then removes them.
Only aggregate/per-call reports are persisted to --output. No original data used.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spateo.io import read_spatial

TECHNOLOGIES = (
    "visium",
    "visium_hd_bin",
    "visium_hd_cellseg",
    "xenium",
    "atera",
    "merfish",
    "seqfish",
    "nanostring",
    "slideseq",
    "starmap_plus",
    "bgi",
)
SCENARIOS = ("valid", "permuted_metadata", "invalid_expression", "invalid_coordinate", "missing_core")


def build(root, tech, seed, scenario):
    """Serialize externally defined truth through independently constructed layouts."""
    rng = np.random.default_rng(seed)
    n, g = int(rng.integers(8, 30)), int(rng.integers(5, 20))
    truth = rng.poisson(2, (n, g)).astype(float)
    truth[:, 0] += 1
    xy = np.column_stack([np.arange(n) * 10 + 20, rng.integers(10, 100, n)]).astype(float)
    ids = [f"c{i}" for i in range(n)]
    genes = [f"G{i}" for i in range(g)]
    X, coords = truth.copy(), xy.copy()
    if scenario == "invalid_expression":
        X[-1, -1] = np.nan
    if scenario == "invalid_coordinate":
        coords[-1, 0] = np.nan
    if tech == "visium_hd_bin":
        root = root / "binned_outputs" / ("square_008um" if seed % 2 else "square_016um")
    root.mkdir(parents=True)
    counts, meta = None, None
    h5tech = tech in TECHNOLOGIES[:5]
    if h5tech:
        name = "filtered_feature_bc_matrix.h5"
        if tech in ("xenium", "atera"):
            name = "cell_feature_matrix.h5"
        elif tech == "visium_hd_cellseg":
            name = "filtered_feature_cell_matrix.h5"
            ids = [f"cellid_{i:09d}-1" for i in range(1, n + 1)]
        counts = root / name
        with h5py.File(counts, "w") as f:
            m = f.create_group("matrix")
            csc = sparse.csc_matrix(X.T)
            for key, arr in dict(
                data=csc.data,
                indices=csc.indices,
                indptr=csc.indptr,
                shape=csc.shape,
                barcodes=np.asarray(ids, dtype="S"),
            ).items():
                m.create_dataset(key, data=arr)
            feat = m.create_group("features")
            feat.create_dataset("id", data=np.asarray([f"gene{i}" for i in range(g)], dtype="S"))
            feat.create_dataset("name", data=np.asarray(genes, dtype="S"))
        if tech in ("visium", "visium_hd_bin"):
            meta = root / "spatial/tissue_positions.csv"
            meta.parent.mkdir()
            frame = pd.DataFrame(
                dict(
                    barcode=ids,
                    in_tissue=1,
                    array_row=np.arange(n),
                    array_col=np.arange(n),
                    pxl_row_in_fullres=coords[:, 1],
                    pxl_col_in_fullres=coords[:, 0],
                )
            )
        elif tech in ("xenium", "atera"):
            meta = root / "cells.csv"
            frame = pd.DataFrame(dict(cell_id=ids, x_centroid=coords[:, 0], y_centroid=coords[:, 1]))
            if tech == "atera":
                (root / "experiment.xenium").write_text(json.dumps({"platform": "Atera", "assay": "human WTA"}))
        else:
            meta = root / "cell_segmentations.geojson"
            feats = []
            for i, (x, y) in enumerate(coords):
                feats.append(
                    dict(
                        type="Feature",
                        properties={"cell_id": i + 1},
                        geometry=dict(
                            type="Polygon",
                            coordinates=[
                                [[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1], [x - 1, y - 1]]
                            ],
                        ),
                    )
                )
            if scenario == "permuted_metadata":
                rng.shuffle(feats)
            meta.write_text(json.dumps(dict(type="FeatureCollection", features=feats)))
            frame = None
    elif tech == "bgi":
        counts = meta = root / "sample.gem"
        rows = []
        for i in range(n):
            for j in range(g):
                rows.append((genes[j], coords[i, 0], coords[i, 1], X[i, j]))
        frame = pd.DataFrame(rows, columns=["geneID", "x", "y", "MIDCount"])
        # First occurrence of each gene fixes gene order; permute within each gene.
        if scenario == "permuted_metadata":
            frame = pd.concat([frame[frame.geneID == gene].sample(frac=1, random_state=seed) for gene in genes])
        if scenario == "missing_core":
            frame = frame.drop(columns="x")
        frame.to_csv(counts, sep="\t", index=False)
        return root, truth, xy, [f"{int(x)}_{int(y)}" for x, y in xy], genes
    else:
        frame = pd.DataFrame(dict(cell_id=ids, center_x=coords[:, 0], center_y=coords[:, 1]))
        expression = pd.DataFrame(X, columns=genes)
        expression.insert(0, "cell_id", ids)
        names = {
            "merfish": ("cell_by_gene_S1.csv", "cell_metadata_S1.csv"),
            "seqfish": ("SG_CxG_S1.csv", "SG_CellCoordinates_S1.csv"),
            "nanostring": ("sample_exprMat_file.csv", "sample_metadata_file.csv"),
            "slideseq": ("MappedDGEForR.csv", "BeadLocationsForR.csv"),
            "starmap_plus": ("sample_raw_expression_pd.csv", "sample_spatial.csv"),
        }
        c, m = names[tech]
        counts, meta = root / c, root / m
        if tech == "seqfish":
            frame = frame.rename(columns={"cell_id": "label"})
        elif tech == "nanostring":
            expression["fov"] = np.arange(n) % 3
            frame["fov"] = np.arange(n) % 3
            frame = frame.rename(columns={"center_x": "CenterX_local_px", "center_y": "CenterY_local_px"})
            ids = [f"{c}_{i % 3}" for i, c in enumerate(ids)]
        elif tech == "slideseq":
            expression = pd.DataFrame(X.T, columns=ids)
            expression.insert(0, "gene", genes)
            frame = frame.rename(columns={"cell_id": "barcodes", "center_x": "xcoord", "center_y": "ycoord"})
        elif tech == "starmap_plus":
            expression = pd.DataFrame(X.T, columns=ids)
            expression.insert(0, "GENE", genes)
            frame = frame.rename(columns={"cell_id": "NAME", "center_x": "X", "center_y": "Y"})
        expression.to_csv(counts, index=False)
    if frame is not None:
        if scenario == "permuted_metadata":
            frame = frame.sample(frac=1, random_state=seed)
        if tech == "starmap_plus":
            frame = pd.concat([pd.DataFrame([dict(NAME="TYPE", X="numeric", Y="numeric")]), frame])
        frame.to_csv(meta, index=False)
    if scenario == "missing_core":
        meta.unlink()
    return root, truth, xy, ids, genes


def summarize(rows, repeats):
    unique = [r for r in rows if r["repeat"] == 1]
    positives = [r for r in unique if r["expected_valid"]]
    negatives = [r for r in unique if not r["expected_valid"]]
    return dict(
        unique_cases=len(unique),
        positive_cases=len(positives),
        negative_cases=len(negatives),
        calls=len(rows),
        identification_correct=sum(r["identified_correct"] for r in positives),
        identification_accuracy=sum(r["identified_correct"] for r in positives) / len(positives),
        ready_positive=sum(r["status"] == "ok" for r in positives),
        positive_read_rate=sum(r["status"] == "ok" for r in positives) / len(positives),
        exact_content_pass=sum(r["content_exact"] for r in positives),
        content_success_rate=sum(r["content_exact"] for r in positives) / len(positives),
        correctly_rejected=sum(r["passed"] for r in negatives),
        negative_rejection_rate=sum(r["passed"] for r in negatives) / len(negatives),
        all_criteria_pass=sum(r["passed"] for r in unique),
        unexpected_ready_negative=sum(r["status"] == "ok" for r in negatives),
        repeat_consistent_cases=sum(
            len({r["fingerprint"] for r in rows if r["case_id"] == u["case_id"]}) == 1 for u in unique
        ),
        repetitions_per_case=repeats,
        median_seconds=float(np.median([r["seconds"] for r in rows])),
        p95_seconds=float(np.quantile([r["seconds"] for r in rows], 0.95)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="spateo-contract-benchmark-") as tmp:
        for round_id in range(1, args.rounds + 1):
            seed = 20260920 + round_id
            for tech in TECHNOLOGIES:
                for scenario in SCENARIOS:
                    case = f"{tech}/round_{round_id}/{scenario}"
                    root, truth, xy, ids, genes = build(Path(tmp) / case, tech, seed, scenario)
                    valid = scenario in SCENARIOS[:2]
                    for repeat in range(1, args.repeats + 1):
                        start = time.perf_counter()
                        result = read_spatial(root, load_images=False)
                        entries = list(result.datasets.values())
                        prediction = "|".join(sorted({e.technology for e in entries if e.status == "ready"}))
                        exact = False
                        if result.status == "ok" and len(entries) == 1:
                            a = result.adata
                            exact = (
                                a.obs_names.tolist() == ids
                                and a.var_names.tolist() == genes
                                and np.array_equal(a.X.toarray(), truth)
                                and np.array_equal(a.obsm["spatial"], xy)
                            )
                        identified = result.status == "ok" and len(entries) == 1 and prediction == tech
                        rejected = result.status == "failed" and not any(e.adata is not None for e in entries)
                        digest = dict(
                            status=result.status,
                            prediction=prediction,
                            content_exact=exact,
                            entry_states=[e.status for e in entries],
                        )
                        rows.append(
                            dict(
                                case_id=case,
                                round=round_id,
                                seed=seed,
                                technology=tech,
                                scenario=scenario,
                                repeat=repeat,
                                expected_valid=valid,
                                status=result.status,
                                predicted_technology=prediction,
                                identified_correct=identified,
                                content_exact=exact,
                                passed=(identified and exact) if valid else rejected,
                                seconds=time.perf_counter() - start,
                                fingerprint=hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest(),
                                diagnostics=[
                                    d["message"].replace(str(Path(tmp)), "<fixture>")
                                    for e in entries
                                    for d in e.diagnostics
                                    if d["severity"] == "error"
                                ],
                            )
                        )
            print(f"Round {round_id}/{args.rounds}: {len(rows)} calls recorded", flush=True)
    report = dict(
        benchmark="synthetic format contracts, not independent biological samples",
        rounds=args.rounds,
        seeds=[20260920 + i for i in range(1, args.rounds + 1)],
        scenarios=SCENARIOS,
        repeats=args.repeats,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        python=platform.python_version(),
        overall=summarize(rows, args.repeats),
        per_technology={t: summarize([r for r in rows if r["technology"] == t], args.repeats) for t in TECHNOLOGIES},
        confusion_matrix={
            t: {
                p: sum(
                    r["technology"] == t
                    and r["expected_valid"]
                    and r["repeat"] == 1
                    and (r["predicted_technology"] or "rejected") == p
                    for r in rows
                )
                for p in (*TECHNOLOGIES, "rejected")
            }
            for t in TECHNOLOGIES
        },
        calls=rows,
    )
    (args.output / "synthetic_benchmark.json").write_text(json.dumps(report, indent=2))
    with (args.output / "synthetic_calls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["overall"], indent=2))
    if not all(r["passed"] for r in rows):
        raise SystemExit("Some benchmark cases failed; inspect the recorded results")


if __name__ == "__main__":
    main()
