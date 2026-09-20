#!/usr/bin/env python3
"""Verify available local vendor-format data separately from synthetic fixtures.

The primary root is the user's existing 0.io data directory; the secondary root
contains Vizgen_MERSCOPE and starmap_plus/starmap+ exports. Only --output is written.
Large sources are explicitly reduced to first-N observations/molecule rows, with
full features preserved. This is a format/content test, not unbiased sampling.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spateo.io import read_spatial


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def h5_truth(path):
    with h5py.File(path) as f:
        m = f["matrix"]
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        return X, m["barcodes"].asstr()[:].tolist(), m["features/name"].asstr()[:].tolist()


def prefix_h5(source, target, n):
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source) as f, h5py.File(target, "w") as out:
        m, dest = f["matrix"], out.create_group("matrix")
        stop = min(n, int(m["shape"][1]))
        ptr = m["indptr"][: stop + 1]
        for key in ("data", "indices"):
            dest.create_dataset(key, data=m[key][: int(ptr[-1])])
        dest.create_dataset("indptr", data=ptr)
        dest.create_dataset("barcodes", data=m["barcodes"][:stop])
        dest.create_dataset("shape", data=[int(m["shape"][0]), stop])
        f.copy(m["features"], dest, name="features")
        for key, value in f.attrs.items():
            out.attrs[key] = value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-root", type=Path, required=True)
    p.add_argument("--secondary-root", type=Path, required=True)
    p.add_argument("--visium-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    primary, secondary, output = args.primary_root.resolve(), args.secondary_root.resolve(), args.output.resolve()
    for root in (primary, secondary, args.visium_root.resolve()):
        if output == root or output.is_relative_to(root):
            raise ValueError("Output must be outside source data roots")
    output.mkdir(parents=True, exist_ok=True)
    source_records, cases, results = [], [], []

    def record_source(path, label):
        source_records.append(dict(label=label, file=path.name, bytes=path.stat().st_size, sha256=sha(path)))

    def add(name, tech, root, kind, X, ids, genes, xy, sources):
        cases.append(
            dict(
                name=name,
                technology=tech,
                root=root,
                input_kind=kind,
                X=sparse.csr_matrix(X),
                ids=list(map(str, ids)),
                genes=list(map(str, genes)),
                xy=np.asarray(xy, dtype=float),
            )
        )
        for file in sources:
            record_source(file, name)

    # Full, existing public Visium bundle.
    v = args.visium_root.resolve()
    X, ids, genes = h5_truth(v / "filtered_feature_bc_matrix.h5")
    pos = pd.read_csv(
        v / "spatial/tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "row", "col", "y", "x"],
        dtype={"barcode": str},
    ).set_index("barcode")
    add(
        "visium_mouse_brain_full",
        "visium",
        v,
        "complete_core",
        X,
        ids,
        genes,
        pos.loc[ids, ["x", "y"]],
        [v / "filtered_feature_bc_matrix.h5", v / "spatial/tissue_positions_list.csv"],
    )

    for folder, meta_name, name in [
        ("Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs", "cells.csv", "xenium_mouse_brain_full"),
        ("Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs", "cells.csv.gz", "xenium_ovary_tiny_full"),
    ]:
        root = primary / folder
        X, ids, genes = h5_truth(root / "cell_feature_matrix.h5")
        # Match the decimal text's correctly rounded float, not pandas' default
        # fast parser approximation. Keep exact equality in the comparison.
        pos = pd.read_csv(root / meta_name, dtype={"cell_id": str}, float_precision="round_trip").set_index("cell_id")
        add(
            name,
            "xenium",
            root,
            "complete_local_core",
            X,
            ids,
            genes,
            pos.loc[ids, ["x_centroid", "y_centroid"]],
            [root / "cell_feature_matrix.h5", root / meta_name],
        )

    # MERFISH small local core is tested in full; larger cell table is an explicit subset.
    for source, suffix, limit, name in [
        (secondary / "Vizgen_MERSCOPE", "", None, "merfish_small_full"),
        (primary / "Merfish", "", 1000, "merfish_first1000_cells"),
        (primary / "Seqfish", "_section3", 1000, "seqfish_first1000_cells"),
    ]:
        seq = name.startswith("seqfish")
        cfile = source / ("SG_MouseKidneyDataRelease_CxG_section3.csv" if seq else "cell_by_gene.csv")
        mfile = source / ("SG_MouseKidneyDataRelease_CellCoordinates_section3.csv" if seq else "cell_metadata.csv")
        frame = pd.read_csv(cfile, nrows=limit, dtype=str, keep_default_na=False)
        meta = pd.read_csv(mfile, dtype=str, keep_default_na=False)
        ids = frame.iloc[:, 0].tolist()
        key = "label" if seq else meta.columns[0]
        meta = meta.set_index(key, drop=False)
        selected = meta.loc[ids]
        root = source if limit is None else output / "derived_inputs" / name
        if limit is not None:
            root.mkdir(parents=True, exist_ok=True)
            frame.to_csv(root / cfile.name, index=False)
            selected.to_csv(root / mfile.name, index=False)
        add(
            name,
            "seqfish" if seq else "merfish",
            root,
            "complete_local_core" if limit is None else "derived_first1000_cells",
            frame.iloc[:, 1:].to_numpy(dtype=float),
            ids,
            frame.columns[1:],
            selected[["center_x", "center_y"]],
            [cfile, mfile],
        )

    source = secondary / "starmap_plus/starmap+"
    cfile, mfile = source / "well2_5raw_expression_pd.csv", source / "well2_5_spatial.csv"
    frame = pd.read_csv(cfile, usecols=range(1001), dtype=str, keep_default_na=False)
    meta = pd.read_csv(mfile, dtype=str, keep_default_na=False)
    ids = list(frame.columns[1:])
    selected = meta[meta.NAME.isin(["TYPE", *ids])]
    root = output / "derived_inputs/starmap_first1000_cells"
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / cfile.name, index=False)
    selected.to_csv(root / mfile.name, index=False)
    xy = meta.set_index("NAME").loc[ids, ["X", "Y", "Z"]]
    add(
        "starmap_first1000_cells",
        "starmap_plus",
        root,
        "derived_first1000_cells",
        frame.iloc[:, 1:].to_numpy(dtype=float).T,
        ids,
        frame.iloc[:, 0],
        xy,
        [cfile, mfile],
    )

    source = primary / "10x_visium_hd/binned_outputs/square_008um"
    root = output / "derived_inputs/visium_hd_first1000_bins/binned_outputs/square_008um"
    prefix_h5(source / "filtered_feature_bc_matrix.h5", root / "filtered_feature_bc_matrix.h5", 1000)
    X, ids, genes = h5_truth(root / "filtered_feature_bc_matrix.h5")
    pos = pq.read_table(source / "spatial/tissue_positions.parquet", filters=[("barcode", "in", ids)]).to_pandas()
    (root / "spatial").mkdir(exist_ok=True)
    pos.to_parquet(root / "spatial/tissue_positions.parquet", index=False)
    add(
        "visium_hd_first1000_bins",
        "visium_hd_bin",
        root,
        "derived_first1000_filtered_bins",
        X,
        ids,
        genes,
        pos.set_index("barcode").loc[ids, ["pxl_col_in_fullres", "pxl_row_in_fullres"]],
        [source / "filtered_feature_bc_matrix.h5", source / "spatial/tissue_positions.parquet"],
    )

    for source, n, name in [
        (primary / "Stereoseq/test2/SS200000135TL_D1_all_bin1.txt.gz", 10000, "bgi_first10000_molecule_rows"),
        (
            Path(__file__).resolve().parents[1] / "tests/fixtures/bgi/SS200000135TL_D1_bin1_small.gem.gz",
            None,
            "bgi_repository_fixture_full",
        ),
    ]:
        frame = pd.read_csv(source, sep="\t", comment="#", nrows=n, dtype=str, keep_default_na=False)
        count = next(c for c in ["MIDCount", "MIDCounts", "UMICount", "UMICounts", "count", "total"] if c in frame)
        root = output / "derived_inputs" / name
        root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / "sample.gem", sep="\t", index=False)
        genes = list(pd.unique(frame.geneID))
        xy = sorted(set(zip(frame.x.astype(int), frame.y.astype(int))))
        rows = {v: i for i, v in enumerate(xy)}
        cols = {v: i for i, v in enumerate(genes)}
        X = sparse.coo_matrix(
            (
                frame[count].to_numpy(dtype=float),
                ([rows[(int(x), int(y))] for x, y in zip(frame.x, frame.y)], [cols[g] for g in frame.geneID]),
            ),
            shape=(len(xy), len(genes)),
        ).tocsr()
        add(
            name,
            "bgi",
            root,
            "derived_first10000_molecule_rows" if n else "repository_real_data_fixture",
            X,
            [f"{x}_{y}" for x, y in xy],
            genes,
            xy,
            [source],
        )

    for case in cases:
        for repeat in range(1, args.repeats + 1):
            start = time.perf_counter()
            r = read_spatial(case["root"], load_images=False)
            entries = list(r.datasets.values())
            checks = dict(
                status_ok=r.status == "ok",
                unique=len(entries) == 1,
                correct_technology=len(entries) == 1 and entries[0].technology == case["technology"],
            )
            if all(checks.values()):
                a = r.adata
                checks.update(
                    matrix_shape=a.shape == case["X"].shape,
                    all_values=a.shape == case["X"].shape and (a.X != case["X"]).nnz == 0,
                    all_observation_ids=a.obs_names.tolist() == case["ids"],
                    all_feature_names=a.var_names.tolist() == case["genes"],
                    all_coordinates=np.array_equal(a.obsm["spatial"], case["xy"]),
                )
            row = dict(
                name=case["name"],
                technology=case["technology"],
                input_kind=case["input_kind"],
                repeat=repeat,
                shape=list(case["X"].shape),
                nnz=case["X"].nnz,
                status=r.status,
                checks=checks,
                passed=all(checks.values()),
                seconds=time.perf_counter() - start,
                diagnostics=[d["message"] for e in entries for d in e.diagnostics if d["severity"] == "error"],
            )
            results.append(row)
            print(json.dumps(row), flush=True)

    # Report original-layout limitations independently; do not replace them by subset success.
    inspections = []
    for name, root in [
        ("original_hd_multiple_encodings", primary / "10x_visium_hd/binned_outputs/square_008um"),
        ("original_slideseq_missing_counts", primary / "Slideseq"),
        ("original_seqfish_cortex_unsupported_layout", secondary / "seqfish/csv_txt_tsv/SScortex"),
    ]:
        r = read_spatial(root, load=False)
        inspections.append(
            dict(
                name=name,
                status=r.status,
                entries=[
                    dict(technology=e.technology, status=e.status, evidence=e.evidence, diagnostics=e.diagnostics)
                    for e in r.datasets.values()
                ],
                diagnostics=r.diagnostics,
            )
        )
    report = dict(
        repeats=args.repeats,
        unique_core_cases=len(cases),
        calls=len(results),
        technology_count=len({c["technology"] for c in cases}),
        unavailable_real_technologies=["atera", "nanostring", "visium_hd_cellseg", "slideseq_complete_core"],
        full_check_pass=sum(r["passed"] for r in results),
        results=results,
        sources=source_records,
        original_layout_inspections=inspections,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        limitations="User-provided local exports; not all original downloads have independently verified provenance. Derived subsets are not complete original datasets; repeated calls are not independent samples.",
    )
    # Keep local paths out of the portable report; source identity is content hashes.
    text = json.dumps(report, indent=2)
    for root, label in [(primary, "<primary-root>"), (secondary, "<secondary-root>"), (output, "<output>")]:
        text = text.replace(str(root), label)
    (output / "local_real_data_benchmark.json").write_text(text)
    if not all(r["passed"] for r in results):
        raise SystemExit("Real-data failures recorded; inspect JSON")


if __name__ == "__main__":
    main()
