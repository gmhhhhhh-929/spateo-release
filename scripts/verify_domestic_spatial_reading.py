"""Full native GSM8816652 read with independent streamed source comparisons."""

import argparse
import csv
import gzip
import itertools
import json
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

import spateo as st

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Native GSM8816652 bundle with standardized filenames")
parser.add_argument("--output", type=Path, required=True, help="New validation output directory")
args = parser.parse_args()
SOURCE = args.source.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
result = st.io.read_spatial(SOURCE, max_memory_bytes=4 * 1024**3)
result.write_report(OUT / "public_bmk_read_report.json")
assert result.status == "ok", result.report
adata = result.adata
read_seconds = time.perf_counter() - started
assert adata.uns["spateo_io"]["technology"] == "bmkmanu"
assert adata.X.dtype == np.int64
barcodes = pd.read_csv(SOURCE / "barcodes.tsv.gz", header=None, dtype=str)[0].tolist()
features = pd.read_csv(SOURCE / "features.tsv.gz", sep="\t", header=None, dtype=str)
with gzip.open(SOURCE / "barcodes_pos.tsv.gz", "rt") as handle:
    positions = {r[0]: [float(r[1]), float(r[2])] for r in csv.reader(handle, delimiter="\t")}
assert list(adata.obs_names) == barcodes
assert list(adata.var_names) == features[0].tolist()
assert list(adata.var.gene_name) == features[1].tolist()
np.testing.assert_array_equal(adata.obsm["spatial"], np.array([positions[b] for b in barcodes]))

obs_totals = np.zeros(adata.n_obs, dtype=np.int64)
var_totals = np.zeros(adata.n_vars, dtype=np.int64)
records = total = 0
with gzip.open(SOURCE / "matrix.mtx.gz", "rt") as handle:
    header = next(handle).strip()
    line = next(handle)
    while line.startswith("%"):
        line = next(handle)
    genes, spots, declared_records = map(int, line.split())
    assert adata.shape == (spots, genes)
    while True:
        lines = list(itertools.islice(handle, 200000))
        if not lines:
            break
        values = np.loadtxt(lines)
        assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
        values = values.astype(np.int64)
        rows, cols, counts = values.T
        rows -= 1
        cols -= 1
        # Compare every source entry to the returned sparse matrix, independently
        # of scipy.io.mmread and the adapter's gene/barcode parsing.
        actual = np.asarray(adata.X[cols, rows]).ravel()
        np.testing.assert_array_equal(actual, counts)
        np.add.at(obs_totals, cols, counts)
        np.add.at(var_totals, rows, counts)
        records += len(counts)
        total += sum(map(int, counts))
assert records == declared_records == adata.X.nnz
np.testing.assert_array_equal(np.asarray(adata.X.sum(axis=1)).ravel(), obs_totals)
np.testing.assert_array_equal(np.asarray(adata.X.sum(axis=0)).ravel(), var_totals)
assert int(adata.X.sum()) == total

output = OUT / "GSM8816652_BMK_native.h5ad"
adata.write_h5ad(output)
restored = anndata.read_h5ad(output)
assert (restored.X != adata.X).nnz == 0
assert list(restored.obs_names) == barcodes
assert list(restored.var_names) == list(adata.var_names)
np.testing.assert_array_equal(restored.obsm["spatial"], adata.obsm["spatial"])
images = {lib: list(slot["images"]) for lib, slot in adata.uns["spatial"].items()}
assert any(images.values()), "Expected public H&E PNG to load"
checks = {
    "accession": "GSM8816652",
    "technology": "bmkmanu",
    "input": "native MEX + barcodes_pos.tsv.gz + PNG",
    "no_h5ad_input": True,
    "shape": list(adata.shape),
    "matrix_dtype": str(adata.X.dtype),
    "source_matrix_header": header,
    "source_entries_compared": records,
    "total_counts": total,
    "all_counts_exact": True,
    "all_coordinates_exact": True,
    "all_axis_ids_exact": True,
    "per_gene_and_spot_totals_exact": True,
    "h5ad_roundtrip_exact": True,
    "images_loaded": images,
    "read_seconds": read_seconds,
    "total_validation_seconds": time.perf_counter() - started,
    "output": str(output),
    "caveat": "One public native dataset; not a clinical accuracy or cross-platform benchmark.",
}
(OUT / "public_bmk_validation.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2), flush=True)
