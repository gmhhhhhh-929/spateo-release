"""Native Slide-seq streaming matrix parser used by both reader routes."""

import csv
from array import array

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._errors import ContractError, ResourceDeferred
from ._matrix import _ids, _numeric, _open, integer_counts, table_values


def _slideseq_counts(path, *, full=False, budget=512 * 1024**2):
    """Stream gene-by-bead CSV into sparse storage without a dense table.

    Probe estimates only the bounded row workspace; the full scan enforces a
    growing sparse-storage budget. All values, row widths and IDs are checked.
    """
    with _open(path) as handle:
        lines = (line for line in handle if line.strip() and not line.startswith("#"))
        first = next(lines, None)
        if first is None:
            raise ContractError(f"Empty table: {path}")
        sep = "\t" if first.count("\t") > first.count(",") else ","
        header = next(csv.reader([first], delimiter=sep))
        if len(header) < 2 or len(set(header)) != len(header):
            raise ContractError("Slide-seq requires unique barcode columns and a gene column")
        ids = _ids(header[1:], "Slide-seq expression barcodes")
        workspace = len(header) * 1024
        if workspace > budget:
            raise ResourceDeferred("Slide-seq row workspace exceeds memory budget")
        reader = csv.reader(lines, delimiter=sep)
        values, indices, indptr = array("q"), array("q"), array("q", [0])
        genes = []
        total = 0
        for row in reader:
            if len(row) != len(header):
                raise ContractError(f"Slide-seq row {len(genes) + 2} has inconsistent field count")
            gene = row[0]
            if not gene.strip() or gene.lower() in ("nan", "none", "<na>"):
                raise ContractError("Missing Slide-seq gene identifier")
            numeric = _numeric(row[1:], "Slide-seq counts", nonnegative=True)
            nz = np.flatnonzero(numeric)
            if np.any(numeric[nz] >= 2**53):
                # Rare large integer strings need exact decimal-to-int parsing;
                # keep the common sparse zero-heavy row on the vectorized path.
                counts = table_values(pd.DataFrame({"counts": [row[i + 1] for i in nz]}), "Slide-seq counts").ravel()
            else:
                counts = integer_counts(numeric[nz], "Slide-seq")
            if not full:
                return dict(estimated_bytes=workspace, storage="streamed_slideseq_csv", n_obs=len(ids))
            total += sum(map(int, counts))
            if total > np.iinfo(np.int64).max:
                raise ContractError("Aggregated Slide-seq counts exceed int64 range")
            # Reserve space for buffers, final CSR conversion, ID tables and one row.
            estimated = workspace + (len(genes) + 1) * 1024 + (len(values) + len(nz)) * 64
            if estimated > budget:
                raise ResourceDeferred("Slide-seq sparse storage exceeds memory budget")
            values.frombytes(counts.tobytes())
            indices.frombytes(nz.astype(np.int64, copy=False).tobytes())
            indptr.append(len(values))
            genes.append(gene)
        if not genes:
            raise ContractError(f"Empty expression table: {path}")
        gene_ids = _ids(genes, "Slide-seq expression genes")
        matrix = sparse.csr_matrix(
            (
                np.frombuffer(values, dtype=np.int64),
                np.frombuffer(indices, dtype=np.int64),
                np.frombuffer(indptr, dtype=np.int64),
            ),
            shape=(len(genes), len(ids)),
        ).T.tocsr()
        return AnnData(matrix, obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=gene_ids))
