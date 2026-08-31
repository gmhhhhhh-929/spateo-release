"""Compatibility API for the maintained Slide-seq reader."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .spatial._slideseq import _read_slideseq_beads
from .spatial._slideseq import read_slideseq as _read_slideseq


def read_slideseq_as_dataframe(path: str) -> pd.DataFrame:
    """Read a historical Slide-seq DGE table in long format."""
    frame = pd.read_csv(path, sep=None, engine="python").rename(columns={"GENE": "gene"})
    if "gene" not in frame:
        frame = frame.rename(columns={frame.columns[0]: "gene"})
    frame = frame.melt(id_vars="gene", var_name="barcode", value_name="count")
    frame = frame[pd.to_numeric(frame["count"], errors="coerce").fillna(0) > 0].copy()
    frame["gene"] = frame["gene"].astype("category")
    frame["barcode"] = frame["barcode"].astype("category")
    frame["count"] = frame["count"].astype(np.uint32)
    return frame


def read_slideseq_beads_as_dataframe(path: str) -> pd.DataFrame:
    """Read and standardize a historical Slide-seq bead coordinate table."""
    return _read_slideseq_beads(Path(path))


def read_slideseq(
    path: str,
    beads_path: Optional[str] = None,
    binsize: Optional[int] = None,
    version: str = "slide2",
    **kwargs,
) -> AnnData:
    """Read directory outputs while accepting the historical two-path call."""
    if beads_path is None:
        return _read_slideseq(path, **kwargs)
    counts_path = Path(path).expanduser().resolve()
    bead_path = Path(beads_path).expanduser().resolve()
    if counts_path.parent != bead_path.parent:
        raise ValueError("Legacy Slide-seq count and bead files must share a parent directory.")
    if binsize not in (None, 1):
        raise ValueError(
            "The compatibility wrapper no longer aggregates Slide-seq while reading; "
            "read bead-level data and preprocess/aggregate explicitly."
        )
    return _read_slideseq(
        counts_path.parent,
        counts_file=counts_path.name,
        bead_file=bead_path.name,
        **kwargs,
    )


__all__ = ["read_slideseq", "read_slideseq_as_dataframe", "read_slideseq_beads_as_dataframe"]
