#!/usr/bin/env python3
"""Verify Spateo's supported interpreter, dependencies, and core workflow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from importlib import import_module
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Verify the checkout that owns this skill even when another editable Spateo
# installation appears earlier in site-packages.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

# Scientific Python imports may initialize caches even during verification.
# Fall back to a writable, process-independent temp location on CI workers and
# machines with a read-only home directory, while respecting user overrides.
for variable, directory in {
    "MPLCONFIGDIR": "spateo-matplotlib-cache",
    "NUMBA_CACHE_DIR": "spateo-numba-cache",
    "XDG_CACHE_HOME": "spateo-xdg-cache",
}.items():
    cache_path = Path(tempfile.gettempdir()) / directory
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(cache_path))


REQUIRED = {
    "numpy": ">=1.23.5,<2",
    "pandas": ">=1.5.3,<2.3",
    "scipy": ">=1.10,<1.14",
    "anndata": ">=0.9,<0.12",
    "h5py": ">=3.8,<4",
    "geopandas": ">=0.14,<1.2",
    "shapely": ">=2,<3",
    "pyarrow": ">=12,<26",
}

IMPORT_NAMES = {"shapely": "shapely"}


def _run_smoke_test() -> dict[str, object]:
    import numpy as np
    from anndata import AnnData
    from scipy import sparse

    import spateo as st
    from spateo._native import sparse_vector_field

    counts = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 1], [2, 1, 1], [1, 2, 0]]))
    adata = AnnData(counts)
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    result = st.pp.preprocess_spatial(
        adata,
        recipe="generic",
        min_cells=1,
        n_top_genes=2,
        n_pca_components=2,
        inplace=False,
    )
    if result is None or "counts" not in result.layers or "X_pca" not in result.obsm:
        raise RuntimeError("Spateo preprocessing smoke test did not produce the expected layers and PCA.")
    velocity = np.tile(np.asarray([0.5, -0.25]), (adata.n_obs, 1))
    vector_field = sparse_vector_field(adata.obsm["spatial"], velocity, Grid=adata.obsm["spatial"], M=4)
    if not np.isfinite(vector_field["grid_V"]).all():
        raise RuntimeError("Spateo native vector-field smoke test returned non-finite values.")
    return {
        "shape": list(result.shape),
        "pca_shape": list(result.obsm["X_pca"].shape),
        "vector_field_shape": list(vector_field["grid_V"].shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny IO/preprocessing import workflow.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        errors.append("Spateo supports CPython 3.10 through 3.12 in this release.")

    packages: dict[str, object] = report["packages"]  # type: ignore[assignment]
    for package, specifier in REQUIRED.items():
        try:
            installed = version(package)
            compatible = Version(installed) in SpecifierSet(specifier)
            import_module(IMPORT_NAMES.get(package, package.replace("-", "_")))
        except PackageNotFoundError:
            installed, compatible = "missing", False
        except Exception as exc:
            installed, compatible = f"import failed: {exc}", False
        packages[package] = {"version": installed, "required": specifier, "compatible": compatible}
        if not compatible:
            errors.append(f"{package}: found {installed}, expected {specifier}")

    try:
        declared = [Requirement(item).name.lower() for item in (requires("spateo-release") or [])]
    except PackageNotFoundError:
        declared = []
    report["dynamo_required"] = any(name in {"dynamo", "dynamo-release"} for name in declared)
    if report["dynamo_required"]:
        errors.append("Installed Spateo metadata still declares Dynamo; reinstall this checkout.")

    try:
        import spateo as st

        report["spateo"] = {"version": st.__version__, "path": st.__file__}
        for dotted in ("io.read_atera", "io.read_visium", "pp.preprocess_spatial"):
            current = st
            for part in dotted.split("."):
                current = getattr(current, part)
    except Exception as exc:
        errors.append(f"spateo import/API check failed: {exc}")

    if args.smoke_test and not errors:
        try:
            report["smoke_test"] = _run_smoke_test()
        except Exception as exc:
            errors.append(f"smoke test failed: {exc}")

    report["ok"] = not errors
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Python: {report['python']} ({report['executable']})")
        for package, details in packages.items():
            marker = "OK" if details["compatible"] else "FAIL"  # type: ignore[index]
            print(f"[{marker}] {package}: {details['version']} (required {details['required']})")  # type: ignore[index]
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Environment verification passed." if not errors else "Environment verification failed.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
