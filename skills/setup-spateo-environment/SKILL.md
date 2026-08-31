---
name: setup-spateo-environment
description: Create, update, diagnose, and verify a reproducible Spateo environment for CPython 3.10-3.12. Use when installing this repository, resolving NumPy/AnnData/Dynamo/geospatial binary conflicts, checking an existing conda environment, or preparing a new user to run Spateo IO and spatial preprocessing.
---

# Setup Spateo Environment

Create a separate environment by default. Preserve the user's current environment unless they explicitly request an in-place update.

## Workflow

1. Locate the Spateo repository and confirm that `environment.yml`, `requirements.txt`, and `setup.py` belong to the same checkout.
2. Inspect the requested or active interpreter with `python --version` and `python -m pip check`. Support CPython 3.10, 3.11, and 3.12 only.
3. Prefer conda-forge for NumPy, SciPy, HDF5, Arrow, GeoPandas, Shapely, and image libraries. These compiled packages are the most common source of macOS and Linux ABI conflicts.
4. For a new environment, run from the repository root:

   ```bash
   conda env create --file environment.yml
   conda activate spateo
   ```

   If the name already exists, choose a new name or request permission before updating it. Do not use `--prune` unless the user explicitly approves removal of packages.
5. For a compatible existing environment, install the repository with `python -m pip install -e .`. Use `python -m pip install -r dev-requirements.txt` only when tests or development tools are needed.
6. Run the verifier:

   ```bash
   python skills/setup-spateo-environment/scripts/verify_environment.py --smoke-test
   python -m pip check
   ```

7. Run focused tests before the full suite:

   ```bash
   python -m pytest -q tests/io tests/preprocessing
   python -m pytest -q
   ```

8. Report the Python executable, resolved core versions, verifier result, test counts, and any optional feature that remains unavailable.

## Compatibility contract

- Keep `numpy>=1.23.5,<2` because legacy Spateo numerical modules are not yet audited for NumPy 2.
- Keep `anndata>=0.9,<0.11` while `dynamo-release>=1.5.3,<2` declares `anndata<0.11`.
- Use Shapely 2 with a modern GeoPandas build; do not mix a PyGEOS-backed GeoPandas installation with Shapely 1.
- Install `pyarrow`, `tifffile`, and `Pillow` for Parquet and microscopy-image spatial readers.
- Never silence `pip check` failures. Resolve them or state the exact remaining conflict.

Read [references/troubleshooting.md](references/troubleshooting.md) only when verification or installation fails.
