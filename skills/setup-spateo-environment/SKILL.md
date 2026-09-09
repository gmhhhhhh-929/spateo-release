---
name: setup-spateo-environment
description: Create, update, diagnose, and verify a reproducible Spateo environment for CPython 3.10-3.12. Use when installing this repository, resolving NumPy/AnnData/geospatial binary conflicts, checking an existing conda environment, or preparing a user to run Spateo IO, optional serial-slice quality control, spatial preprocessing, and native morphogenesis tools.
---

# Setup Spateo Environment

Create a separate environment by default. Preserve the user's current environment unless they explicitly request an in-place update.

## Workflow

1. Locate the Spateo repository and confirm that `environment.yml`, `requirements.txt`, `setup.py`, and `spateo/preprocessing/slice_quality.py` belong to the same checkout.
2. Inspect the requested or active interpreter with `python --version` and `python -m pip check`. Support CPython 3.10, 3.11, and 3.12 only.
3. Prefer conda-forge for NumPy, SciPy, HDF5, Arrow, GeoPandas, Shapely, and image libraries. These compiled packages are the most common source of macOS and Linux ABI conflicts.
4. For a new environment, run from the repository root:

   ```bash
   conda env create --file environment.yml
   conda activate spateo
   ```

   If the name already exists, choose a new name or request permission before updating it. Do not use `--prune` unless the user explicitly approves removal of packages.
5. For a compatible existing environment, install the repository with `python -m pip install -e .`. Use `python -m pip install -r dev-requirements.txt` only when tests or development tools are needed.
   Core marching-cubes surface construction is installed with Spateo. When repairing an older environment, install the complete built-in path together: PyMCubes extracts the surface, PyMeshFix repairs it, and PyACVD remeshes it.

   ```bash
   python -m pip install \
     "PyMCubes>=0.1.6,<0.2" \
     "pymeshfix>=0.18.1,<0.19" \
     "pyacvd>=0.4,<0.5"
   ```

   The distribution is named `PyMCubes`, but Spateo imports it as `mcubes`. For the full optional 3D toolkit, including Open3D-based methods, install and verify the 3D extra:

   ```bash
   python -m pip install -e ".[3d]"
   python skills/setup-spateo-environment/scripts/verify_environment.py --smoke-test-3d
   ```
6. Keep Python environments and Codex skills separate. Install Spateo into the
   conda environment; install Codex skills under
   `${CODEX_HOME:-$HOME/.codex}/skills`. Never `pip install` a Codex skill.
   The repository's `setup-spateo-environment` skill manages the environment.
   The companion `spatial-slice-quality-qc` skill owns the HTML viewer and
   orchestration around the package-level detector. When the user asks Codex
   to install these skills, use the Codex skill installer and preserve an
   existing installation unless replacement is explicitly authorized.

7. Run the verifier:

   ```bash
   python skills/setup-spateo-environment/scripts/verify_environment.py --smoke-test
   python -m pip check
   ```

   The default smoke test now covers ordinary IO/preprocessing plus the
   serial-slice QC API, including an in-memory dataset, controlled expression
   thinning, and two independent file-backed datasets. Use the narrower check
   when only the newly added QC surface needs verification:

   ```bash
   python skills/setup-spateo-environment/scripts/verify_environment.py \
     --smoke-test-slice-qc
   ```

   Add `--smoke-test-3d` when the environment must support marching-cubes mesh reconstruction.

8. Run focused tests before the full suite:

   ```bash
   python -m pytest -q tests/io tests/preprocessing
   python -m pytest -q
   ```

9. Report the Python executable, resolved core versions, verifier result,
   slice-QC smoke-test result, test counts, installed Codex skill locations,
   and any optional feature that remains unavailable.

## Compatibility contract

- Keep `numpy>=1.23.5,<2` because legacy Spateo numerical modules are not yet audited for NumPy 2.
- Keep `anndata>=0.9,<0.12`; the maintained Spateo APIs are tested across the 0.9-0.11 storage conventions.
- Do not install Dynamo for Spateo. Neighbor graphs, sampling, normalization, and sparse vector fields are implemented inside this repository.
- Use Shapely 2 with a modern GeoPandas build; do not mix a PyGEOS-backed GeoPandas installation with Shapely 1.
- Install `pyarrow`, `tifffile`, and `Pillow` for Parquet and microscopy-image spatial readers.
- Install `PyMCubes>=0.1.6,<0.2` for `spateo.tdr` marching-cubes mesh reconstruction; verify the import name `mcubes`, not `pymcubes`.
- Install `pymeshfix>=0.18.1,<0.19` for `spateo.tdr` mesh repair. Spateo calls `MeshFix.repair()` without the removed `verbose` keyword so the current API remains compatible.
- Install `pyacvd>=0.4,<0.5` for the uniform remeshing stage that `construct_surface` runs after surface repair.
- Never silence `pip check` failures. Resolve them or state the exact remaining conflict.
- Serial-slice QC adds no dependency beyond the existing `anndata`, `numpy`,
  `pandas`, `scipy`, and `h5py` compatibility window.
- Verify the package-facing QC API through `spateo.pp`; do not make the Spateo
  distribution depend on the skill-side HTML viewer.

Read [references/troubleshooting.md](references/troubleshooting.md) only when verification or installation fails.
