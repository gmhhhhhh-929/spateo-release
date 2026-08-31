# Installation

## Supported Python versions

This branch supports CPython 3.10, 3.11, and 3.12. Python 3.10 is the reference environment because it also has the broadest binary-wheel support for Spateo's optional 3D and segmentation dependencies.

Do not independently upgrade NumPy to 2 or AnnData to 0.11+: the current Dynamo 1.5 dependency and legacy Spateo numerical modules require the compatibility window declared in `requirements.txt`.

## Recommended conda installation

Clone the repository and create a fresh environment from the tested definition:

```bash
git clone https://github.com/gmhhhhhh-929/spateo-release.git
cd spateo-release
conda env create --file environment.yml
conda activate spateo
```

The environment uses conda-forge for compiled scientific, HDF5, Arrow, and geospatial packages, then installs this checkout in editable mode.

Verify the result before analysis:

```bash
python skills/setup-spateo-environment/scripts/verify_environment.py --smoke-test
python -m pip check
```

## Existing environment

For an existing CPython 3.10-3.12 environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r dev-requirements.txt  # only for development/testing
python -m pytest -q tests/io tests/preprocessing
```

Prefer creating a new environment over repeatedly replacing NumPy, HDF5, Shapely, or VTK wheels in an old environment. Mixed conda/pip binary stacks are a common cause of import and GEOS errors.

## MPI

The cell-cell interaction modeling framework additionally requires MPI:

```bash
conda install -c conda-forge mpi4py
mpiexec --version
```

## Codex environment skill

The repository includes `$setup-spateo-environment` under `skills/setup-spateo-environment`. It guides a coding agent through non-destructive environment creation, compatibility checks, focused tests, and troubleshooting.

## Development

Install development tools and run the suite:

```bash
python -m pip install -r dev-requirements.txt
python -m pytest -q
```

See [](contributing) for contribution guidelines.
