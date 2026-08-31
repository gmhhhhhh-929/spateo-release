# Spateo environment troubleshooting

## Binary or ABI import errors

If NumPy, SciPy, HDF5, Arrow, Shapely, GeoPandas, OpenCV, or VTK fails to import, avoid repeatedly replacing individual wheels. Create a fresh conda-forge environment from `environment.yml`, then install Spateo editable inside it.

On Apple Silicon, confirm all compiled packages use the same architecture:

```bash
python -c "import platform; print(platform.machine())"
conda list | grep -E 'numpy|scipy|h5py|pyarrow|shapely|geopandas'
```

## AnnData and Dynamo resolver conflict

`dynamo-release` 1.5.x requires `anndata<0.11`. If pip selects AnnData 0.11 or newer, reinstall using the repository requirements rather than forcing Dynamo with `--no-deps`.

## Shapely and PyGEOS warning

A warning about GEOS versions usually means an older PyGEOS-backed GeoPandas stack is mixed with Shapely 1. Create or update to the conda-forge Shapely 2 and GeoPandas 0.14+ stack. Do not set `USE_PYGEOS=0` as a permanent substitute for correcting the environment.

## Numba cache errors

Set a writable cache directory for restricted or cluster environments:

```bash
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"
```

For headless test workers, a writable Matplotlib cache also avoids repeated font discovery:

```bash
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

## `pip check` reports unrelated notebook conflicts

Treat the check as failed until resolved. For example, an old `nbformat` with a modern `nbconvert` should be repaired by installing the repository requirements, which constrain both packages to a compatible range.
