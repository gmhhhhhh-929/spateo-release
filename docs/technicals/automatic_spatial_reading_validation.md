# Automatic spatial reading: validation record

Validation date: 2026-09-20. Policy: `spatial-contracts-v1`.
Tests ran against the actual checkout in a working Python 3.10.21 Spateo
environment, including normal package imports.

## Automated checks

```bash
SPATEO_VISIUM_DATA=/path/to/V1_Adult_Mouse_Brain python -m pytest -q tests/io tests/preprocessing --disable-warnings
make check
python -m black --check scripts/verify_automatic_spatial_reading.py
python -m isort --profile black --check-only scripts/verify_automatic_spatial_reading.py
git diff --check
```

- IO and preprocessing: **81 passed**, 2 warnings, no failures or skips.
- `make check`: compilation and configured isort/Black checks passed (87 files).
- The verification script passed separate formatting/import-order checks.
- Warnings are not converted into passes for failed content checks. The real
  10x source has duplicated gene symbols; symbols are preserved, while feature
  IDs and observation IDs retain their identities.

Synthetic fixtures cover each supported core technology, incomplete and corrupt
inputs, invalid expression values, identifier mismatches, multiple samples and
representations, deferred loads, optional image failures, bounded discovery and
source changes. The original validation used a guard that made the legacy score ranker raise if invoked by
the new entry point. Small fixtures do not establish support for every vendor
export version.

## Independent real Visium verification

Public dataset: [10x Mouse Brain Section (Coronal), standard 1.1.0](https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0),
local bundle name `V1_Adult_Mouse_Brain`.

| Measurement | Result |
|---|---:|
| Observations | 2,702 |
| Features | 32,285 |
| Nonzero matrix entries | 16,031,101 |
| Total UMI | 85,825,294 |
| Matrix dtype | int32 |
| Collection status | ok |

The independent script verifies all matrix entries against the original H5,
every barcode and its order, all feature IDs and symbols, and every XY pair
joined by barcode against the original six-column positions CSV. It writes an
H5AD outside the input directory and checks matrix, observation/feature tables,
coordinates, loaded images and scale factors after reloading. All checks passed.
Standard hires and lowres images were loaded before optional QC rasters.

The [machine-readable record](automatic_spatial_reading_validation.json) includes
the exact core-file SHA-256 hashes, dependency versions, image shapes and check
outcomes. The dataset and generated H5AD are not included in the repository.

Reproduce with:

```bash
python scripts/verify_automatic_spatial_reading.py /path/to/V1_Adult_Mouse_Brain --output /path/to/report
```

This creates `real_visium_validation.json`, `spatial_read_report.json`, and
`V1_Adult_Mouse_Brain_score_free.h5ad`. Choose an output directory outside the
input dataset. The report is evidence for this dataset and code version; it is
not a calibrated detector accuracy estimate or validation of all platforms on
real data.

Cross-platform expansion: see the [complete benchmark report](automatic_spatial_reading_benchmark_zh.md) for 11 format categories, synthetic positive/negative cases, seven real-source categories and retained initial failures.


## Removal of the former scoring API

The score ranker and detector APIs have since been deleted. Regression coverage now verifies that the retired exports are absent, all three automatic reader names share one implementation and return type, obsolete keyword arguments fail explicitly, and discovery can be followed by a complete validated read. Historical benchmark figures above describe their original run; they are not a new accuracy estimate.
