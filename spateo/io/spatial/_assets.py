"""Optional raster loading shared by explicit and automatic spatial readers."""

import json

import numpy as np
from PIL import Image


def _diagnostic(code, message, severity="error", **extra):
    return dict(code=code, message=str(message), severity=severity, **extra)


def load_assets(adata, candidate, enabled, budget, diagnostics, *, image_budget=32 * 1024**2):
    """Optional raster failures never alter the core result or platform identity."""
    slot = next(iter(adata.uns["spatial"].values()))
    root = candidate.root
    scales = root / "spatial/scalefactors_json.json"
    if scales.is_file() and not scales.is_symlink():
        try:
            if scales.stat().st_size > 1024**2:
                raise ValueError("Scale metadata exceeds size limit")
            value = json.loads(scales.read_text())
            if not isinstance(value, dict):
                raise ValueError("Scale metadata is not an object")
            for key, v in value.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or v <= 0:
                    raise ValueError(f"Invalid scale factor {key}")
            slot["scalefactors"] = value
        except (OSError, ValueError) as exc:
            diagnostics.append(_diagnostic("optional_scale_error", exc, "warning", path=str(scales)))
    files = []
    # Bounded optional inventory; a directory is never recursively expanded here.
    folders = [root, root / "spatial", root / "images", root / "morphology_focus"]
    if candidate.technology == "nanostring":
        folders.extend([root / "CellComposite", root / "CellLabels"])
    for folder in folders:
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            # Directory enumeration errors belong to optional assets, not core counts.
            from itertools import islice

            children = list(islice(folder.iterdir(), 1001))
        except OSError as exc:
            diagnostics.append(_diagnostic("optional_image_error", exc, "warning", path=str(folder)))
            continue
        for i, p in enumerate(children):
            if i >= 1000:
                diagnostics.append(
                    _diagnostic("asset_inventory_limit", f"Optional inventory truncated at {folder}", "warning")
                )
                break
            if (
                not p.is_symlink()
                and p.is_file()
                and p.name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                and (
                    not candidate.options.get("image_prefix")
                    or p.name.startswith(str(candidate.options["image_prefix"]))
                )
            ):
                files.append(p)
    slot["image_files"] = {str(i): str(p.relative_to(root)) for i, p in enumerate(sorted(set(files)))}
    slot["asset_status"] = {}
    used = 0
    image_priority = {"tissue_hires_image.png": 0, "tissue_lowres_image.png": 1}
    for p in sorted(set(files), key=lambda p: (image_priority.get(p.name, 2), str(p))):
        relative = p.relative_to(root).as_posix()
        status = "not_requested"
        if enabled:
            try:
                with Image.open(p) as im:
                    estimated = (
                        im.width
                        * im.height
                        * max(len(im.getbands()), 1)
                        * (4 if im.mode in ("I", "F") else 2 if "16" in im.mode else 1)
                    )
                    if getattr(im, "n_frames", 1) > 1:
                        status = "deferred_multiframe"
                        diagnostics.append(
                            _diagnostic(
                                "optional_image_multiframe",
                                "Multiframe raster retained by path",
                                "warning",
                                path=relative,
                            )
                        )
                    elif estimated > min(image_budget, budget) - used:
                        status = "deferred_resource"
                        diagnostics.append(
                            _diagnostic(
                                "optional_image_resource_limit",
                                "Raster exceeds remaining optional image budget",
                                "warning",
                                path=relative,
                                estimated_bytes=estimated,
                            )
                        )
                    else:
                        key = {"tissue_hires_image.png": "hires", "tissue_lowres_image.png": "lowres"}.get(
                            p.name, relative.replace("/", "__")
                        )
                        arr = np.asarray(im).copy()
                        slot["images"][key] = arr
                        used += arr.nbytes
                        status = "loaded"
            except Image.DecompressionBombError as exc:
                status = "deferred_resource"
                diagnostics.append(_diagnostic("optional_image_resource_limit", exc, "warning", path=relative))
            except (OSError, ValueError) as exc:
                status = "unreadable"
                diagnostics.append(_diagnostic("optional_image_error", exc, "warning", path=relative))
        slot["asset_status"][relative.replace("/", "__")] = status
    if not files:
        diagnostics.append(_diagnostic("optional_images_missing", "No optional raster assets found", "warning"))
    # Images alone do not establish coordinate registration.
    slot["metadata"]["image_registration"] = "scale_metadata_present" if slot["scalefactors"] else "not_established"
    return used
