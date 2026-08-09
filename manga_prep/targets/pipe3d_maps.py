from pathlib import Path
import json

import numpy as np
from astropy.io import fits


DEFAULT_TARGET_SIZE = 76

# Network prediction targets (0-1 scaled arrays in amara_maps.npz).
AMARA_TARGET_KEYS = (
    "ha_flux",
    "hbeta_flux",
    "oiii_5007_flux",
    "nii_6584_flux",
    "ha_ew",
    "stellar_av",
)

PIPE3D_MAP_SPECS = [
    {
        "key": "ha_flux",
        "label": "Halpha flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Halpha", "Ha"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
    },
    {
        "key": "hbeta_flux",
        "label": "Hbeta flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Hbeta", "Hb"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
    },
    {
        "key": "oiii_5007_flux",
        "label": "[OIII]5007 flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["[OIII]5007", "[OIII] 5007", "OIII5007"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
    },
    {
        "key": "nii_6584_flux",
        "label": "[NII]6584 flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["[NII]6584", "[NII] 6584", "NII6584"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
    },
    {
        "key": "ha_ew",
        "label": "Halpha EW",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Halpha", "Ha"],
        "plane_offset": 171,
        "unit": "Angstrom",
        "transform": "log10_negative_emission",
        "clip_min": 0.0,
        "clip_max": 3.0,
    },
    {
        "key": "stellar_av",
        "label": "stellar Av",
        "extension": "SSP",
        "plane": 11,
        "unit": "mag",
        "transform": "linear",
        "clip_min": 0.0,
        "clip_max": 3.0,
    },
]


def infer_plateifu_from_path(path):
    name = Path(path).name
    if name.startswith("manga-") and ".Pipe3D" in name:
        return name.replace("manga-", "").split(".Pipe3D")[0]
    return Path(path).stem.replace("_", "-")


def discover_pipe3d_cubes(data_root):
    return sorted(Path(data_root).glob("**/manga-*.Pipe3D.cube.fits*"))


def native_float_array(values):
    return np.array(values, dtype=np.float64, copy=True)


def flux_elines_lookup(hdul):
    hdr = hdul["FLUX_ELINES"].header
    rows = []
    for i in range(57):
        raw_name = str(hdr.get(f"NAME{i}", ""))
        rows.append(
            {
                "line_index": i,
                "raw_name": raw_name,
                "clean_name": raw_name.replace("flux ", "").strip(),
                "wave": hdr.get(f"WAVE{i}", np.nan),
                "unit": hdr.get(f"UNIT{i}", ""),
            }
        )
    return rows


def find_line_index(line_lookup, patterns):
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for row in line_lookup:
            if pattern_lower in str(row["clean_name"]).lower():
                return int(row["line_index"])
    return None


def native_shape_from_pipe3d(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SSP"].data
        return int(data.shape[1]), int(data.shape[2])


def max_native_shape(paths):
    shapes = [native_shape_from_pipe3d(path) for path in paths]
    if not shapes:
        raise ValueError("No Pipe3D cubes were found.")
    max_y = max(shape[0] for shape in shapes)
    max_x = max(shape[1] for shape in shapes)
    return max_y, max_x


def _line_plane(spec, line_lookup):
    line_index = find_line_index(line_lookup, spec["line_patterns"])
    if line_index is None:
        raise KeyError(f"Could not find {spec['key']} in FLUX_ELINES.")
    return line_index + int(spec.get("plane_offset", 0))


def extract_direct_pipe3d_maps(path):
    maps = {}
    with fits.open(path, memmap=True) as hdul:
        line_lookup = flux_elines_lookup(hdul)
        extension_cache = {}
        for spec in PIPE3D_MAP_SPECS:
            extension = spec["extension"]
            if extension not in extension_cache:
                extension_cache[extension] = native_float_array(hdul[extension].data)
            cube = extension_cache[extension]
            if "plane" in spec:
                plane = int(spec["plane"])
            else:
                plane = _line_plane(spec, line_lookup)
            maps[spec["key"]] = np.asarray(cube[plane], dtype=np.float64)
    return maps


def transform_map(values, transform):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if transform == "linear":
        good = np.isfinite(values)
        out[good] = values[good]
    elif transform == "log10_positive":
        good = np.isfinite(values) & (values > 0)
        out[good] = np.log10(values[good])
    elif transform == "log10_negative_emission":
        good = np.isfinite(values) & (values < 0)
        out[good] = np.log10(-values[good])
    else:
        raise ValueError(f"Unknown transform: {transform!r}")
    return out


def scale_map(values, spec, clip=True):
    transformed = transform_map(values, spec["transform"])
    clip_min = float(spec["clip_min"])
    clip_max = float(spec["clip_max"])
    if clip_max <= clip_min:
        raise ValueError(f"Invalid clipping range for {spec['key']}: {clip_min}, {clip_max}")
    clipped = np.clip(transformed, clip_min, clip_max) if clip else transformed
    return (clipped - clip_min) / (clip_max - clip_min)


def _align_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop or center-pad a 2D mask to target_shape."""
    mask = np.asarray(mask)
    ty, tx = map(int, target_shape)
    sy, sx = mask.shape
    if (sy, sx) == (ty, tx):
        return mask

    out = np.zeros((ty, tx), dtype=mask.dtype)
    copy_y = min(sy, ty)
    copy_x = min(sx, tx)
    src_y0 = max(0, (sy - ty) // 2)
    src_x0 = max(0, (sx - tx) // 2)
    dst_y0 = max(0, (ty - sy) // 2)
    dst_x0 = max(0, (tx - sx) // 2)
    out[dst_y0 : dst_y0 + copy_y, dst_x0 : dst_x0 + copy_x] = mask[
        src_y0 : src_y0 + copy_y, src_x0 : src_x0 + copy_x
    ]
    return out


def extract_select_reg_footprint(pipe3d_path, native_shape: tuple[int, int]) -> np.ndarray:
    """Pipe3D SELECT_REG: 1 where spaxels are in the analysis region."""
    native_y, native_x = map(int, native_shape)
    with fits.open(pipe3d_path, memmap=True) as hdul:
        if "SELECT_REG" not in hdul or hdul["SELECT_REG"].data is None:
            return np.ones((native_y, native_x), dtype=np.uint8)
        select_reg = np.asarray(hdul["SELECT_REG"].data)
        if select_reg.shape != (native_y, native_x):
            select_reg = _align_mask_to_shape(select_reg, (native_y, native_x))
        return (select_reg > 0).astype(np.uint8)


def center_pad(image, target_shape, pad_value=np.nan):
    image = np.asarray(image)
    target_y, target_x = map(int, target_shape)
    native_y, native_x = image.shape
    if native_y > target_y or native_x > target_x:
        raise ValueError(
            f"Native map shape {(native_y, native_x)} is larger than target_shape={target_shape}."
        )
    y0 = (target_y - native_y) // 2
    x0 = (target_x - native_x) // 2
    out = np.full((target_y, target_x), pad_value, dtype=image.dtype)
    out[y0 : y0 + native_y, x0 : x0 + native_x] = image
    return out


def map_stats(raw_map, scaled_map):
    raw = np.asarray(raw_map, dtype=np.float64)
    scaled = np.asarray(scaled_map, dtype=np.float64)
    finite_raw = raw[np.isfinite(raw)]
    finite_scaled = scaled[np.isfinite(scaled)]
    return {
        "raw_finite_count": int(finite_raw.size),
        "raw_min": float(np.min(finite_raw)) if finite_raw.size else None,
        "raw_max": float(np.max(finite_raw)) if finite_raw.size else None,
        "scaled_finite_count": int(finite_scaled.size),
        "scaled_min": float(np.min(finite_scaled)) if finite_scaled.size else None,
        "scaled_max": float(np.max(finite_scaled)) if finite_scaled.size else None,
        "scaled_at_0_count": int(np.sum(finite_scaled == 0.0)) if finite_scaled.size else 0,
        "scaled_at_1_count": int(np.sum(finite_scaled == 1.0)) if finite_scaled.size else 0,
    }


def build_collaborator_arrays(pipe3d_path, target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE), clip=True):
    raw_maps = extract_direct_pipe3d_maps(pipe3d_path)
    first_map = next(iter(raw_maps.values()))
    native_y, native_x = first_map.shape
    arrays = {
        "native_shape": np.array([native_y, native_x], dtype=np.int16),
        "native_ny": np.array(native_y, dtype=np.int16),
        "native_nx": np.array(native_x, dtype=np.int16),
        "native_spaxel_count": np.array(native_y * native_x, dtype=np.int32),
        "target_shape": np.array(target_shape, dtype=np.int16),
    }
    metadata = {
        "pipe3d_path": str(pipe3d_path),
        "plateifu": infer_plateifu_from_path(pipe3d_path),
        "native_shape": [int(native_y), int(native_x)],
        "native_spaxel_count": int(native_y * native_x),
        "target_shape": [int(target_shape[0]), int(target_shape[1])],
        "clip_scaled_to_0_1": bool(clip),
        "maps": [],
    }

    footprint = extract_select_reg_footprint(pipe3d_path, (native_y, native_x))
    padded_footprint = center_pad(footprint, target_shape, pad_value=0)
    arrays["native_footprint_mask"] = padded_footprint.astype(np.uint8)

    for spec in PIPE3D_MAP_SPECS:
        key = spec["key"]
        raw_map = raw_maps[key]
        scaled_map = scale_map(raw_map, spec, clip=clip)
        arrays[f"{key}_raw"] = center_pad(raw_map, target_shape, pad_value=np.nan)
        arrays[f"{key}_scaled"] = center_pad(scaled_map, target_shape, pad_value=np.nan)
        feature_valid = np.isfinite(arrays[f"{key}_scaled"]).astype(np.uint8)
        arrays[f"{key}_valid_mask"] = feature_valid
        arrays[f"{key}_loss_mask"] = (
            (padded_footprint.astype(bool) & feature_valid.astype(bool)).astype(np.uint8)
        )
        metadata["maps"].append(
            {
                "key": key,
                "raw_array": f"{key}_raw",
                "scaled_array": f"{key}_scaled",
                "valid_mask": f"{key}_valid_mask",
                "loss_mask": f"{key}_loss_mask",
                "label": spec["label"],
                "unit": spec["unit"],
                "transform": spec["transform"],
                "clip_min": float(spec["clip_min"]),
                "clip_max": float(spec["clip_max"]),
                "scaled_formula": "(clip(transform(raw), clip_min, clip_max) - clip_min) / (clip_max - clip_min)",
                **map_stats(raw_map, scaled_map),
            }
        )
    return arrays, metadata


def write_amara_maps(
    pipe3d_path,
    galaxy_dir=None,
    target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    clip=True,
):
    """Write network-ready maps into a galaxy folder as amara_maps.npz."""
    pipe3d_path = Path(pipe3d_path)
    plateifu = infer_plateifu_from_path(pipe3d_path)
    if galaxy_dir is None:
        galaxy_dir = pipe3d_path.parent
    galaxy_dir = Path(galaxy_dir)
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = build_collaborator_arrays(pipe3d_path, target_shape=target_shape, clip=clip)
    npz_path = galaxy_dir / "amara_maps.npz"
    json_path = galaxy_dir / "amara_maps_metadata.json"
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"plateifu": plateifu, "galaxy_dir": galaxy_dir, "npz": npz_path, "metadata": json_path, **metadata}


def load_amara_maps(galaxy_dir):
    """Load amara_maps.npz from a manga_sdss_fits/<plate_ifu> folder."""
    galaxy_dir = Path(galaxy_dir)
    npz_path = galaxy_dir / "amara_maps.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    return np.load(npz_path)


def _loss_mask_from_npz(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if f"{key}_loss_mask" in arrays:
        return arrays[f"{key}_loss_mask"].astype(np.uint8)
    footprint = arrays["native_footprint_mask"].astype(bool)
    valid = arrays[f"{key}_valid_mask"].astype(bool)
    return (footprint & valid).astype(np.uint8)


def load_amara_training_targets(
    galaxy_dir,
    *,
    scaled: bool = True,
    keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    """
    Load UNet targets and masks from amara_maps.npz.

    Returns dict with:
      targets: {feature: (76, 76) float array}
      target_valid_masks: per-feature valid mask after Amara transforms
      target_loss_masks: footprint & valid — use for masked training loss
      footprint_mask: Pipe3D SELECT_REG padded to target canvas
      native_shape, target_shape
    """
    if keys is None:
        keys = AMARA_TARGET_KEYS
    keys = tuple(keys)

    with load_amara_maps(galaxy_dir) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    suffix = "_scaled" if scaled else "_raw"
    targets = {key: arrays[f"{key}{suffix}"].astype(np.float32) for key in keys}
    target_valid_masks = {key: arrays[f"{key}_valid_mask"].astype(np.uint8) for key in keys}
    target_loss_masks = {key: _loss_mask_from_npz(arrays, key) for key in keys}

    return {
        "targets": targets,
        "target_valid_masks": target_valid_masks,
        "target_loss_masks": target_loss_masks,
        "footprint_mask": arrays["native_footprint_mask"].astype(np.uint8),
        "native_shape": tuple(int(x) for x in arrays["native_shape"]),
        "target_shape": tuple(int(x) for x in arrays["target_shape"]),
    }


def patch_amara_footprint(pipe3d_path, npz_path, *, target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE)):
    """Update footprint and loss masks in an existing amara_maps.npz from SELECT_REG."""
    pipe3d_path = Path(pipe3d_path)
    npz_path = Path(npz_path)
    with np.load(npz_path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    native_y, native_x = int(arrays["native_ny"]), int(arrays["native_nx"])
    footprint = extract_select_reg_footprint(pipe3d_path, (native_y, native_x))
    padded_footprint = center_pad(footprint, target_shape, pad_value=0).astype(np.uint8)
    arrays["native_footprint_mask"] = padded_footprint

    for key in AMARA_TARGET_KEYS:
        valid_key = f"{key}_valid_mask"
        if valid_key not in arrays:
            scaled = arrays.get(f"{key}_scaled")
            if scaled is not None:
                arrays[valid_key] = np.isfinite(scaled).astype(np.uint8)
        arrays[f"{key}_loss_mask"] = (
            padded_footprint.astype(bool) & arrays[valid_key].astype(bool)
        ).astype(np.uint8)

    np.savez_compressed(npz_path, **arrays)
    return {
        "npz": npz_path,
        "footprint_spaxels": int(padded_footprint.sum()),
        "native_shape": (native_y, native_x),
    }


def write_collaborator_maps(
    pipe3d_path,
    out_dir,
    target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    clip=True,
):
    pipe3d_path = Path(pipe3d_path)
    plateifu = infer_plateifu_from_path(pipe3d_path)
    galaxy_dir = Path(out_dir) / plateifu
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = build_collaborator_arrays(pipe3d_path, target_shape=target_shape, clip=clip)
    size_label = f"{int(target_shape[0])}x{int(target_shape[1])}"
    npz_path = galaxy_dir / f"{plateifu}_pipe3d_direct_maps_{size_label}.npz"
    json_path = galaxy_dir / f"{plateifu}_pipe3d_direct_maps_{size_label}_metadata.json"
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"plateifu": plateifu, "npz": npz_path, "metadata": json_path, **metadata}
