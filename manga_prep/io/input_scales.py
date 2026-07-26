"""
Per-channel asinh soft-scale factors for imaging and spectra.

Scales ``s_b`` are training-split percentiles of |flux| (finite samples), stored under
``manga_sdss_fits/stats/input_asinh_scales.json``. Runtime applies ``asinh(f / s_b)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PERCENTILES: tuple[float, ...] = (95.0, 99.0, 99.5)
DEFAULT_SCALES_PATH = Path("manga_sdss_fits/stats/input_asinh_scales.json")

# Accept common aliases for 99.5 in config / CLI.
_PERCENTILE_ALIASES: dict[str, float] = {
    "95": 95.0,
    "99": 99.0,
    "99.5": 99.5,
    "995": 99.5,
    "99_5": 99.5,
}


def normalize_percentile(value: float | int | str) -> float:
    """Map config/CLI percentile to a canonical float (95, 99, or 99.5)."""
    key = str(value).strip().lower().replace(" ", "")
    if key in _PERCENTILE_ALIASES:
        return _PERCENTILE_ALIASES[key]
    p = float(value)
    if abs(p - 99.5) < 1e-6 or abs(p - 995.0) < 1e-6:
        return 99.5
    if abs(p - 95.0) < 1e-6:
        return 95.0
    if abs(p - 99.0) < 1e-6:
        return 99.0
    raise ValueError(
        f"Unsupported percentile {value!r}; expected one of 95, 99, 99.5 (alias 995)."
    )


def percentile_key(value: float | int | str) -> str:
    p = normalize_percentile(value)
    return "99.5" if abs(p - 99.5) < 1e-6 else f"{p:g}"


def load_input_scales(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input asinh scales not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if int(data.get("version", 0)) < 1:
        raise ValueError(f"Unrecognized scales file version in {path}")
    return data


def save_input_scales(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def imaging_scales_for_percentile(
    scales: dict[str, Any],
    *,
    survey: str,
    percentile: float | int | str,
) -> tuple[tuple[str, ...], list[float]]:
    """Return (bands, s_b list) for SDSS or Legacy at the chosen percentile."""
    block = scales.get(survey)
    if not block:
        raise KeyError(f"Scales file has no '{survey}' block")
    key = percentile_key(percentile)
    if key not in block["scales"]:
        raise KeyError(f"{survey} scales missing percentile {key!r}; have {sorted(block['scales'])}")
    bands = tuple(str(b) for b in block["bands"])
    values = [float(v) for v in block["scales"][key]]
    if len(values) != len(bands):
        raise ValueError(f"{survey} bands/scales length mismatch: {len(bands)} vs {len(values)}")
    if any(v <= 0 for v in values):
        raise ValueError(f"{survey} scales must be > 0, got {values}")
    return bands, values


def spectrum_scale_for_percentile(
    scales: dict[str, Any],
    *,
    mode: str,
    percentile: float | int | str,
) -> float:
    """Return scalar s for fake or real spectrum flux at the chosen percentile."""
    if mode not in ("fake", "real"):
        raise ValueError(f"spectrum mode must be 'fake' or 'real', got {mode!r}")
    block_key = f"spectrum_{mode}"
    block = scales.get(block_key)
    if not block:
        raise KeyError(f"Scales file has no '{block_key}' block")
    key = percentile_key(percentile)
    if key not in block["scales"]:
        raise KeyError(
            f"{block_key} scales missing percentile {key!r}; have {sorted(block['scales'])}"
        )
    value = float(block["scales"][key])
    if value <= 0:
        raise ValueError(f"{block_key} scale must be > 0, got {value}")
    return value


def resolve_runtime_asinh_scales(
    scales_path: Path | str,
    *,
    imaging_percentile: float | int | str = 99,
    spectrum_percentile: float | int | str = 99,
    use_sdss: bool = True,
    use_legacy: bool = False,
) -> tuple[list[float], float, float]:
    """
    Load scales JSON and return (imaging_s_list, spectrum_fake_s, spectrum_real_s).

    Imaging channel order matches model concat: SDSS ugriz then Legacy bands.
    """
    scales = load_input_scales(scales_path)
    imaging: list[float] = []
    if use_sdss:
        _bands, values = imaging_scales_for_percentile(
            scales, survey="sdss", percentile=imaging_percentile
        )
        imaging.extend(values)
    if use_legacy:
        _bands, values = imaging_scales_for_percentile(
            scales, survey="legacy", percentile=imaging_percentile
        )
        imaging.extend(values)
    if not imaging:
        raise ValueError("resolve_runtime_asinh_scales requires use_sdss and/or use_legacy")
    s_fake = spectrum_scale_for_percentile(
        scales, mode="fake", percentile=spectrum_percentile
    )
    s_real = spectrum_scale_for_percentile(
        scales, mode="real", percentile=spectrum_percentile
    )
    return imaging, s_fake, s_real


def default_scales_path(data_root: Path | str) -> Path:
    return Path(data_root) / "stats" / "input_asinh_scales.json"


def ensure_input_asinh_scales(
    *,
    data_top: dict[str, Any],
    model_top: dict[str, Any],
    imaging_resolution: str = "aligned",
    auto_compute: bool | None = None,
) -> Path:
    """
    Resolve ``input_norm.scales_path``, computing train-split scales if the file is missing.

    Returns the path to a readable scales JSON.
    """
    from manga_prep.export.compute_input_scales import compute_input_scales

    norm_top = model_top.get("input_norm", {}) or {}
    data_root = Path(data_top.get("data_root", "manga_sdss_fits"))
    scales_path = Path(
        norm_top.get("scales_path") or default_scales_path(data_root)
    )
    if auto_compute is None:
        auto_compute = bool(norm_top.get("auto_compute", True))

    if scales_path.is_file() and scales_path.stat().st_size > 0:
        return scales_path

    if not auto_compute:
        raise FileNotFoundError(
            f"Input asinh scales not found: {scales_path}\n"
            f"Run: python -m manga_prep compute-input-scales "
            f"(or set model.input_norm.auto_compute=true)."
        )

    split_csv = Path(
        data_top.get("split", {}).get(
            "split_csv_path",
            str(data_root / "splits" / "default_split.csv"),
        )
    )
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Cannot auto-compute asinh scales: split CSV missing ({split_csv}). "
            f"Create it with: python -m src.data.make_splits"
        )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Cannot auto-compute asinh scales: data root missing ({data_root})."
        )

    aligned_oversample = data_top.get("aligned_oversample")
    use_legacy = bool(data_top.get("use_legacy", False))
    print(
        f"  input_norm: scales file missing ({scales_path}); "
        f"computing train-split asinh scales "
        f"(imaging_resolution={imaging_resolution})…",
        flush=True,
    )
    payload = compute_input_scales(
        data_root=data_root,
        split_csv=split_csv,
        split="train",
        imaging_resolution=imaging_resolution,
        aligned_oversample=None if aligned_oversample is None else int(aligned_oversample),
        use_legacy=use_legacy,
    )
    save_input_scales(scales_path, payload)
    print(f"  input_norm: wrote {scales_path}", flush=True)
    return scales_path
