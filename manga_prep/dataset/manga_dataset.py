"""
PyTorch dataset: imaging + optional spectrum -> Amara map targets (UNet training).

Inputs: SDSS/Legacy cutouts, optional 1D spectrum (real or fake SDSS fiber).
Targets: 6 Amara Pipe3D maps (0-1 scaled) on a fixed 76x76 canvas with loss masks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS, load_amara_training_targets
from manga_prep.io.fits_io import open_fits
from manga_prep.io.aligned_cache import (
    ImagingGrid,
    aligned_legacy_path_from_row,
    aligned_sdss_path_from_row,
    export_legacy_aligned,
    export_sdss_aligned,
    load_aligned_imaging,
)
from manga_prep.io.imaging_alignment import (
    SDSS_NATIVE_CANVAS,
    _pipe3d_cube_path,
    amara_aligned_pixel_shape,
    reproject_cutout_stack_to_amara_grid,
    reproject_cutout_stack_to_sdss_native_grid,
)
from manga_prep.dataset.index import (
    build_manga_dataset_index,
    legacy_imaging_ready,
    read_manga_dataset_index,
    sdss_imaging_ready,
    write_manga_dataset_index,
)

SpectrumMode = Literal["real", "fake"] | None

_SDSS_BANDS = ("u", "g", "r", "i", "z")
_LEGACY_BANDS = ("g", "r", "i", "z")

# Fixed canvas for SDSS-native aligned HR imaging (and debug raw stacks).
NATIVE_IMAGING_CANVAS = SDSS_NATIVE_CANVAS


def _load_fits_image(path: Path) -> np.ndarray:
    with open_fits(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}, got shape {data.shape}")
    return data


DEFAULT_SPECTRUM_WAVE_MIN = 3622.0
DEFAULT_SPECTRUM_WAVE_MAX = 10354.0
DEFAULT_SPECTRUM_N_WAVE = 4563


def _center_crop_2d(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Center-crop a 2D array to ``(crop_h, crop_w)``."""
    h, w = image.shape
    if h == crop_h and w == crop_w:
        return image
    if h < crop_h or w < crop_w:
        raise ValueError(f"Cannot crop {image.shape} to ({crop_h}, {crop_w})")
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _center_pad_2d(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-pad or center-crop a 2D array to ``(target_h, target_w)``."""
    h, w = image.shape
    if h > target_h or w > target_w:
        return _center_crop_2d(image, target_h, target_w)
    if h == target_h and w == target_w:
        return image
    out = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = image
    return out


def _stack_native_imaging_bands(
    paths: list[Path],
    *,
    canvas: int = NATIVE_IMAGING_CANVAS,
) -> np.ndarray:
    """
    Load native FITS cutouts, align band shapes, and pad to a fixed square canvas.

    Some galaxies have mixed 128×128 / 196×196 band downloads; we center-crop all
    bands to the smallest common shape, then center-pad to ``canvas`` for batching.
    """
    bands = [_load_fits_image(path) for path in paths]
    min_h = min(a.shape[0] for a in bands)
    min_w = min(a.shape[1] for a in bands)
    bands = [_center_crop_2d(a, min_h, min_w) for a in bands]
    stack = np.stack(bands, axis=0)
    if stack.shape[1] != canvas or stack.shape[2] != canvas:
        stack = np.stack(
            [_center_pad_2d(b, canvas, canvas) for b in stack],
            axis=0,
        )
    return stack


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def default_spectrum_wave_grid(
    n_wave: int = DEFAULT_SPECTRUM_N_WAVE,
    wave_min: float = DEFAULT_SPECTRUM_WAVE_MIN,
    wave_max: float = DEFAULT_SPECTRUM_WAVE_MAX,
) -> np.ndarray:
    return np.linspace(wave_min, wave_max, n_wave, dtype=np.float32)


def resample_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    target_wave: np.ndarray,
    ivar: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float32)
    target_wave = np.asarray(target_wave, dtype=np.float32)

    good = np.isfinite(wave) & np.isfinite(flux)
    if good.sum() < 2:
        out_flux = np.full(target_wave.shape, np.nan, dtype=np.float32)
        out_ivar = np.full(target_wave.shape, np.nan, dtype=np.float32)
    else:
        order = np.argsort(wave[good])
        w = wave[good][order]
        f = flux[good][order]
        out_flux = np.interp(target_wave, w, f, left=0.0, right=0.0).astype(np.float32)
        if ivar is not None:
            iv = np.asarray(ivar, dtype=np.float32)[good][order]
            out_ivar = np.interp(target_wave, w, iv, left=0.0, right=0.0).astype(np.float32)
        else:
            out_ivar = np.full(target_wave.shape, np.nan, dtype=np.float32)

    return {"wave": target_wave, "flux": out_flux, "ivar": out_ivar}


class MangaGalaxyDataset(Dataset):
    """
    Lazy-loading dataset for image(+spectrum) -> Amara map prediction.

    Parameters
    ----------
    include_sdss_imaging, include_legacy_imaging:
        Input cutout FITS stacks.
    include_targets:
        Load Amara map targets and masks (required for UNet training).
    spectrum:
        None = no spectrum input; "real" = SDSS fiber (falls back to fake if missing
        when spectrum_fallback=True); "fake" = MaNGA aperture coadd spectrum.
    require_all:
        Keep only galaxies with every requested input/target available.
    target_scaled:
        If True (default), targets are Amara 0-1 scaled maps. If False, raw physical maps.
    """

    def __init__(
        self,
        data_root: Path | str = "manga_sdss_fits",
        index_path: Path | str | None = None,
        *,
        include_sdss_imaging: bool = False,
        include_legacy_imaging: bool = False,
        include_targets: bool = True,
        spectrum: SpectrumMode = None,
        spectrum_fallback: bool = True,
        require_all: bool = True,
        target_scaled: bool = True,
        resample_spectrum: bool = True,
        spectrum_wave_grid: np.ndarray | None = None,
        align_imaging_to_amara_grid: bool = True,
        prefer_aligned_cache: bool = True,
        imaging_grid: ImagingGrid = "amara",
        aligned_oversample: int = 1,
        write_aligned_cache: bool = True,
        include_hr_imaging: bool = False,
        hr_survey: Literal["sdss", "legacy"] = "sdss",
        rebuild_index: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.include_sdss_imaging = include_sdss_imaging
        self.include_legacy_imaging = include_legacy_imaging
        self.include_targets = include_targets
        self.spectrum = spectrum
        self.spectrum_fallback = spectrum_fallback
        self.require_all = require_all
        self.target_scaled = target_scaled
        self.resample_spectrum = resample_spectrum
        self.align_imaging_to_amara_grid = bool(align_imaging_to_amara_grid)
        self.prefer_aligned_cache = bool(prefer_aligned_cache)
        self.imaging_grid: ImagingGrid = imaging_grid  # type: ignore[assignment]
        if self.imaging_grid not in ("amara", "sdss_native"):
            raise ValueError(f"imaging_grid must be 'amara' or 'sdss_native', got {imaging_grid!r}")
        self.aligned_oversample = int(aligned_oversample)
        self.write_aligned_cache = bool(write_aligned_cache)
        self.include_hr_imaging = bool(include_hr_imaging)
        self.hr_survey = hr_survey
        if self.hr_survey not in ("sdss", "legacy"):
            raise ValueError(f"hr_survey must be 'sdss' or 'legacy', got {hr_survey!r}")
        if self.include_hr_imaging and self.imaging_grid != "amara":
            raise ValueError(
                "include_hr_imaging requires imaging_grid='amara' "
                "(76×76 backbone + separate HR stream)."
            )
        if self.imaging_grid == "sdss_native":
            self.aligned_oversample = 1
        if self.aligned_oversample < 1:
            raise ValueError(f"aligned_oversample must be >= 1, got {aligned_oversample}")
        self.spectrum_wave_grid = (
            np.asarray(spectrum_wave_grid, dtype=np.float32)
            if spectrum_wave_grid is not None
            else default_spectrum_wave_grid()
        )

        if not self.align_imaging_to_amara_grid:
            raise ValueError(
                "align_imaging_to_amara_grid=False is not supported for training. "
                "All survey imaging must be WCS-reprojected in the dataloader "
                "(Amara grid or SDSS-native Amara-oriented grid)."
            )

        if not any((include_sdss_imaging, include_legacy_imaging, include_targets, spectrum is not None)):
            raise ValueError("Enable at least one input modality, targets, or spectrum.")

        index_path = Path(index_path) if index_path is not None else self.data_root / "manga_dataset_index.csv"
        if rebuild_index or not index_path.is_file():
            rows = build_manga_dataset_index(self.data_root)
            write_manga_dataset_index(rows, index_path)
        else:
            rows = read_manga_dataset_index(index_path)

        self.index_path = index_path
        self.rows = self._filter_rows(rows)

    def _requested_flags(self) -> list[tuple[str, bool]]:
        flags: list[tuple[str, bool]] = []
        if self.include_sdss_imaging:
            flags.append(("has_sdss_imaging", True))
        if self.include_legacy_imaging:
            flags.append(("has_legacy_imaging", True))
        if self.include_targets:
            flags.append(("has_amara_maps", True))
        if self.spectrum == "fake":
            flags.append(("has_fake_spectrum", True))
        elif self.spectrum == "real" and not self.spectrum_fallback:
            flags.append(("has_real_spectrum", True))
        elif self.spectrum == "real" and self.spectrum_fallback:
            flags.append(("has_fake_spectrum", True))
        return flags

    def _filter_rows(self, rows: list[dict]) -> list[dict]:
        if self.require_all:
            flags = self._requested_flags()
            filtered = [row for row in rows if all(row[flag] == required for flag, required in flags)]
        else:
            filtered = rows

        if self.include_sdss_imaging:
            filtered = [row for row in filtered if sdss_imaging_ready(self.data_root, row)]
        if self.include_legacy_imaging:
            filtered = [row for row in filtered if legacy_imaging_ready(self.data_root, row)]
        if self.include_hr_imaging:
            if self.hr_survey == "sdss":
                filtered = [row for row in filtered if sdss_imaging_ready(self.data_root, row)]
            else:
                filtered = [row for row in filtered if legacy_imaging_ready(self.data_root, row)]
        return filtered

    def __len__(self) -> int:
        return len(self.rows)

    def _galaxy_dir(self, row: dict) -> Path:
        return self.data_root / row["galaxy_dir"]

    def _target_shape(self, row: dict) -> tuple[int, int]:
        if row.get("amara_maps_npz"):
            with np.load(self.data_root / row["amara_maps_npz"]) as archive:
                return tuple(int(x) for x in archive["target_shape"])
        from manga_prep.targets.pipe3d_maps import DEFAULT_TARGET_SIZE

        return (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE)

    def _imaging_pixel_shape(self, row: dict) -> tuple[int, int]:
        """Expected imaging canvas for the configured grid."""
        if self.imaging_grid == "sdss_native":
            return (NATIVE_IMAGING_CANVAS, NATIVE_IMAGING_CANVAS)
        return amara_aligned_pixel_shape(self._target_shape(row), oversample=self.aligned_oversample)

    def _load_aligned_cache(self, row: dict, *, survey: str) -> dict[str, object] | None:
        if not self.prefer_aligned_cache:
            return None
        if survey == "sdss":
            cache_path = aligned_sdss_path_from_row(
                self.data_root,
                row,
                grid=self.imaging_grid,
                oversample=self.aligned_oversample,
            )
        elif survey == "legacy":
            cache_path = aligned_legacy_path_from_row(
                self.data_root,
                row,
                grid=self.imaging_grid,
                oversample=self.aligned_oversample,
            )
        else:
            raise ValueError(f"Unknown survey: {survey!r}")
        if not cache_path.is_file() or cache_path.stat().st_size == 0:
            return None
        cached = load_aligned_imaging(cache_path)
        data = np.asarray(cached["data"])
        expected = self._imaging_pixel_shape(row)
        if data.shape[-2:] != expected:
            return None
        cached["aligned_oversample"] = self.aligned_oversample
        cached["grid"] = self.imaging_grid
        return cached

    def _load_imaging_stack(
        self,
        row: dict,
        *,
        cutout_subdir: str,
        file_prefix: str,
        bands: tuple[str, ...],
    ) -> dict[str, object]:
        gal_dir = self._galaxy_dir(row)
        plate, ifu = row["plateifu"].split("-", 1)
        paths = [gal_dir / cutout_subdir / f"{file_prefix}-{plate}-{ifu}-{b}.fits" for b in bands]
        pipe3d_path = _pipe3d_cube_path(gal_dir)
        target_shape = self._target_shape(row)
        if self.imaging_grid == "sdss_native":
            stack, scale = reproject_cutout_stack_to_sdss_native_grid(
                paths,
                pipe3d_path,
                shape_out=(NATIVE_IMAGING_CANVAS, NATIVE_IMAGING_CANVAS),
                target_shape=target_shape,
            )
            return {
                "bands": bands,
                "data": stack,
                "aligned_to_amara_grid": True,
                "aligned_oversample": 1,
                "grid": "sdss_native",
                "pixel_scale_arcsec": scale,
            }
        stack = reproject_cutout_stack_to_amara_grid(
            paths,
            pipe3d_path,
            target_shape=target_shape,
            oversample=self.aligned_oversample,
        )
        return {
            "bands": bands,
            "data": stack,
            "aligned_to_amara_grid": True,
            "aligned_oversample": self.aligned_oversample,
            "grid": "amara",
        }

    def _load_sdss_imaging(self, row: dict) -> dict[str, object]:
        cached = self._load_aligned_cache(row, survey="sdss")
        if cached is not None:
            return cached
        if self.write_aligned_cache:
            gal_dir = self._galaxy_dir(row)
            export_sdss_aligned(
                gal_dir,
                skip_existing=False,
                oversample=self.aligned_oversample,
                grid=self.imaging_grid,
                canvas=NATIVE_IMAGING_CANVAS,
            )
            cached = self._load_aligned_cache(row, survey="sdss")
            if cached is not None:
                return cached
        return self._load_imaging_stack(
            row,
            cutout_subdir="sdss_cutouts",
            file_prefix="sdss",
            bands=_SDSS_BANDS,
        )

    def _load_legacy_imaging(self, row: dict) -> dict[str, object]:
        cached = self._load_aligned_cache(row, survey="legacy")
        if cached is not None:
            return cached
        if self.write_aligned_cache:
            gal_dir = self._galaxy_dir(row)
            export_legacy_aligned(
                gal_dir,
                skip_existing=False,
                oversample=self.aligned_oversample,
                grid=self.imaging_grid,
                canvas=NATIVE_IMAGING_CANVAS,
            )
            cached = self._load_aligned_cache(row, survey="legacy")
            if cached is not None:
                return cached

        gal_dir = self._galaxy_dir(row)
        plate, ifu = row["plateifu"].split("-", 1)
        for band_set in (_LEGACY_BANDS, ("g", "r", "z")):
            paths = [gal_dir / "legacy_cutouts" / f"legacy-{plate}-{ifu}-{b}.fits" for b in band_set]
            if all(path.is_file() for path in paths):
                return self._load_imaging_stack(
                    row,
                    cutout_subdir="legacy_cutouts",
                    file_prefix="legacy",
                    bands=band_set,
                )
        raise FileNotFoundError(f"No consistent legacy imaging for {row['plateifu']}")

    def _with_imaging_grid(self, grid: ImagingGrid, oversample: int = 1):
        """Temporarily switch imaging grid for a nested load."""
        prev_grid = self.imaging_grid
        prev_os = self.aligned_oversample
        self.imaging_grid = grid
        self.aligned_oversample = 1 if grid == "sdss_native" else int(oversample)
        return prev_grid, prev_os

    def _load_hr_imaging(self, row: dict) -> dict[str, object]:
        """Load high-res morphology stream (SDSS-native or Legacy native)."""
        prev_grid, prev_os = self._with_imaging_grid("sdss_native", oversample=1)
        try:
            if self.hr_survey == "sdss":
                bundle = self._load_sdss_imaging(row)
            else:
                bundle = self._load_legacy_imaging(row)
        finally:
            self.imaging_grid = prev_grid
            self.aligned_oversample = prev_os
        bundle = dict(bundle)
        bundle["grid"] = "sdss_native"
        bundle["hr_survey"] = self.hr_survey
        return bundle

    def _load_spectrum(self, row: dict) -> dict[str, object]:
        if self.spectrum is None:
            raise RuntimeError("_load_spectrum called with spectrum=None")

        arrays: dict[str, np.ndarray] | None = None
        is_real = False

        if self.spectrum == "fake":
            if not row.get("has_fake_spectrum") or not row.get("fake_spectrum_npz"):
                raise FileNotFoundError(f"No fake spectrum for {row['plateifu']}")
            arrays = _load_npz_arrays(self.data_root / row["fake_spectrum_npz"])
        elif self.spectrum == "real":
            if row.get("has_real_spectrum") and row.get("real_spectrum_npz"):
                arrays = _load_npz_arrays(self.data_root / row["real_spectrum_npz"])
                is_real = True
            elif self.spectrum_fallback and row.get("has_fake_spectrum") and row.get("fake_spectrum_npz"):
                arrays = _load_npz_arrays(self.data_root / row["fake_spectrum_npz"])
                is_real = False
            else:
                raise FileNotFoundError(f"No real (or fallback fake) spectrum for {row['plateifu']}")
        else:
            raise ValueError(f"Unknown spectrum mode: {self.spectrum!r}")

        wave = arrays["wave"]
        flux = arrays["flux"]
        ivar = arrays.get("ivar")
        if self.resample_spectrum:
            resampled = resample_spectrum(wave, flux, self.spectrum_wave_grid, ivar=ivar)
            return {
                "wave": resampled["wave"],
                "flux": resampled["flux"],
                "ivar": resampled["ivar"],
                "is_real_sdss_fiber": is_real,
                "spectrum_mode": "real" if is_real else "fake",
            }

        return {
            "wave": wave.astype(np.float32),
            "flux": flux.astype(np.float32),
            "ivar": None if ivar is None else ivar.astype(np.float32),
            "is_real_sdss_fiber": is_real,
            "spectrum_mode": "real" if is_real else "fake",
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        sample: dict[str, object] = {
            "plateifu": row["plateifu"],
            "index": index,
        }
        if row.get("ra_deg"):
            sample["ra_deg"] = float(row["ra_deg"])
        if row.get("dec_deg"):
            sample["dec_deg"] = float(row["dec_deg"])

        inputs: dict[str, object] = {}
        if self.include_sdss_imaging:
            inputs["sdss_imaging"] = self._load_sdss_imaging(row)
        if self.include_legacy_imaging:
            inputs["legacy_imaging"] = self._load_legacy_imaging(row)
        if self.include_hr_imaging:
            inputs["hr_imaging"] = self._load_hr_imaging(row)
        if self.spectrum is not None:
            inputs["spectrum"] = self._load_spectrum(row)
        if inputs:
            sample["inputs"] = inputs

        if self.include_targets:
            target_bundle = load_amara_training_targets(
                self._galaxy_dir(row),
                scaled=self.target_scaled,
            )
            sample["targets"] = target_bundle["targets"]
            sample["target_valid_masks"] = target_bundle["target_valid_masks"]
            sample["target_loss_masks"] = target_bundle["target_loss_masks"]
            sample["footprint_mask"] = target_bundle["footprint_mask"]
            sample["native_shape"] = target_bundle["native_shape"]
            sample["target_shape"] = target_bundle["target_shape"]

        return sample


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked MSE for map prediction. pred/target/mask shape: (B, H, W) or (B, C, H, W)."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    mask = loss_mask.to(dtype=pred.dtype)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(1)
    diff2 = (pred - target) ** 2
    masked = torch.where(mask > 0, diff2, torch.zeros_like(diff2))
    return masked.sum() / mask.sum().clamp_min(eps)


def collate_manga_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate samples for UNet training."""
    out: dict[str, object] = {
        "plateifu": [item["plateifu"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }
    if "ra_deg" in batch[0]:
        out["ra_deg"] = torch.tensor([item["ra_deg"] for item in batch], dtype=torch.float64)
    if "dec_deg" in batch[0]:
        out["dec_deg"] = torch.tensor([item["dec_deg"] for item in batch], dtype=torch.float64)

    if "inputs" in batch[0]:
        inputs: dict[str, object] = {}
        first_inputs = batch[0]["inputs"]
        for key in ("sdss_imaging", "legacy_imaging", "hr_imaging"):
            if key in first_inputs:
                inputs[key] = torch.from_numpy(
                    np.stack([item["inputs"][key]["data"] for item in batch], axis=0)
                )
                inputs[f"{key}_bands"] = first_inputs[key]["bands"]
        if "spectrum" in first_inputs:
            ivars = []
            for item in batch:
                iv = item["inputs"]["spectrum"].get("ivar")
                flux = item["inputs"]["spectrum"]["flux"]
                if iv is None:
                    iv = np.ones_like(flux, dtype=np.float32)
                ivars.append(np.asarray(iv, dtype=np.float32))
            inputs["spectrum"] = {
                "wave": torch.from_numpy(
                    np.stack([item["inputs"]["spectrum"]["wave"] for item in batch], axis=0)
                ),
                "flux": torch.from_numpy(
                    np.stack([item["inputs"]["spectrum"]["flux"] for item in batch], axis=0)
                ),
                "ivar": torch.from_numpy(np.stack(ivars, axis=0)),
                "is_real_sdss_fiber": torch.tensor(
                    [item["inputs"]["spectrum"]["is_real_sdss_fiber"] for item in batch],
                    dtype=torch.bool,
                ),
            }
        out["inputs"] = inputs

    if "targets" in batch[0]:
        out["targets"] = {
            key: torch.from_numpy(
                np.stack([item["targets"][key] for item in batch], axis=0)
            )
            for key in AMARA_TARGET_KEYS
        }
        out["target_valid_masks"] = {
            key: torch.from_numpy(
                np.stack([item["target_valid_masks"][key] for item in batch], axis=0)
            )
            for key in AMARA_TARGET_KEYS
        }
        out["target_loss_masks"] = {
            key: torch.from_numpy(
                np.stack([item["target_loss_masks"][key] for item in batch], axis=0)
            )
            for key in AMARA_TARGET_KEYS
        }
        out["footprint_mask"] = torch.from_numpy(
            np.stack([item["footprint_mask"] for item in batch], axis=0)
        )

    return out
