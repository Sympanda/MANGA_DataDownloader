"""
Benchmark and smoke tests for MangaGalaxyDataset.

Run:
  python -m unittest tests.test_manga_dataset -v
"""
from __future__ import annotations

import time
import unittest
from pathlib import Path

import numpy as np

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS
from manga_prep.dataset.index import (
    inspect_galaxy_index_row,
    read_manga_dataset_index,
    summarize_index,
    write_manga_dataset_index,
)
from manga_prep.dataset.index import build_manga_dataset_index
from manga_prep.dataset.manga_dataset import MangaGalaxyDataset, collate_manga_batch, masked_mse_loss

DATA_ROOT = Path(__file__).resolve().parents[1] / "manga_sdss_fits"
INDEX_PATH = DATA_ROOT / "manga_dataset_index.csv"

MAX_SECONDS_PER_SAMPLE_UNET = 0.05
MAX_SECONDS_PER_SAMPLE_ALL = 0.15


class NativeImagingStackTests(unittest.TestCase):
    def test_stack_mixed_band_shapes(self) -> None:
        from manga_prep.dataset.manga_dataset import _center_crop_2d, _stack_native_imaging_bands
        from unittest.mock import patch

        bands = {
            "u": np.ones((196, 196), dtype=np.float32),
            "g": np.ones((196, 196), dtype=np.float32),
            "r": np.ones((128, 128), dtype=np.float32),
            "i": np.ones((196, 196), dtype=np.float32),
            "z": np.ones((196, 196), dtype=np.float32),
        }
        paths = [Path(f"/fake/{b}.fits") for b in ("u", "g", "r", "i", "z")]

        def fake_load(path: Path) -> np.ndarray:
            band = path.stem.split("-")[-1]
            return bands[band]

        with patch("manga_prep.dataset.manga_dataset._load_fits_image", side_effect=fake_load):
            stack = _stack_native_imaging_bands(paths, canvas=196)

        self.assertEqual(stack.shape, (5, 196, 196))
        cropped_r = _center_crop_2d(bands["r"], 128, 128)
        np.testing.assert_array_equal(stack[2, 34:162, 34:162], cropped_r)


class AlignmentPolicyTests(unittest.TestCase):
    def test_align_false_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MangaGalaxyDataset(
                DATA_ROOT if DATA_ROOT.is_dir() else Path("."),
                INDEX_PATH if INDEX_PATH.is_file() else None,
                include_sdss_imaging=False,
                include_targets=False,
                spectrum=None,
                require_all=False,
                align_imaging_to_amara_grid=False,
                rebuild_index=False,
            )

    def test_amara_oversample_shapes(self) -> None:
        from manga_prep.io.imaging_alignment import amara_aligned_pixel_shape
        from manga_prep.io.aligned_cache import aligned_sdss_path

        self.assertEqual(amara_aligned_pixel_shape((76, 76), oversample=1), (76, 76))
        self.assertEqual(amara_aligned_pixel_shape((76, 76), oversample=2), (152, 152))
        self.assertEqual(aligned_sdss_path("x", grid="amara").name, "sdss_aligned.npz")
        self.assertEqual(aligned_sdss_path("x", grid="sdss_native").name, "sdss_aligned_native.npz")

    def test_data_config_grid_defaults(self) -> None:
        from src.data.make_dataloader import DataConfig

        self.assertEqual(DataConfig(imaging_resolution="aligned").resolve_imaging_grid(), "amara")
        self.assertEqual(DataConfig(imaging_resolution="native").resolve_imaging_grid(), "sdss_native")
        self.assertEqual(DataConfig(imaging_resolution="aligned").resolve_aligned_oversample(), 1)
        self.assertEqual(DataConfig(imaging_resolution="native").resolve_aligned_oversample(), 1)
        self.assertEqual(
            DataConfig(imaging_resolution="aligned", aligned_oversample=3).resolve_aligned_oversample(),
            3,
        )


def _ensure_index() -> None:
    if not INDEX_PATH.is_file():
        rows = build_manga_dataset_index(DATA_ROOT)
        write_manga_dataset_index(rows, INDEX_PATH)


@unittest.skipUnless(DATA_ROOT.is_dir(), "manga_sdss_fits not present")
class MangaDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_index()
        cls.summary = summarize_index(read_manga_dataset_index(INDEX_PATH))

    def test_unet_sample_structure(self) -> None:
        dataset = MangaGalaxyDataset(
            DATA_ROOT,
            INDEX_PATH,
            include_sdss_imaging=True,
            include_targets=True,
            spectrum="fake",
            require_all=True,
        )
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)
        self.assertIn("target_loss_masks", sample)
        self.assertIn("footprint_mask", sample)
        self.assertIn("ha_flux", sample["targets"])
        self.assertEqual(sample["targets"]["ha_flux"].shape, (76, 76))
        self.assertEqual(sample["inputs"]["sdss_imaging"]["data"].shape, (5, 76, 76))
        self.assertTrue(sample["inputs"]["sdss_imaging"]["aligned_to_amara_grid"])
        self.assertEqual(sample["inputs"]["sdss_imaging"].get("aligned_oversample", 1), 1)
        self.assertEqual(sample["footprint_mask"].shape, (76, 76))

    def test_oversampled_imaging_still_aligned(self) -> None:
        dataset = MangaGalaxyDataset(
            DATA_ROOT,
            INDEX_PATH,
            include_sdss_imaging=True,
            include_targets=True,
            spectrum=None,
            require_all=True,
            prefer_aligned_cache=False,
            imaging_grid="sdss_native",
        )
        sample = dataset[0]
        self.assertEqual(sample["inputs"]["sdss_imaging"]["data"].shape, (5, 196, 196))
        self.assertTrue(sample["inputs"]["sdss_imaging"]["aligned_to_amara_grid"])
        self.assertEqual(sample["inputs"]["sdss_imaging"].get("grid"), "sdss_native")
        self.assertEqual(sample["targets"]["ha_flux"].shape, (76, 76))

    def test_spectrum_none_excludes_spectrum(self) -> None:
        dataset = MangaGalaxyDataset(
            DATA_ROOT,
            INDEX_PATH,
            include_sdss_imaging=True,
            include_targets=True,
            spectrum=None,
            require_all=True,
        )
        sample = dataset[0]
        self.assertNotIn("spectrum", sample.get("inputs", {}))

    def test_collate_batch(self) -> None:
        dataset = MangaGalaxyDataset(
            DATA_ROOT,
            INDEX_PATH,
            include_sdss_imaging=True,
            include_targets=True,
            spectrum="fake",
            require_all=True,
        )
        batch = collate_manga_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["inputs"]["sdss_imaging"].shape[0], 2)
        self.assertEqual(batch["targets"]["ha_flux"].shape, (2, 76, 76))
        self.assertEqual(batch["target_loss_masks"]["ha_flux"].shape, (2, 76, 76))

    def test_masked_mse_loss(self) -> None:
        pred = np.zeros((2, 76, 76), dtype=np.float32)
        target = np.ones((2, 76, 76), dtype=np.float32)
        mask = np.zeros((2, 76, 76), dtype=np.float32)
        mask[:, 30:46, 30:46] = 1.0
        import torch

        loss = masked_mse_loss(
            torch.from_numpy(pred),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )
        self.assertAlmostEqual(float(loss), 1.0, places=5)

    def test_full_pass_unet_config_is_fast(self) -> None:
        dataset = MangaGalaxyDataset(
            DATA_ROOT,
            INDEX_PATH,
            include_sdss_imaging=True,
            include_targets=True,
            spectrum="fake",
            require_all=True,
        )
        n = min(len(dataset), 500)
        t0 = time.perf_counter()
        for i in range(n):
            sample = dataset[i]
            self.assertIn("targets", sample)
            self.assertIn("inputs", sample)
        elapsed = time.perf_counter() - t0
        per_sample = elapsed / n
        print(f"\n[benchmark] UNet config (subset n={n}): {1000 * per_sample:.1f} ms/sample")
        self.assertLess(per_sample, MAX_SECONDS_PER_SAMPLE_UNET)


@unittest.skipUnless(DATA_ROOT.is_dir(), "manga_sdss_fits not present")
class MangaDatasetIndexTests(unittest.TestCase):
    def test_index_row_fields(self) -> None:
        gal_dir = next(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.count("_") == 1)
        row = inspect_galaxy_index_row(gal_dir, data_root=DATA_ROOT)
        for field in ("plateifu", "has_amara_maps", "has_fake_spectrum", "has_sdss_imaging"):
            self.assertIn(field, row)


if __name__ == "__main__":
    unittest.main()
