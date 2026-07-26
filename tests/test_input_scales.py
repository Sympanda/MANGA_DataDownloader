"""Unit tests for asinh input soft-scales."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from manga_prep.io.input_scales import (
    normalize_percentile,
    percentile_key,
    resolve_runtime_asinh_scales,
    save_input_scales,
)
from src.models.config import ModelConfig
from src.models.wrapper import prepare_imaging_input, prepare_spectrum_input


class PercentileHelpersTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_percentile(95), 95.0)
        self.assertEqual(normalize_percentile(99), 99.0)
        self.assertEqual(normalize_percentile(99.5), 99.5)
        self.assertEqual(normalize_percentile(995), 99.5)
        self.assertEqual(normalize_percentile("995"), 99.5)
        self.assertEqual(percentile_key(995), "99.5")


class ResolveScalesTests(unittest.TestCase):
    def test_resolve_runtime(self) -> None:
        payload = {
            "version": 1,
            "sdss": {
                "bands": ["u", "g", "r", "i", "z"],
                "scales": {
                    "95": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "99": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "99.5": [11.0, 21.0, 31.0, 41.0, 51.0],
                },
            },
            "spectrum_fake": {"scales": {"95": 1.0, "99": 2.0, "99.5": 3.0}},
            "spectrum_real": {"scales": {"95": 4.0, "99": 5.0, "99.5": 6.0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scales.json"
            save_input_scales(path, payload)
            imaging, s_fake, s_real = resolve_runtime_asinh_scales(
                path,
                imaging_percentile=99,
                spectrum_percentile=99.5,
                use_sdss=True,
                use_legacy=False,
            )
        self.assertEqual(imaging, [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(s_fake, 3.0)
        self.assertEqual(s_real, 6.0)


class EnsureScalesTests(unittest.TestCase):
    def test_ensure_computes_when_missing(self) -> None:
        from unittest.mock import patch

        from manga_prep.io.input_scales import ensure_input_asinh_scales

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "stats" / "scales.json"
            data_top = {
                "data_root": str(root),
                "use_legacy": False,
                "split": {"split_csv_path": str(root / "splits" / "default_split.csv")},
            }
            model_top = {
                "input_norm": {
                    "mode": "asinh",
                    "scales_path": str(out),
                    "auto_compute": True,
                }
            }
            (root / "splits").mkdir(parents=True)
            (root / "splits" / "default_split.csv").write_text(
                "plateifu,galaxy_dir,split\n8485-1901,8485_1901,train\n",
                encoding="utf-8",
            )
            # data_root must exist as a directory
            fake_payload = {
                "version": 1,
                "sdss": {"bands": ["u"], "scales": {"99": [1.0]}},
                "spectrum_fake": {"scales": {"99": 1.0}},
                "spectrum_real": {"scales": {"99": 1.0}},
            }

            with patch(
                "manga_prep.export.compute_input_scales.compute_input_scales",
                return_value=fake_payload,
            ) as mocked:
                path = ensure_input_asinh_scales(
                    data_top=data_top,
                    model_top=model_top,
                    imaging_resolution="aligned",
                )
            mocked.assert_called_once()
            self.assertEqual(path, out)
            self.assertTrue(out.is_file())

    def test_ensure_skips_when_present(self) -> None:
        from unittest.mock import patch

        from manga_prep.io.input_scales import ensure_input_asinh_scales

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scales.json"
            save_input_scales(out, {"version": 1})
            data_top = {"data_root": tmp}
            model_top = {"input_norm": {"scales_path": str(out), "auto_compute": True}}
            with patch(
                "manga_prep.export.compute_input_scales.compute_input_scales"
            ) as mocked:
                path = ensure_input_asinh_scales(
                    data_top=data_top,
                    model_top=model_top,
                    imaging_resolution="aligned",
                )
            mocked.assert_not_called()
            self.assertEqual(path, out)


class AsinhPrepareTests(unittest.TestCase):
    def test_imaging_asinh(self) -> None:
        cfg = ModelConfig(
            use_sdss=True,
            use_legacy=False,
            use_spectrum=False,
            input_norm_mode="asinh",
            imaging_asinh_scales=[2.0, 2.0, 2.0, 2.0, 2.0],
            imaging_clamp_min=None,
            imaging_clamp_max=None,
            spectrum_asinh_scale_fake=1.0,
            spectrum_asinh_scale_real=1.0,
        )
        batch = {
            "inputs": {
                "sdss_imaging": torch.full((1, 5, 4, 4), 2.0),
            }
        }
        out = prepare_imaging_input(batch, cfg)
        # asinh(2/2) = asinh(1)
        expected = torch.asinh(torch.tensor(1.0))
        self.assertTrue(torch.allclose(out, expected.expand_as(out)))

    def test_spectrum_asinh_per_sample(self) -> None:
        cfg = ModelConfig(
            use_sdss=True,
            use_legacy=False,
            use_spectrum=True,
            spectrum_use_wavelength=False,
            spectrum_use_ivar=False,
            input_norm_mode="asinh",
            imaging_asinh_scales=[1.0, 1.0, 1.0, 1.0, 1.0],
            spectrum_asinh_scale_fake=2.0,
            spectrum_asinh_scale_real=4.0,
            imaging_clamp_min=None,
            imaging_clamp_max=None,
        )
        batch = {
            "inputs": {
                "spectrum": {
                    "flux": torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
                    "is_real_sdss_fiber": torch.tensor([False, True]),
                }
            }
        }
        out = prepare_spectrum_input(batch, cfg)
        assert out is not None
        # fake: asinh(2/2)=asinh(1); real: asinh(4/4)=asinh(1)
        expected = torch.asinh(torch.ones(2, 1, 2))
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
