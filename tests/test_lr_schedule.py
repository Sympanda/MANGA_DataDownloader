import unittest

from src.training.lr_schedule import resolve_lr, warmup_cosine_lr


class TestWarmupCosineLR(unittest.TestCase):
    def test_constant_schedule(self) -> None:
        self.assertEqual(resolve_lr(10, schedule="constant", peak_lr=1e-3, total_epochs=100), 1e-3)

    def test_warmup_starts_small(self) -> None:
        lr1 = warmup_cosine_lr(1, peak_lr=1e-3, total_epochs=100, warmup_epochs=5)
        self.assertAlmostEqual(lr1, 2e-4)

    def test_warmup_end_matches_peak(self) -> None:
        lr5 = warmup_cosine_lr(5, peak_lr=1e-3, total_epochs=100, warmup_epochs=5)
        self.assertAlmostEqual(lr5, 1e-3)

    def test_cosine_ends_at_min(self) -> None:
        lr_end = warmup_cosine_lr(100, peak_lr=1e-3, total_epochs=100, warmup_epochs=5, min_lr_ratio=0.01)
        self.assertAlmostEqual(lr_end, 1e-5)

    def test_monotone_decay_after_warmup(self) -> None:
        lrs = [
            warmup_cosine_lr(e, peak_lr=1e-3, total_epochs=50, warmup_epochs=5)
            for e in range(6, 51)
        ]
        self.assertTrue(all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)))


if __name__ == "__main__":
    unittest.main()
