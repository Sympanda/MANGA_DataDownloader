"""PyTorch dataset and dataset index."""

from manga_prep.dataset.index import build_manga_dataset_index, read_manga_dataset_index

__all__ = [
    "MangaGalaxyDataset",
    "collate_manga_batch",
    "masked_mse_loss",
    "build_manga_dataset_index",
    "read_manga_dataset_index",
]


def __getattr__(name: str):
    if name in {"MangaGalaxyDataset", "collate_manga_batch", "masked_mse_loss"}:
        from manga_prep.dataset import manga_dataset

        return getattr(manga_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
