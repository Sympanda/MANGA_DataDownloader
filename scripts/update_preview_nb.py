import json
from pathlib import Path

nb_path = Path("notebooks/manga_dataloader_preview.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

config_src = '''# Toggle which inputs to load (init is slow when SDSS/Legacy imaging is enabled)
INCLUDE_SDSS = True
INCLUDE_LEGACY = False
SPECTRUM = "fake"  # None | "real" | "fake"
ALIGN_IMAGING = True  # reproject cutouts onto Pipe3D / Amara spaxel WCS (76x76)

dataset = MangaGalaxyDataset(
    DATA_ROOT,
    INDEX_PATH,
    include_sdss_imaging=INCLUDE_SDSS,
    include_legacy_imaging=INCLUDE_LEGACY,
    include_targets=True,
    spectrum=SPECTRUM,
    align_imaging_to_amara_grid=ALIGN_IMAGING,
    require_all=True,
)

print(f"Dataset size: {len(dataset):,} galaxies")
print(f"Target channels: {list(AMARA_TARGET_KEYS)}")
'''

viz_src = '''def _percentile_norm(x, lo_pct=5, hi_pct=99):
    x = np.nan_to_num(x, nan=0.0)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(pos, [lo_pct, hi_pct])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def show_imaging_stack(title, stack, bands, *, rgb_bands=("r", "g", "i")):
    """Show every aligned input band plus RGB composite (76x76 spaxel grid)."""
    bidx = {b: i for i, b in enumerate(bands)}
    n = len(bands)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.4 * (n + 1), 2.6))
    if n + 1 == 1:
        axes = [axes]

    for ax, band in zip(axes[:n], bands):
        img = stack[bidx[band]]
        im = ax.imshow(_percentile_norm(img), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(band)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax_rgb = axes[-1]
    if all(b in bidx for b in rgb_bands):
        r, g, b = (_percentile_norm(stack[bidx[band]]) for band in rgb_bands)
        rgb = np.dstack([r, g, b])
        ax_rgb.imshow(rgb, origin="lower")
        ax_rgb.set_title("rgb (" + "".join(rgb_bands) + ")")
    ax_rgb.axis("off")

    fig.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()


def show_input_images(sample):
    inputs = sample.get("inputs", {})
    if not inputs:
        print("No inputs in sample")
        return

    aligned = " (aligned to Amara spaxel WCS)" if inputs.get("sdss_imaging", {}).get("aligned_to_amara_grid") else ""
    if "sdss_imaging" in inputs:
        show_imaging_stack(
            f"SDSS inputs{aligned} — {sample['plateifu']}",
            inputs["sdss_imaging"]["data"],
            inputs["sdss_imaging"]["bands"],
            rgb_bands=("r", "g", "i"),
        )

    if "legacy_imaging" in inputs:
        show_imaging_stack(
            f"Legacy inputs{aligned} — {sample['plateifu']}",
            inputs["legacy_imaging"]["data"],
            inputs["legacy_imaging"]["bands"],
            rgb_bands=("r", "g", "z"),
        )


def show_imaging_target_overlay(sample, band="r", target_key="ha_flux"):
    """Overlay aligned imaging on an Amara target map to verify orientation."""
    img_in = sample.get("inputs", {}).get("sdss_imaging")
    if img_in is None:
        print("Need SDSS imaging for overlay")
        return
    bidx = {b: i for i, b in enumerate(img_in["bands"])}
    if band not in bidx:
        band = img_in["bands"][0]
    image = _percentile_norm(img_in["data"][bidx[band]])
    target = sample["targets"][target_key]
    mask = sample["target_loss_masks"][target_key].astype(bool)
    target_masked = np.where(mask, target, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"SDSS {band} (aligned)")
    im1 = axes[1].imshow(target_masked, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(TARGET_LABELS.get(target_key, target_key))
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].imshow(image, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.55)
    axes[2].imshow(target_masked, origin="lower", cmap="magma", vmin=0, vmax=1, alpha=0.55)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Orientation check — {sample['plateifu']}")
    plt.tight_layout()
    plt.show()
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "INCLUDE_SDSS = True" in src and "MangaGalaxyDataset" in src:
        cell["source"] = [line + "\n" for line in config_src.splitlines()]
    if "def show_input_images(sample):" in src or "def show_imaging_stack" in src:
        cell["source"] = [line + "\n" for line in viz_src.splitlines()]

# replace show_input_images-only cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip() == "show_input_images(sample)":
        nb["cells"][i]["source"] = [
            "show_input_images(sample)\n",
            'show_imaging_target_overlay(sample, band="r", target_key="ha_flux")\n',
        ]

for cell in nb["cells"]:
    cell["outputs"] = []
    cell["execution_count"] = None

nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("updated", nb_path)
