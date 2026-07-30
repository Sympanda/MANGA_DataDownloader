"""Architecture ablation grid definitions (factorial over big knobs, no Optuna)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal


SpectrumMode = Literal["off", "on"]
GridName = Literal["core", "extended"]


@dataclass(frozen=True)
class ArchCell:
    """One architecture combination to train + evaluate."""

    name: str
    architecture: Literal["unet", "unetpp"]
    deep_supervision: bool
    spectrum: SpectrumMode  # off = no spectrum/FiLM; on = attention + λ + ivar + encoder FiLM
    hr_cross_attn: bool
    film_injection: Literal["none", "encoder"] | None = None  # None → derived from spectrum
    note: str = ""

    def resolved_film(self) -> str:
        if self.film_injection is not None:
            return self.film_injection
        return "encoder" if self.spectrum == "on" else "none"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["film_injection"] = self.resolved_film()
        return d


def core_grid() -> list[ArchCell]:
    """
    Compact grid answering the main architecture questions (~8 runs).

    Compares: UNet vs UNet++, deep supervision, spectrum package, HR cross-attn.
    """
    return [
        ArchCell(
            "A_unet",
            "unet",
            False,
            "off",
            False,
            note="Plain UNet, imaging-only",
        ),
        ArchCell(
            "B_unetpp",
            "unetpp",
            False,
            "off",
            False,
            note="UNet++ without deep supervision",
        ),
        ArchCell(
            "C_unetpp_ds",
            "unetpp",
            True,
            "off",
            False,
            note="UNet++ + DS (Model B baseline)",
        ),
        ArchCell(
            "D_unetpp_ds_spec",
            "unetpp",
            True,
            "on",
            False,
            note="UNet++ + DS + spectrum (Model C)",
        ),
        ArchCell(
            "E_unetpp_ds_hr",
            "unetpp",
            True,
            "off",
            True,
            note="UNet++ + DS + HR xattn (Model D)",
        ),
        ArchCell(
            "F_unetpp_ds_spec_hr",
            "unetpp",
            True,
            "on",
            True,
            note="UNet++ + DS + spectrum + HR (Model E)",
        ),
        ArchCell(
            "G_unet_spec",
            "unet",
            False,
            "on",
            False,
            note="UNet + spectrum (vs D: is UNet++ needed with spectrum?)",
        ),
        ArchCell(
            "H_unetpp_spec_nods",
            "unetpp",
            False,
            "on",
            False,
            note="UNet++ + spectrum without DS (DS ablate)",
        ),
    ]


def extended_grid() -> list[ArchCell]:
    """Core plus film=none control with spectrum features still loaded."""
    cells = list(core_grid())
    cells.append(
        ArchCell(
            "I_unetpp_ds_spec_nofilm",
            "unetpp",
            True,
            "on",
            False,
            film_injection="none",
            note="Spectrum features on but FiLM disabled (film worth it?)",
        )
    )
    cells.append(
        ArchCell(
            "J_unet_spec_hr",
            "unet",
            False,
            "on",
            True,
            note="UNet + spectrum + HR (can UNet use HR as well as UNet++?)",
        )
    )
    return cells


def get_grid(name: GridName) -> list[ArchCell]:
    if name == "core":
        return core_grid()
    if name == "extended":
        return extended_grid()
    raise ValueError(f"Unknown grid {name!r}")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = deepcopy(val)
    return out


def apply_cell_to_config(base_cfg: dict[str, Any], cell: ArchCell) -> dict[str, Any]:
    """Return a full user config dict for ``cell`` (training/data/model)."""
    spectrum_on = cell.spectrum == "on"
    film = cell.resolved_film()

    model_overlay: dict[str, Any] = {
        "architecture": cell.architecture,
        "deep_supervision": bool(cell.deep_supervision),
        "film_injection": film,
        "use_hr_cross_attention": bool(cell.hr_cross_attn),
        "use_hr_cross_attn": bool(cell.hr_cross_attn),
        "imaging_resolution": "aligned",
        "spatial_pipeline": "symmetric",
        "output_head": "single",
        "footprint_mode": "spatial_channel",
    }

    if spectrum_on:
        model_overlay.update(
            {
                "spectrum_pooling": "attention",
                "spectrum_use_wavelength": True,
                "spectrum_use_ivar": True,
            }
        )
    else:
        # Keep encoder settings inert; data.use_spectrum=false is the real switch.
        model_overlay.update(
            {
                "spectrum_pooling": "avg",
                "spectrum_use_wavelength": False,
                "spectrum_use_ivar": False,
            }
        )

    if cell.hr_cross_attn:
        model_overlay.update(
            {
                "hr_survey": "sdss",
                "hr_cross_attention_levels": [0, 1],
                "hr_cross_attn_levels": [0, 1],
                "hr_encoder_n_down": 1,
                "hr_attention_mode": "local",
                "hr_attention_window": 7,
                "hr_attn_dropout": 0.0,
            }
        )

    data_overlay: dict[str, Any] = {
        "use_spectrum": spectrum_on,
        "use_sdss": True,
        "use_legacy": False,
        "imaging_resolution": "aligned",
    }

    training_overlay: dict[str, Any] = {}
    if cell.hr_cross_attn:
        # Level-0 local HR is VRAM-heavy; keep eval ≤ train.
        training_overlay["batching"] = {
            "train_batch_size": 16,
            "eval_batch_size": 16,
        }

    # UNet cannot use deep supervision — enforce.
    if cell.architecture == "unet" and cell.deep_supervision:
        raise ValueError(f"Invalid cell {cell.name}: unet + deep_supervision")

    cfg = deep_merge(base_cfg, {"model": model_overlay, "data": data_overlay})
    if training_overlay:
        cfg = deep_merge(cfg, {"training": training_overlay})
    return cfg


__all__ = [
    "ArchCell",
    "GridName",
    "apply_cell_to_config",
    "core_grid",
    "deep_merge",
    "extended_grid",
    "get_grid",
]
