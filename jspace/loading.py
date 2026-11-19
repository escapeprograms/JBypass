"""Attach a J-lens to an already-loaded model. All network and file access for the
jspace module lives here; the model itself is loaded by `model.loading.load_model`."""

from dataclasses import dataclass
from pathlib import Path

import jlens
from huggingface_hub import hf_hub_download

from jspace.baseline import ReadoutStats, load_readout_stats
from jspace.config import JSpaceConfig
from jspace.vectors import layer_range, workspace_band
from model.loading import LanguageModel


@dataclass
class JSpace:
    """Everything the interventions and readouts need.

    lm: the shared model (forward passes, hooks, tokenizer, unembedding).
    lens_model: jlens view of lm.base_model (used by lens.apply / jlens.fit); it shares
        lm's block modules, so hooks registered on lm.blocks fire inside jlens.
    lens: fitted jlens.JacobianLens, J_l per layer.
    band: workspace layer indices.
    readout_stats: reference readout statistics (needed by ReadoutClamp), or None.
    """

    lm: LanguageModel
    lens_model: jlens.HFLensModel
    lens: jlens.JacobianLens
    band: list[int]
    cfg: JSpaceConfig
    readout_stats: ReadoutStats | None = None


def lens_view(lm: LanguageModel) -> jlens.HFLensModel:
    """jlens.from_hf on lm.base_model with its side effects on the model undone.

    from_hf freezes every parameter and calls .eval(); both are restored to what they
    were, so this is safe before or after attach_lora and during training. Its
    force_bos may set tokenizer.add_bos_token, which only affects tokenising with
    special tokens (jlens's own encode); model.inference never adds special tokens.
    For Qwen (bos_token_id is None) it does nothing."""
    # TODO(debt-001): jlens keeps its own tokenisation (force_bos) inside lens.apply / jlens.fit.
    grads = [(p, p.requires_grad) for p in lm.base_model.parameters()]
    modes = [(m, m.training) for m in lm.base_model.modules()]
    view = jlens.from_hf(lm.base_model, lm.tokenizer)
    for param, requires_grad in grads:
        param.requires_grad_(requires_grad)
    for module, training in modes:
        module.training = training
    return view


def load_lens(lm: LanguageModel, lens_path: str | None = None) -> jlens.JacobianLens:
    """The lens at `lens_path`, else the profile's prefit lens from the Hub, with its J
    matrices on lm's device. Raises if neither exists or if the lens does not fit lm."""
    lens = jlens.JacobianLens.load(_lens_file(lm, lens_path))
    _check_lens_fits(lens, lm)
    lens.jacobians = {l: J.to(lm.device) for l, J in lens.jacobians.items()}
    return lens


def load_jspace(
    lm: LanguageModel, cfg: JSpaceConfig = JSpaceConfig(), lens_path: str | None = None,
    stats_path: str | Path | None = None,
) -> JSpace:
    """Bundle a loaded model with its lens (`lens_path`, or the profile's prefit lens) and
    its readout statistics (`stats_path`, else the default file if it exists, else None).
    Raises if the lens has no J matrix for a band layer."""
    lens = load_lens(lm, lens_path)
    band = _band(lm, cfg)
    _check_lens_covers(lens, band)
    return JSpace(lm, lens_view(lm), lens, band, cfg, _load_stats(lm, cfg, band, stats_path))


def _band(lm: LanguageModel, cfg: JSpaceConfig) -> list[int]:
    """cfg.band_layers (inclusive) if set, else the band_start/band_end fractions of lm's depth."""
    if cfg.band_layers is not None:
        return layer_range(*cfg.band_layers, lm.n_layers)
    return workspace_band(lm.n_layers, cfg.band_start, cfg.band_end)


def _check_lens_covers(lens: jlens.JacobianLens, band: list[int]) -> None:
    missing = [l for l in band if l not in lens.jacobians]
    if missing:
        raise ValueError(f"the lens has no J matrix for band layers {missing} "
                         f"(lens layers: {sorted(lens.source_layers)})")


def default_stats_path(lm: LanguageModel, cfg: JSpaceConfig) -> Path:
    """Where the readout statistics of lm's profile live by default."""
    return profile_stats_path(lm.profile.name, cfg)


def profile_stats_path(profile_name: str, cfg: JSpaceConfig) -> Path:
    """Where the readout statistics of the named profile live by default (no model needed)."""
    return Path(cfg.readout_stats_dir) / f"{profile_name}.pt"


def _load_stats(
    lm: LanguageModel, cfg: JSpaceConfig, band: list[int], stats_path: str | Path | None
) -> ReadoutStats | None:
    """Stats at an explicit path (a missing file raises), else the default file if present."""
    if stats_path is None:
        stats_path = default_stats_path(lm, cfg)
        if not stats_path.exists():
            return None
    return load_readout_stats(stats_path, lm, band)


def _lens_file(lm: LanguageModel, lens_path: str | None) -> str:
    """Local path of the lens file, downloading the profile's prefit lens if needed."""
    if lens_path is not None:
        return lens_path
    source = lm.profile.prefit_lens
    if source is None:
        raise ValueError(f"profile {lm.profile.name!r} has no prefit_lens; pass --lens_path")
    return hf_hub_download(source.repo, source.file, revision=source.revision)


def _check_lens_fits(lens: jlens.JacobianLens, lm: LanguageModel) -> None:
    """A lens file does not record its model, so check that its shape fits lm."""
    if lens.d_model != lm.d_model:
        raise ValueError(f"lens d_model {lens.d_model} does not match the model's {lm.d_model}")
    outside = [l for l in lens.source_layers if not 0 <= l < lm.n_layers]
    if outside:
        raise ValueError(f"lens layers {outside} do not exist in a {lm.n_layers}-layer model")
    shapes = {tuple(J.shape) for J in lens.jacobians.values()}
    if shapes != {(lm.d_model, lm.d_model)}:
        raise ValueError(f"lens matrices have shapes {shapes}, expected {(lm.d_model, lm.d_model)}")
