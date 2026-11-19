"""Keeps at most one model loaded: switching models frees the old one before loading the new."""

import dataclasses
import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from evaluation.cli_common import load_frozen_jspace
from jspace import JSpace, jspace_config, load_lens
from model.config import ModelProfile, RunConfig, load_model_profile
from viz.catalog import ModelChoice


@dataclass(frozen=True)
class Loaded:
    """A loaded model read through its lenses. `js` (the original, prefit lens) drives generation,
    interventions and the replay check; `own` (the model's refit lens, or None) is readout-only."""
    js: JSpace
    own: JSpace | None = None

    @property
    def readers(self) -> list[JSpace]:
        """The JSpaces to read out with, original lens first."""
        return [self.js] if self.own is None else [self.js, self.own]


# TODO(debt-003): the refit lens is trusted by file name; nothing checks it was fitted on this model
def attach_refit(js: JSpace, lens_file: Path | None) -> Loaded:
    """`js` plus a copy of it reading through the lens at `lens_file` (same model, band and
    statistics); no second reader when `lens_file` is None."""
    if lens_file is None:
        return Loaded(js)
    return Loaded(js, dataclasses.replace(js, lens=load_lens(js.lm, str(lens_file))))


class Session:
    """Holds the Loaded model of the current ModelChoice; `load` builds one from a choice."""

    def __init__(self, load: Callable[[ModelChoice], Loaded]):
        self._load = load
        self._choice: ModelChoice | None = None
        self._loaded: Loaded | None = None

    def get(self, choice: ModelChoice) -> Loaded:
        """The Loaded model for `choice`, loading it (after releasing the old one) if it is new."""
        # TODO(debt): switching adapter reloads the whole model; PEFT multi-adapter would avoid it
        if self._loaded is None or choice != self._choice:
            self.release()
            self._loaded = self._load(choice)
            self._choice = choice
        return self._loaded

    def release(self) -> None:
        """Drop the loaded model and free its memory."""
        self._loaded = None
        self._choice = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_loader(run: RunConfig, device: str | None) -> Callable[[ModelChoice], Loaded]:
    """choice -> Loaded: the frozen JSpace of its profile (from run.profiles_dir) and adapter,
    on `device` (None = the profile's device), with the profile's prefit lens and the
    J-space settings (band, readout statistics folder) of run.yaml's `jspace:` section,
    plus its refit lens (`choice.lens`) as a second reader when it has one."""
    jspace_cfg = jspace_config(run)

    def load(choice: ModelChoice) -> Loaded:
        profile = _on_device(load_model_profile(choice.profile, run.profiles_dir), device)
        return attach_refit(load_frozen_jspace(profile, jspace_cfg, choice.adapter), choice.lens)
    return load


def _on_device(profile: ModelProfile, device: str | None) -> ModelProfile:
    return dataclasses.replace(profile, device=device or profile.device)
