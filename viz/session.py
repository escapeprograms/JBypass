"""Keeps at most one model loaded: switching models frees the old one before loading the new."""

import dataclasses
import gc
from collections.abc import Callable

import torch

from evaluation.cli_common import load_frozen_jspace
from jspace import JSpace
from model.config import ModelProfile, RunConfig, load_model_profile
from viz.catalog import ModelChoice


class Session:
    """Holds the JSpace of the current ModelChoice; `load` builds one from a choice."""

    def __init__(self, load: Callable[[ModelChoice], JSpace]):
        self._load = load
        self._choice: ModelChoice | None = None
        self._js: JSpace | None = None

    def get(self, choice: ModelChoice) -> JSpace:
        """The JSpace for `choice`, loading it (after releasing the old one) if it is new."""
        # TODO(debt): switching adapter reloads the whole model; PEFT multi-adapter would avoid it
        if self._js is None or choice != self._choice:
            self.release()
            self._js = self._load(choice)
            self._choice = choice
        return self._js

    def release(self) -> None:
        """Drop the loaded model and free its memory."""
        self._js = None
        self._choice = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_loader(run: RunConfig, device: str | None) -> Callable[[ModelChoice], JSpace]:
    """choice -> frozen JSpace of its profile (from run.profiles_dir) and adapter,
    on `device` (None = the profile's device), with the profile's prefit lens."""
    def load(choice: ModelChoice) -> JSpace:
        profile = _on_device(load_model_profile(choice.profile, run.profiles_dir), device)
        return load_frozen_jspace(profile, choice.adapter)
    return load


def _on_device(profile: ModelProfile, device: str | None) -> ModelProfile:
    return dataclasses.replace(profile, device=device or profile.device)
