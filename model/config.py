"""Settings dataclasses and the YAML loaders that build them.

Model-specific facts live in one YAML file per model (`configs/models/<name>.yaml`).
Experiment choices (which model, generation, LoRA and training values) live in
`configs/run.yaml`.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelProfile:
    """Everything specific to one model. No other file names a model attribute."""

    name: str                       # "qwen3-1.7b"; must match the YAML file name
    hf_id: str                      # Hugging Face repo id, e.g. "Qwen/Qwen3-1.7B"
    dtype: str                      # torch dtype name, e.g. "bfloat16"
    device: str                     # single device, e.g. "cuda"
    blocks_path: str                # submodule path of the decoder block list
    final_norm_path: str            # submodule path of the final norm before lm_head
    chat_template_kwargs: dict      # extra kwargs for tokenizer.apply_chat_template
    lora_target_modules: list[str]  # module names LoRA adapts


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class TrainSettings:
    learning_rate: float
    batch_size: int
    steps: int                      # optimiser steps; examples are cycled as needed
    seed: int


@dataclass(frozen=True)
class RunConfig:
    model: str                      # name of the profile to load
    profiles_dir: Path              # folder holding the model profiles
    generation: GenerationSettings
    lora: LoraSettings
    train: TrainSettings


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_run_config(path: str | Path) -> RunConfig:
    """Read run.yaml. `profiles_dir` in the file is relative to the file's folder."""
    path = Path(path)
    data = _read_yaml(path)
    return RunConfig(
        model=data["model"],
        profiles_dir=(path.parent / data["profiles_dir"]).resolve(),
        generation=GenerationSettings(**data["generation"]),
        lora=LoraSettings(**data["lora"]),
        train=TrainSettings(**data["train"]),
    )


def load_model_profile(name: str, profiles_dir: str | Path) -> ModelProfile:
    """Read `<profiles_dir>/<name>.yaml`; raises if the file's `name` differs."""
    profile = ModelProfile(**_read_yaml(Path(profiles_dir) / f"{name}.yaml"))
    if profile.name != name:
        raise ValueError(f"profile file {name}.yaml declares name {profile.name!r}")
    return profile
