"""Settings dataclasses and the YAML loaders that build them.

Model-specific facts live in one YAML file per model (`configs/models/<name>.yaml`).
Experiment choices (which model, generation, LoRA and training values, and optional
J-space overrides) live in `configs/run.yaml`.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LensSource:
    """Hub location of a prefit J-lens file."""

    repo: str                       # Hub repo id, e.g. "neuronpedia/jacobian-lens"
    file: str                       # path of the .pt inside the repo
    revision: str                   # branch, tag or commit


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
    prefit_lens: LensSource | None  # Hub location of the model's prefit lens; None if there is none


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
    jspace: dict                    # raw optional `jspace:` section ({} if absent); read by jspace.jspace_config
    config_dir: Path                # resolved folder of the YAML file (base for relative paths in it)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lens_source(data: dict | None) -> LensSource | None:
    """The profile's `prefit_lens` mapping as a LensSource; YAML `null` stays None."""
    return None if data is None else LensSource(**data)


# TODO(debt-002): EvalConfig and FinetuneConfig are dataclass defaults, not sections of run.yaml.
def load_run_config(path: str | Path) -> RunConfig:
    """Read run.yaml. `profiles_dir` in the file is relative to the file's folder. The optional
    `jspace:` section is kept raw; `jspace.jspace_config` interprets it."""
    path = Path(path)
    data = _read_yaml(path)
    config_dir = path.parent.resolve()
    return RunConfig(
        model=data["model"],
        profiles_dir=(config_dir / data["profiles_dir"]).resolve(),
        generation=GenerationSettings(**data["generation"]),
        lora=LoraSettings(**data["lora"]),
        train=TrainSettings(**data["train"]),
        jspace=data.get("jspace") or {},
        config_dir=config_dir,
    )


def load_model_profile(name: str, profiles_dir: str | Path) -> ModelProfile:
    """Read `<profiles_dir>/<name>.yaml`; raises if the file's `name` differs."""
    data = _read_yaml(Path(profiles_dir) / f"{name}.yaml")
    profile = ModelProfile(**{**data, "prefit_lens": _lens_source(data["prefit_lens"])})
    if profile.name != name:
        raise ValueError(f"profile file {name}.yaml declares name {profile.name!r}")
    return profile
