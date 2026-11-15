"""Shared command-line plumbing for the evaluation scripts: model/lens flags,
dataset loading, JSON output."""

import argparse
import json
from pathlib import Path

from evaluation.config import EvalConfig
from evaluation.data_loading import load_dataset
from jspace import JSpace, JSpaceConfig, load_jspace
from model.cli import add_profile_args, resolve_profile
from model.loading import LanguageModel, freeze, load_model
from model.lora import load_adapter


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """--config, --model, --device (model.cli), plus --adapter, --lens_path and --limit."""
    add_profile_args(parser)
    parser.add_argument("--adapter", default=None, help="LoRA dir (Mctrl / Mexp); default = base model M")
    parser.add_argument("--lens_path", default=None, help="local lens .pt (e.g. recomputed); default = the profile's prefit lens")
    parser.add_argument("--limit", type=int, default=None, help="first N items only (quick runs)")


def load_model_from_args(args) -> LanguageModel:
    """Load the profile's model and the adapter if given (unmerged); every weight frozen."""
    _, profile = resolve_profile(args)
    lm = load_model(profile)
    if args.adapter is not None:
        load_adapter(lm, args.adapter)
    freeze(lm)
    return lm


def load_from_args(args) -> JSpace:
    """load_model_from_args plus the lens (--lens_path, or the profile's prefit lens)."""
    # TODO(debt-002): J-space settings are the dataclass defaults, not read from a file.
    return load_jspace(load_model_from_args(args), JSpaceConfig(), lens_path=args.lens_path)


def load_items(name: str, cfg: EvalConfig, split_file: str | None = None, split: str | None = None,
               limit: int | None = None):
    items = load_dataset(name, cfg.datasets[name])
    if split_file is not None:
        keep = set(json.loads(Path(split_file).read_text())[split])
        items = [it for it in items if it.name in keep]
    return items[:limit]


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    print(f"wrote {path}")
