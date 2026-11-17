"""Shared command-line plumbing for the evaluation scripts: model/lens flags,
dataset loading, JSON output."""

import argparse
import json
from pathlib import Path

from evaluation.config import EvalConfig
from evaluation.data_loading import load_dataset
from jspace import JSpace, JSpaceConfig, load_jspace
from model.cli import add_profile_args, resolve_profile
from model.config import ModelProfile
from model.loading import LanguageModel, freeze, load_model
from model.lora import load_adapter

STATS_PATH_HELP = ("reference readout statistics .pt from evaluation.run_readout_stats; "
                   "default = stats/<model>.pt if present")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """--config, --model, --device (model.cli), plus --adapter, --lens_path, --stats_path and --limit."""
    add_profile_args(parser)
    parser.add_argument("--adapter", default=None, help="LoRA dir (Mctrl / Mexp); default = base model M")
    parser.add_argument("--lens_path", default=None, help="local lens .pt (e.g. recomputed); default = the profile's prefit lens")
    parser.add_argument("--stats_path", default=None, help=STATS_PATH_HELP)
    parser.add_argument("--limit", type=int, default=None, help="first N items only (quick runs)")


def add_output_args(parser: argparse.ArgumentParser, cfg: EvalConfig) -> None:
    """--out_dir: folder for this run's result JSON (default cfg.out_dir), e.g. "results/exp 3"."""
    parser.add_argument("--out_dir", type=Path, default=cfg.out_dir,
                        help=f"folder to write results into (one per run); default {cfg.out_dir}")


def load_frozen_model(profile: ModelProfile, adapter: str | Path | None = None) -> LanguageModel:
    """Load the profile's model and the adapter if given (unmerged); every weight frozen."""
    lm = load_model(profile)
    if adapter is not None:
        load_adapter(lm, adapter)
    freeze(lm)
    return lm


def load_frozen_jspace(profile: ModelProfile, adapter: str | Path | None = None,
                       lens_path: str | None = None, stats_path: str | Path | None = None) -> JSpace:
    """load_frozen_model plus the lens (`lens_path`, or the profile's prefit lens) and the
    readout statistics (`stats_path`, else the profile's default file if present, else None)."""
    # TODO(debt-002): J-space settings are the dataclass defaults, not read from a file.
    lm = load_frozen_model(profile, adapter)
    return load_jspace(lm, JSpaceConfig(), lens_path=lens_path, stats_path=stats_path)


def load_model_from_args(args) -> LanguageModel:
    """load_frozen_model with the profile from --config/--model/--device and --adapter."""
    _, profile = resolve_profile(args)
    return load_frozen_model(profile, args.adapter)


def load_from_args(args) -> JSpace:
    """load_frozen_jspace with the profile, --adapter, --lens_path and --stats_path from args."""
    _, profile = resolve_profile(args)
    return load_frozen_jspace(profile, args.adapter, args.lens_path, args.stats_path)


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
