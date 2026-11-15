"""Command-line plumbing shared by the scripts: --config / --model and profile lookup."""

import argparse
from pathlib import Path

from jbypass.model.config import ModelProfile, RunConfig, load_model_profile, load_run_config

DEFAULT_RUN_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "run.yaml"


def parse_run_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG, help="run.yaml path")
    parser.add_argument("--model", default=None, help="profile name; overrides `model` in run.yaml")
    return parser.parse_args()


def resolve_run(args: argparse.Namespace) -> tuple[RunConfig, ModelProfile]:
    """Load run.yaml and the selected profile (--model wins over run.yaml)."""
    config = load_run_config(args.config)
    name = args.model or config.model
    return config, load_model_profile(name, config.profiles_dir)
