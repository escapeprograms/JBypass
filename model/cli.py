"""Command-line plumbing shared by every command: --config / --model / --device and profile lookup."""

import argparse
import dataclasses
from pathlib import Path

from model.config import ModelProfile, RunConfig, load_model_profile, load_run_config

DEFAULT_RUN_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "run.yaml"


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    """Add --config (run.yaml path), --model (profile name) and --device (overrides the profile's)."""
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG, help="run.yaml path")
    parser.add_argument("--model", default=None, help="profile name; overrides `model` in run.yaml")
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu, mps; default = the profile's device")


def resolve_profile(args: argparse.Namespace) -> tuple[RunConfig, ModelProfile]:
    """Load run.yaml and the selected profile (--model wins over run.yaml, --device over the profile)."""
    config = load_run_config(args.config)
    profile = load_model_profile(args.model or config.model, config.profiles_dir)
    return config, dataclasses.replace(profile, device=args.device or profile.device)
