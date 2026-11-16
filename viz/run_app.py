"""Entry point: python -m viz.run_app [--config run.yaml] [--model NAME] [--device cuda] [--viz_config viz.yaml]"""

import argparse
import random
from pathlib import Path

from evaluation.config import EvalConfig
from model.cli import add_profile_args, resolve_profile
from viz.app import AppContext, build_app
from viz.catalog import ModelChoice, model_choices
from viz.config import DEFAULT_VIZ_CONFIG, load_viz_config
from viz.session import Session, make_loader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive J-space viewer (Gradio).")
    add_profile_args(parser)
    parser.add_argument("--viz_config", type=Path, default=DEFAULT_VIZ_CONFIG, help="viz.yaml path")
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace) -> AppContext:
    """Configs, the empty model session and the model choices; the default is --model's base model."""
    run, profile = resolve_profile(args)
    viz = load_viz_config(args.viz_config)
    choices = {c.label: c for c in model_choices(run.profiles_dir, viz.runs_dir)}
    default = ModelChoice(profile.name).label
    if default not in choices:
        raise ValueError(f"profile {profile.name!r} has no prefit lens; choose one of {list(choices)}")
    eval_cfg = EvalConfig()  # TODO(debt-002): eval settings are the dataclass defaults, not read from a file.
    return AppContext(run=run, viz=viz, eval_cfg=eval_cfg, session=Session(make_loader(run, args.device)),
                      choices=choices, default_model=default, rng=random.Random(eval_cfg.seed))


def main() -> None:
    ctx = build_context(parse_args())
    build_app(ctx).launch(server_name=ctx.viz.server_name, server_port=ctx.viz.server_port,
                          show_error=True)


if __name__ == "__main__":
    main()
