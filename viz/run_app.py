"""Entry point: python -m viz.run_app [--config run.yaml] [--model NAME] [--device cuda] [--viz_config viz.yaml]"""

import argparse
import random
from pathlib import Path

from evaluation.config import EvalConfig
from jspace import jspace_config, profile_stats_path
from model.cli import add_profile_args, resolve_profile
from model.config import RunConfig
from model.loading import load_tokenizer
from viz.app import AppContext, build_app
from viz.cache import ResultCache
from viz.catalog import ModelChoice, lens_profiles, model_choices, with_refit_lenses
from viz.cedict import Cedict, load_cedict
from viz.config import DEFAULT_VIZ_CONFIG, VizConfig, load_viz_config
from viz.session import Session, make_loader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive J-space viewer (Gradio).")
    add_profile_args(parser)
    parser.add_argument("--viz_config", type=Path, default=DEFAULT_VIZ_CONFIG, help="viz.yaml path")
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace) -> AppContext:
    """Configs, the empty model session, the session profile's tokenizer (no model is loaded),
    whether its readout statistics exist, its model choices (the default is its base model),
    the dictionary for the translation box and the empty result cache."""
    run, profile = resolve_profile(args)
    viz = load_viz_config(args.viz_config)
    # TODO(debt): one profile per viewer process; the question list uses its tokenizer
    choices = profile_choices(run, viz, profile.name)
    # TODO(debt): has_stats is file existence; a stats file that does not fit the band fails only at the first Run
    has_stats = profile_stats_path(profile.name, jspace_config(run)).exists()
    eval_cfg = EvalConfig()  # TODO(debt-002): eval settings are the dataclass defaults, not read from a file.
    return AppContext(run=run, viz=viz, eval_cfg=eval_cfg, session=Session(make_loader(run, args.device)),
                      tokenizer=load_tokenizer(profile), has_stats=has_stats, choices=choices,
                      default_model=ModelChoice(profile.name).label, rng=random.Random(eval_cfg.seed),
                      dictionary=load_dictionary(viz.dictionary_file), cache=ResultCache())


def profile_choices(run: RunConfig, viz: VizConfig, profile_name: str) -> dict[str, ModelChoice]:
    """label -> choice for the named profile and its adapters in viz.runs_dir, each with its refit
    lens from viz.lenses_dir where one exists; raises if the profile has no prefit lens."""
    choices = [c for c in model_choices(run.profiles_dir, viz.runs_dir) if c.profile == profile_name]
    if ModelChoice(profile_name) not in choices:
        raise ValueError(f"profile {profile_name!r} has no prefit lens; "
                         f"choose one of {lens_profiles(run.profiles_dir)}")
    return {c.label: c for c in with_refit_lenses(choices, viz.lenses_dir)}


def load_dictionary(path: Path | None) -> Cedict | None:
    """CC-CEDICT from `path` (a missing file raises), or None when translations are off; prints which."""
    if path is None:
        print("Translations off (dictionary_file: null)")
        return None
    cedict = load_cedict(path)
    print(f"CC-CEDICT: {cedict.n_entries:,} entries, {len(cedict.index):,} headwords <- {path}")
    return cedict


def main() -> None:
    ctx = build_context(parse_args())
    build_app(ctx).launch(server_name=ctx.viz.server_name, server_port=ctx.viz.server_port,
                          show_error=True)


if __name__ == "__main__":
    main()
