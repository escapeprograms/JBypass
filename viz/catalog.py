"""What the viewer can offer: models (base profiles and saved adapters), splits, examples,
prompt fields and conditions. Invalid combinations are filtered out here, never caught later."""

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

from evaluation.cli_common import load_items
from evaluation.conditions import NEEDS_STATS
from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from evaluation.runner import tokenizer_readable_items
from model.config import load_model_profile
from model.lora import PROFILE_FILE, saved_profile

ALL_SPLITS = "all"   # split choice meaning "no split filter"
NEEDS_SWAP = frozenset({"wrong_bridge", "wrong_bridge_clamp"})   # conditions built from item.swap_to


@dataclass(frozen=True)
class ModelChoice:
    profile: str                    # model profile name
    adapter: Path | None = None     # saved LoRA adapter folder; None = base model
    lens: Path | None = None        # the model's refit lens file (readout only); None = none found

    @property
    def label(self) -> str:
        return self.profile if self.adapter is None else f"{self.profile} + {self.adapter.name}"


def lens_profiles(profiles_dir: str | Path) -> list[str]:
    """Names of the profiles in `profiles_dir` that have a prefit lens, sorted."""
    names = sorted(p.stem for p in Path(profiles_dir).glob("*.yaml"))
    return [n for n in names if load_model_profile(n, profiles_dir).prefit_lens is not None]


def model_choices(profiles_dir: str | Path, runs_dir: str | Path) -> list[ModelChoice]:
    """Each lens profile, then each adapter in `runs_dir/*/` saved from a lens profile."""
    profiles = lens_profiles(profiles_dir)
    return [ModelChoice(p) for p in profiles] + _adapter_choices(Path(runs_dir), set(profiles))


def _adapter_choices(runs_dir: Path, profiles: set[str]) -> list[ModelChoice]:
    if not runs_dir.is_dir():
        return []
    dirs = sorted(d for d in runs_dir.iterdir() if (d / PROFILE_FILE).is_file())
    pairs = [(saved_profile(d), d) for d in dirs]
    return [ModelChoice(profile, d) for profile, d in pairs if profile in profiles]


# TODO(debt-003): a lens is paired with its model by file name only (profiles and adapter folders share one namespace)
def refit_lens_file(choice: ModelChoice, lenses_dir: Path) -> Path:
    """Where the choice's refit lens lives: <lenses_dir>/<adapter folder name>.pt, or
    <lenses_dir>/<profile>.pt for a base model."""
    stem = choice.profile if choice.adapter is None else choice.adapter.name
    return Path(lenses_dir) / f"{stem}.pt"


def with_refit_lenses(choices: list[ModelChoice], lenses_dir: Path) -> list[ModelChoice]:
    """The choices, each with `.lens` set to its refit lens file where that file exists."""
    found = [(c, refit_lens_file(c, lenses_dir)) for c in choices]
    return [dataclasses.replace(c, lens=f) if f.is_file() else c for c, f in found]


def fit_command(choice: ModelChoice, lens_file: Path) -> str:
    """The command line that fits the choice's refit lens into `lens_file`."""
    adapter = "" if choice.adapter is None else f" --adapter {choice.adapter}"
    return f"python -m jspace.fit_lens --model {choice.profile}{adapter} --out {lens_file}"


def split_names(items: list[TwoHopItem], split_file: str | Path | None) -> list[str]:
    """ALL_SPLITS plus every split in `split_file` that names at least one of the items."""
    if split_file is None:
        return [ALL_SPLITS]
    splits = json.loads(Path(split_file).read_text(encoding="utf-8"))
    names = {it.name for it in items}
    return [ALL_SPLITS] + [s for s, members in splits.items() if names & set(members)]


def examples(tokenizer, dataset: str, split: str, eval_cfg: EvalConfig,
             split_file: str | Path | None) -> list[TwoHopItem]:
    """The dataset's items in `split` (every item for ALL_SPLITS) whose bridge has a single-token
    spelling under `tokenizer` (so the lens can read it); needs no model."""
    split_file, split = (None, None) if split == ALL_SPLITS else (split_file, split)
    return tokenizer_readable_items(tokenizer, load_items(dataset, eval_cfg, split_file, split))


def valid_fields(item: TwoHopItem, fields) -> list[str]:
    """The prompt fields the item has (drops fields that are None on it)."""
    return [f for f in fields if getattr(item, f) is not None]


def valid_conditions(item: TwoHopItem, conditions) -> list[str]:
    """The conditions that can be built for the item."""
    # TODO(debt): this rule lives away from its builder in evaluation/conditions.py; move it there when eval needs it
    return [c for c in conditions if not (c in NEEDS_SWAP and item.swap_to is None)]


def buildable_conditions(has_stats: bool, item: TwoHopItem, conditions) -> list[str]:
    """valid_conditions, minus the ones that need readout statistics when there are none."""
    valid = valid_conditions(item, conditions)
    return valid if has_stats else [c for c in valid if c not in NEEDS_STATS]


def pick_random(items: list[TwoHopItem], rng: random.Random) -> TwoHopItem:
    return rng.choice(items)
