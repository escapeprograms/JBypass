"""What the viewer can offer: models (base profiles and saved adapters), splits, examples,
prompt fields and conditions. Invalid combinations are filtered out here, never caught later."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from evaluation.cli_common import load_items
from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from evaluation.runner import readable_items
from model.config import load_model_profile
from model.lora import PROFILE_FILE, saved_profile

ALL_SPLITS = "all"   # split choice meaning "no split filter"


@dataclass(frozen=True)
class ModelChoice:
    profile: str                    # model profile name
    adapter: Path | None = None     # saved LoRA adapter folder; None = base model

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


def split_names(items: list[TwoHopItem], split_file: str | Path | None) -> list[str]:
    """ALL_SPLITS plus every split in `split_file` that names at least one of the items."""
    if split_file is None:
        return [ALL_SPLITS]
    splits = json.loads(Path(split_file).read_text(encoding="utf-8"))
    names = {it.name for it in items}
    return [ALL_SPLITS] + [s for s, members in splits.items() if names & set(members)]


def examples(js, dataset: str, split: str, eval_cfg: EvalConfig,
             split_file: str | Path | None) -> list[TwoHopItem]:
    """The dataset's items in `split` (every item for ALL_SPLITS) whose bridge the lens can read."""
    split_file, split = (None, None) if split == ALL_SPLITS else (split_file, split)
    return readable_items(js, load_items(dataset, eval_cfg, split_file, split))


def valid_fields(item: TwoHopItem, fields) -> list[str]:
    """The prompt fields the item has (drops fields that are None on it)."""
    return [f for f in fields if getattr(item, f) is not None]


def valid_conditions(item: TwoHopItem, conditions) -> list[str]:
    """The conditions that can be built for the item."""
    # TODO(debt): this rule lives away from its builder in evaluation/conditions.py; move it there when eval needs it
    return [c for c in conditions if not (c == "wrong_bridge" and item.swap_to is None)]


def pick_random(items: list[TwoHopItem], rng: random.Random) -> TwoHopItem:
    return rng.choice(items)
