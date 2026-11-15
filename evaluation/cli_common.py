"""Shared command-line plumbing for the evaluation scripts: model/lens flags,
dataset loading, JSON output."""

import argparse
import dataclasses
import json
from pathlib import Path

from evaluation.config import EvalConfig
from evaluation.data_loading import load_dataset
from jspace import JSpaceConfig, load_jspace


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", default=None, help="LoRA dir (Mctrl / Mexp); default = base model M")
    parser.add_argument("--lens_path", default=None, help="local lens .pt (e.g. recomputed); default = Neuronpedia prefit")
    parser.add_argument("--device", default=JSpaceConfig.device)
    parser.add_argument("--limit", type=int, default=None, help="first N items only (quick runs)")


def load_from_args(args):
    cfg = dataclasses.replace(JSpaceConfig(), lens_path=args.lens_path, device=args.device)
    return load_jspace(cfg, adapter_path=args.adapter)


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
