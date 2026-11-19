"""LoRA fine-tune on the train split under one eval condition, then save the adapter
that every eval script loads with --adapter.

python -m finetune.run_train --condition bridge --split_file results/splits.json --out runs/mexp
python -m finetune.run_train --condition none   --split_file results/splits.json --out runs/mctrl
python -m finetune.run_train --condition bridge_clamp --split_file results/splits.json --out runs/mclamp

bridge_clamp needs readout statistics: run `python -m evaluation.run_readout_stats` first
(writes <readout_stats_dir>/<model>.pt, loaded by default) or pass --stats_path.

Steps, batch size, learning rate, LoRA rank and seed come from configs/run.yaml (--config);
so do J-space overrides (band, clamp_z, readout_stats_dir) in its optional `jspace:` section.
"""

import argparse
from pathlib import Path

import torch

from evaluation.cli_common import STATS_PATH_HELP, load_items, write_json
from evaluation.config import EvalConfig
from evaluation.runner import readable_items
from finetune.ablated import condition_factory, item_text
from finetune.config import TRAIN_CONDITIONS, FinetuneConfig
from jspace import JSpace, default_stats_path, jspace_config, load_jspace
from model.cli import add_profile_args, resolve_profile
from model.loading import load_model
from model.lora import attach_lora, save_adapter
from model.training import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_profile_args(p)
    p.add_argument("--condition", required=True, choices=TRAIN_CONDITIONS)
    p.add_argument("--split_file", required=True, help="from evaluation.make_splits")
    p.add_argument("--out", required=True, help="adapter directory")
    p.add_argument("--lens_path", default=None, help="local lens .pt; default = the profile's prefit lens")
    p.add_argument("--stats_path", default=None, help=STATS_PATH_HELP)
    return p.parse_args()


def used_stats_path(js: JSpace, stats_path: str | None) -> str | None:
    """The readout statistics file js was loaded with, None if it has none."""
    if js.readout_stats is None:
        return None
    return str(stats_path or default_stats_path(js.lm, js.cfg))


def main():
    args = parse_args()
    cfg, eval_cfg = FinetuneConfig(), EvalConfig()
    run, profile = resolve_profile(args)
    lm = load_model(profile)
    js = load_jspace(lm, jspace_config(run), lens_path=args.lens_path, stats_path=args.stats_path)
    attach_lora(lm, run.lora)
    items = readable_items(js, load_items(cfg.dataset, eval_cfg, args.split_file, cfg.split))
    print(f"training {profile.name} under {args.condition!r} on {len(items)} {cfg.split} items, "
          f"{run.train.steps} steps of {run.train.batch_size}")
    factory = condition_factory(js, args.condition, eval_cfg.seed)
    losses = train(lm, items, run.train, item_text, factory,
                   lambda step, loss, m: print(f"  step {step}: loss {loss:.4f}"))
    save_adapter(lm, args.out)
    write_json({"condition": args.condition, "seed": run.train.seed, "ablation_seed": eval_cfg.seed,
                "split_file": args.split_file, "split": cfg.split,
                "stats_path": used_stats_path(js, args.stats_path), "clamp_z": js.cfg.clamp_z,
                "band": js.band, "losses": losses},
               Path(args.out) / cfg.info_file)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
