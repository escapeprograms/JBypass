"""LoRA fine-tune on the train split under one eval condition, then save the adapter
that every eval script loads with --adapter.

python -m finetune.run_train --condition bridge --split_file results/splits.json --out runs/mexp
python -m finetune.run_train --condition none   --split_file results/splits.json --out runs/mctrl

Steps, batch size, learning rate, LoRA rank and seed come from configs/run.yaml (--config).
"""

import argparse
from pathlib import Path

import torch

from evaluation.cli_common import load_items, write_json
from evaluation.config import EvalConfig
from evaluation.runner import readable_items
from finetune.ablated import condition_factory, item_text
from finetune.config import TRAIN_CONDITIONS, FinetuneConfig
from jspace import JSpaceConfig, load_jspace
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
    return p.parse_args()


def main():
    args = parse_args()
    cfg, eval_cfg = FinetuneConfig(), EvalConfig()
    run, profile = resolve_profile(args)
    lm = load_model(profile)
    # TODO(debt-002): J-space settings are the dataclass defaults, not read from a file.
    js = load_jspace(lm, JSpaceConfig(), lens_path=args.lens_path)
    attach_lora(lm, run.lora)
    items = readable_items(js, load_items(cfg.dataset, eval_cfg, args.split_file, cfg.split))
    print(f"training {profile.name} under {args.condition!r} on {len(items)} {cfg.split} items, "
          f"{run.train.steps} steps of {run.train.batch_size}")
    factory = condition_factory(js, args.condition, eval_cfg.seed)
    losses = train(lm, items, run.train, item_text, factory,
                   lambda step, loss, m: print(f"  step {step}: loss {loss:.4f}"))
    save_adapter(lm, args.out)
    write_json({"condition": args.condition, "seed": run.train.seed, "ablation_seed": eval_cfg.seed,
                "split_file": args.split_file, "split": cfg.split, "losses": losses},
               Path(args.out) / cfg.info_file)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
