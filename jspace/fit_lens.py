"""Fit a new J-lens with Anthropic's jlens.fit, e.g. on a fine-tuned model
(proposal: recompute the lens after training and compare with the original).

python -m jspace.fit_lens --adapter runs/some_adapter --out lenses/recomputed.pt
python -m jspace.fit_lens --n_prompts 100 --out lenses/quick.pt     # quick lens
python -m jspace.fit_lens --model qwen3-0.6b --out lenses/qwen3-0.6b.pt

Prompts are WikiText-103 records via jlens.examples.load_wikitext_prompts,
the same corpus the prefit lens used.
"""

import argparse
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

from jspace.config import JSpaceConfig
from jspace.loading import lens_view
from model.cli import add_profile_args, resolve_profile
from model.loading import freeze, load_model
from model.lora import load_adapter


def parse_args(cfg: JSpaceConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_profile_args(p)
    p.add_argument("--adapter", default=None, help="LoRA dir saved by model.lora.save_adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--n_prompts", type=int, default=cfg.fit_n_prompts)
    p.add_argument("--dim_batch", type=int, default=cfg.fit_dim_batch)
    return p.parse_args()


def main():
    cfg = JSpaceConfig()
    args = parse_args(cfg)
    jlens.configure_logging()
    _, profile = resolve_profile(args)
    lm = load_model(profile)
    if args.adapter is not None:
        load_adapter(lm, args.adapter)
    # Fitting needs gradients only with respect to activations; frozen weights keep the
    # retained graph to the blocks after the first source layer.
    freeze(lm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # TODO(debt-001): jlens.fit tokenises each prompt itself (jlens's encode, not model.inference).
    lens = jlens.fit(lens_view(lm), load_wikitext_prompts(args.n_prompts), dim_batch=args.dim_batch,
                     max_seq_len=cfg.fit_max_seq_len, checkpoint_path=f"{out}.ckpt")
    lens.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
