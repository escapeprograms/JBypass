"""Fit a new J-lens with Anthropic's jlens.fit, e.g. on a fine-tuned model
(proposal: recompute the lens after training and compare with the original).

python -m jspace.fit_lens --adapter runs/some_adapter --out lenses/recomputed.pt
python -m jspace.fit_lens --n_prompts 100 --out lenses/quick.pt     # quick lens
python -m jspace.fit_lens --model qwen3-0.6b --out lenses/qwen3-0.6b.pt

Prompts are WikiText-103 records via jlens.examples.load_wikitext_prompts,
the same corpus the prefit lens used. The fit resumes from <out>.ckpt: rerunning the
same --out with a larger --n_prompts extends it, so a different model or adapter
must use a different --out.
"""

import argparse
import dataclasses
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

from jspace.config import JSpaceConfig, jspace_config
from jspace.loading import lens_view
from model.cli import add_profile_args, resolve_profile
from model.loading import LanguageModel, freeze, load_model
from model.lora import load_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_profile_args(p)
    p.add_argument("--adapter", default=None, help="LoRA dir saved by model.lora.save_adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--n_prompts", type=int, default=None,
                   help="default = jspace fit_n_prompts (JSpaceConfig, or --config's jspace: section)")
    p.add_argument("--dim_batch", type=int, default=None,
                   help="default = jspace fit_dim_batch (JSpaceConfig, or --config's jspace: section)")
    return p.parse_args()


def with_flags(cfg: JSpaceConfig, args: argparse.Namespace) -> JSpaceConfig:
    """cfg with --n_prompts / --dim_batch applied where given."""
    flags = {"fit_n_prompts": args.n_prompts, "fit_dim_batch": args.dim_batch}
    return dataclasses.replace(cfg, **{k: v for k, v in flags.items() if v is not None})


# TODO(debt-003): the lens file and its .ckpt record no model or adapter; resume checks neither
def fit_to_file(lm: LanguageModel, prompts: list[str], cfg: JSpaceConfig, out: Path) -> jlens.JacobianLens:
    """Fit a lens on lm as loaded (adapter included), checkpointing to <out>.ckpt every
    cfg.fit_checkpoint_every prompts, save it to `out` and return it. An existing
    <out>.ckpt is resumed, so rerunning with a longer prompt list extends the fit."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # TODO(debt-001): jlens.fit tokenises each prompt itself (jlens's encode, not model.inference).
    lens = jlens.fit(lens_view(lm), prompts, dim_batch=cfg.fit_dim_batch, max_seq_len=cfg.fit_max_seq_len,
                     checkpoint_path=f"{out}.ckpt", checkpoint_every=cfg.fit_checkpoint_every)
    lens.save(str(out))
    return lens


def main():
    args = parse_args()
    run, profile = resolve_profile(args)
    cfg = with_flags(jspace_config(run), args)
    jlens.configure_logging()
    lm = load_model(profile)
    if args.adapter is not None:
        load_adapter(lm, args.adapter)
    # Fitting needs gradients only with respect to activations; frozen weights keep the
    # retained graph to the blocks after the first source layer.
    freeze(lm)
    out = Path(args.out)
    fit_to_file(lm, load_wikitext_prompts(cfg.fit_n_prompts), cfg, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
