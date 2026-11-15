"""Fit a new J-lens with Anthropic's jlens.fit, e.g. on a fine-tuned model
(proposal: recompute the lens after training and compare with the original).

python -m jspace.fit_lens --adapter runs/some_adapter --out lenses/recomputed.pt
python -m jspace.fit_lens --n_prompts 100 --out lenses/quick.pt     # quick lens

Prompts are WikiText-103 records via jlens.examples.load_wikitext_prompts,
the same corpus the prefit lens used.
"""

import argparse
import dataclasses
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

from jspace.config import JSpaceConfig
from jspace.loading import load_jspace


def main():
    cfg = JSpaceConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n_prompts", type=int, default=cfg.fit_n_prompts)
    p.add_argument("--dim_batch", type=int, default=cfg.fit_dim_batch)
    p.add_argument("--device", default=cfg.device)
    args = p.parse_args()

    jlens.configure_logging()
    js = load_jspace(dataclasses.replace(cfg, device=args.device), adapter_path=args.adapter)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(js.model, load_wikitext_prompts(args.n_prompts), dim_batch=args.dim_batch,
                     max_seq_len=cfg.fit_max_seq_len, checkpoint_path=f"{out}.ckpt")
    lens.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
