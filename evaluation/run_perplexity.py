"""Perplexity on held-out WikiText (validation split), for M or M + adapter.

python -m evaluation.run_perplexity --tag M
python -m evaluation.run_perplexity --adapter runs/some_adapter --tag some_adapter
"""

import argparse

from datasets import load_dataset

from evaluation.cli_common import add_model_args, load_model_from_args, write_json
from evaluation.config import EvalConfig
from evaluation.perplexity import perplexity


def heldout_texts(n: int, min_chars: int) -> list[str]:
    stream = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", streaming=True)
    texts = []
    for record in stream:
        if len(record["text"].strip()) >= min_chars:
            texts.append(record["text"])
        if len(texts) == n:
            break
    return texts


def main():
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="M")
    add_model_args(p)
    args = p.parse_args()

    lm = load_model_from_args(args)   # --lens_path is accepted but unused here
    texts = heldout_texts(args.limit or cfg.ppl_n_texts, cfg.ppl_min_chars)
    ppl = perplexity(lm, texts, cfg.ppl_max_len, cfg.ppl_batch_size)
    print(f"[{args.tag}] held-out perplexity = {ppl:.2f}")
    write_json({"tag": args.tag, "adapter": args.adapter, "perplexity": ppl},
               cfg.out_dir / f"perplexity_{args.tag}.json")


if __name__ == "__main__":
    main()
