"""Keep the jbypass items the clean model M fully solves (two-hop and both
one-hops), then split them by bridge.

python -m evaluation.make_splits --out results/splits.json
"""

import argparse
from pathlib import Path

from evaluation.cli_common import add_model_args, load_from_args, load_items, write_json
from evaluation.config import EvalConfig
from evaluation.runner import readable_items, solved_items
from evaluation.splits import split_by_bridge


def main():
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--unseen_frac", type=float, default=0.3)
    p.add_argument("--out", default=str(cfg.out_dir / "splits.json"))
    add_model_args(p)
    args = p.parse_args()

    js = load_from_args(args)
    items = readable_items(js, load_items("jbypass", cfg, limit=args.limit))
    solved = solved_items(js, items, cfg.prompt_fields, cfg.max_answer_tokens)
    print(f"{len(items)} readable items, {len(solved)} fully solved by the clean model")
    splits = split_by_bridge(solved, args.unseen_frac, cfg.seed)
    for name, names in splits.items():
        print(f"  {name:12s} {len(names)} items")
    write_json(splits, Path(args.out))


if __name__ == "__main__":
    main()
