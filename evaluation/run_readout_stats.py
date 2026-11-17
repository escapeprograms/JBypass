"""Reference statistics of the J-lens readout (per token, per band layer: mean and std of
the lens logit over the dataset's prompts), which the clamp conditions cap the bridge at.
Each prompt leaves its own bridge's tokens out. Stats are keyed by model profile, so an
adapter run uses base M's file.

python -m evaluation.run_readout_stats                                      # -> stats/<model>.pt
python -m evaluation.run_readout_stats --split_file results/splits.json --split train --out stats/train.pt
"""

import argparse
from collections.abc import Sequence

from evaluation.cli_common import add_model_args, load_from_args, load_items
from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from evaluation.runner import readable_items
from jspace import collect_readout_stats, default_stats_path, save_readout_stats, token_variants


def reference_texts(tokenizer, items: Sequence[TwoHopItem]) -> tuple[list[str], list[list[int]]]:
    """(texts, exclude_ids): each item's prompt and its bridge's single-token variants."""
    return [it.prompt for it in items], [token_variants(tokenizer, it.bridge) for it in items]


def parse_args() -> argparse.Namespace:
    cfg = EvalConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="jbypass", choices=sorted(cfg.datasets))
    p.add_argument("--split_file", default=None, help="restrict to one split; default = whole dataset")
    p.add_argument("--split", default=None)
    p.add_argument("--out", default=None, help="output .pt; default = stats/<model>.pt")
    add_model_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    # TODO(debt): this also loads an existing default stats file, so a stale one that no longer
    # fits the band raises here; delete it (or pass --stats_path to a valid file) to recompute.
    js = load_from_args(args)
    items = readable_items(js, load_items(args.dataset, EvalConfig(), args.split_file, args.split, args.limit))
    texts, exclude = reference_texts(js.lm.tokenizer, items)
    stats = collect_readout_stats(js, texts, exclude, js.cfg.stats_batch_size)
    out = args.out or default_stats_path(js.lm, js.cfg)
    save_readout_stats(stats, out)
    print(f"{stats.n_texts} texts from {args.dataset} ({args.split or 'all'}), "
          f"{int(stats.count.max())} editable positions, band layers {stats.layers[0]}-{stats.layers[-1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
