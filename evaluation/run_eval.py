"""Accuracy table: prompt type x ablation condition.

python -m evaluation.run_eval --dataset jbypass --tag M
python -m evaluation.run_eval --dataset jbypass --split_file results/splits.json --split test_unseen --adapter runs/mctrl --tag Mctrl
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --limit 20 --conditions none,bridge
"""

import argparse

from evaluation.cli_common import add_model_args, load_from_args, load_items, write_json
from evaluation.config import EvalConfig
from evaluation.runner import accuracy_table, evaluate, format_table, readable_items, solved_items


def main():
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="jbypass", choices=sorted(cfg.datasets))
    p.add_argument("--split_file", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--conditions", default=",".join(cfg.conditions))
    p.add_argument("--fields", default=",".join(cfg.prompt_fields))
    p.add_argument("--all_items", action="store_true", help="keep items the clean model gets wrong")
    p.add_argument("--tag", default="M")
    add_model_args(p)
    args = p.parse_args()

    js = load_from_args(args)
    conditions, fields = args.conditions.split(","), args.fields.split(",")
    items = readable_items(js, load_items(args.dataset, cfg, args.split_file, args.split, args.limit))
    if not args.all_items:
        items = solved_items(js, items, fields)
    print(f"[{args.tag}] {args.dataset}: {len(items)} items, band layers {js.band[0]}-{js.band[-1]}")

    records = evaluate(js, items, conditions, fields, seed=cfg.seed)
    table = accuracy_table(records)
    print(format_table(table, conditions))
    write_json({"tag": args.tag, "dataset": args.dataset, "split": args.split, "adapter": args.adapter,
                "band": js.band, "table": table, "records": records},
               cfg.out_dir / f"eval_{args.tag}_{args.dataset}_{args.split or 'all'}.json")


if __name__ == "__main__":
    main()
