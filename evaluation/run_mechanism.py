"""Mechanism analysis from the proposal: bridge readout at the positions where it
is computed (original vs recomputed lens), and linear probes on the full
activation vs its non-J-space part.

python -m evaluation.run_mechanism --tag M
python -m evaluation.run_mechanism --adapter runs/some_adapter --extra_lens recomputed=lenses/recomputed.pt \
    --split_file results/splits.json --split test_unseen --tag some_adapter
"""

import argparse
import dataclasses

from evaluation.cli_common import add_model_args, add_output_args, load_from_args, load_items, write_json
from evaluation.conditions import item_rng
from evaluation.config import EvalConfig
from evaluation.mechanism import bridge_ranks_all_positions, collect_final_token, summarize_ranks
from evaluation.probes import chance_level, probe_accuracy_by_layer, strip_jspace
from evaluation.runner import readable_items
from jspace import load_lens


def readout_section(js, items, conditions, cfg) -> dict:
    out = {}
    for cond in conditions:
        per_item = [bridge_ranks_all_positions(js, it, cond, item_rng(cfg.seed, it, cond)) for it in items]
        out[cond] = summarize_ranks(per_item, cfg.readout_k, js.band)
        f, q = out[cond]["final"], out[cond]["question_best"]
        print(f"  readout {cond:12s} pass@{cfg.readout_k} all layers / band: "
              f"final={f['pass_at_k']:.2f}/{f['band_pass_at_k']:.2f} "
              f"question_best={q['pass_at_k']:.2f}/{q['band_pass_at_k']:.2f}")
    return out


def probe_section(js, items, conditions, cfg) -> dict:
    labels = [it.bridge for it in items]
    groups = [subject_key(it) for it in items]
    out = {"chance": chance_level(labels, groups, cfg.probe_folds)}
    for cond in conditions:
        X = collect_final_token(js, items, cond, lambda it: item_rng(cfg.seed, it, cond))
        args = (labels, groups, cfg.probe_folds, cfg.probe_steps, cfg.probe_weight_decay)
        out[cond] = {"full": probe_accuracy_by_layer(X, *args),
                     "non_jspace": probe_accuracy_by_layer(strip_jspace(js, X, js.cfg.decompose_k), *args)}
        print(f"  probe   {cond:12s} best layer acc: full={max(out[cond]['full'].values()):.2f} "
              f"non_jspace={max(out[cond]['non_jspace'].values()):.2f} (chance {out['chance']:.2f})")
    return out


def subject_key(item) -> str:
    """Items sharing a hop-1 prompt are about the same subject (one landmark,
    several relations). Datasets without hop-1 prompts group by the prompt."""
    return item.hop1_prompt or item.prompt


def main():
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="jbypass", choices=sorted(cfg.datasets))
    p.add_argument("--split_file", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--extra_lens", action="append", default=[], help="label=path, repeatable")
    p.add_argument("--conditions", default=",".join(cfg.mechanism_conditions))
    p.add_argument("--tag", default="M")
    add_model_args(p)
    add_output_args(p, cfg)
    args = p.parse_args()

    js = load_from_args(args)
    items = readable_items(js, load_items(args.dataset, cfg, args.split_file, args.split, args.limit))
    lenses = {"original": js.lens}
    for spec in args.extra_lens:
        label, path = spec.split("=", 1)
        lenses[label] = load_lens(js.lm, path)

    out = {}
    for label, lens in lenses.items():
        print(f"[{args.tag}] lens={label}, {len(items)} items")
        js_l = dataclasses.replace(js, lens=lens)
        conds = args.conditions.split(",")
        out[label] = {"readout": readout_section(js_l, items, conds, cfg),
                      "probes": probe_section(js_l, items, conds, cfg)}
    write_json(out, args.out_dir / f"mechanism_{args.tag}_{args.dataset}_{args.split or 'all'}.json")


if __name__ == "__main__":
    main()
