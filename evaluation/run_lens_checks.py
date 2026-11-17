"""Validate the J-lens setup against the paper before trusting the ablations.

python -m evaluation.run_lens_checks --tag M

Prints:
  bridge readout pass@k on the paper's lens-eval-multihop set and on ours,
  clean and with the bridge projected out (should drop inside the band);
  swap success on the paper's probe-swap set.
"""

import argparse

from evaluation.cli_common import add_model_args, add_output_args, load_from_args, load_items, write_json
from evaluation.config import EvalConfig
from evaluation.lens_checks import bridge_readout, swap_success
from evaluation.runner import readable_items


def main():
    cfg = EvalConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="M")
    add_model_args(p)
    add_output_args(p, cfg)
    args = p.parse_args()
    js = load_from_args(args)

    out = {}
    for name in ("paper_multihop", "jbypass"):
        items = readable_items(js, load_items(name, cfg, limit=args.limit))
        for ablate in (False, True):
            key = f"{name}/{'bridge_ablated' if ablate else 'clean'}"
            out[key] = bridge_readout(js, items, cfg.readout_k, ablate_bridge=ablate)
            print(f"{key:32s} pass@{cfg.readout_k} = {out[key]['pass_at_k']:.2f} (n={out[key]['n']})")
    swap_items = readable_items(js, load_items("paper_probe_swap", cfg, limit=args.limit))
    out["paper_probe_swap/swap"] = swap_success(js, swap_items, cfg.max_answer_tokens)
    s = out["paper_probe_swap/swap"]
    print(f"{'paper_probe_swap/swap':32s} success = {s['success']:.2f} ({s['n_solved']} solved of {s['n']})")
    write_json(out, args.out_dir / f"lens_checks_{args.tag}.json")


if __name__ == "__main__":
    main()
