# evaluation

Evaluation suite for the baselines and the main experiment: accuracy on two-hop and one-hop prompts under each ablation condition, plus checks that our J-lens setup reproduces the paper.
Scoring follows the paper: greedy next token equals the answer.

Depends on `jspace/`.

## Run order
```
python -m evaluation.run_lens_checks --tag M                       # 1. does the lens behave like the paper?
python -m evaluation.make_splits --out results/splits.json         # 2. keep items M solves, split by bridge
python -m evaluation.run_eval --dataset jbypass --tag M            # 3. baseline table on M
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt
python -m evaluation.run_eval --dataset jbypass --split_file results/splits.json --split test_unseen --adapter runs/mctrl --tag Mctrl
```
For a quick first try, add `--limit 10 --conditions none,bridge`. Results go to `results/`.

## Conditions (`conditions.py`)
| name | removes | role |
|---|---|---|
| `none` | nothing | clean accuracy |
| `bridge` | J-lens vectors of the item's bridge, over the band | the intervention |
| `wrong_bridge` | same, for `swap_to`, another bridge of the same kind | specificity control |
| `random_tok` | same number of random word tokens' J-lens vectors | Mrand-style control |
| `random_dir` | same number of Gaussian directions | "does any ablation hurt?" control |
| `topk` | paper's ablation, top 10 active per position | positive control |

## Prompt types
- `prompt`: two-hop (main metric).
- `hop2_prompt`: bridge to answer, with the bridge written in the prompt. This is the fair one-hop control.
- `hop1_prompt`: subject to bridge. The answer is the bridge itself, so `bridge` ablation breaks it by construction. It only shows that the ablation has an effect.

## Files
- `config.py`: `EvalConfig`: dataset paths, conditions, prompt types, seed, readout k, output dir.
- `data_loading.py`
  - `TwoHopItem`: the one item type every dataset is loaded into.
  - `load_jbypass`, `load_paper_probe_swap`, `load_paper_multihop`, `load_dataset(name, path)`.
- `scoring.py`
  - `next_token_logits(js, prompt)`: final-position logits under any active intervention.
  - `answer_token_ids(tokenizer, answers)`: first token of every spelling of each answer.
  - `is_correct(logits, ids)`.
- `conditions.py`: one builder per condition, `(js, item, prompt, rng) -> intervention or None`, registered in `CONDITIONS`.
- `runner.py`
  - `evaluate(js, items, conditions, fields)`: one record per item, prompt type, and condition.
  - `solved_items`: items the clean model gets right on every prompt type.
  - `readable_items`: drops bridges with no single-token spelling.
  - `accuracy_table`, `format_table`.
- `lens_checks.py`
  - `bridge_readout`: the paper's lens-eval pass@k for the bridge, clean or with the bridge ablated.
  - `swap_success`: the paper's swap check. On Claude the paper reports 54% to 70%.
- `splits.py`: `split_by_bridge(items, unseen_frac, seed)` into `train`, `test_seen`, `test_unseen`.
- `cli_common.py`: shared CLI flags (`--adapter`, `--lens_path`, `--device`, `--limit`), item loading, JSON output.
- `run_eval.py`, `run_lens_checks.py`, `make_splits.py`: command-line entry points.
- `data/`
  - `build_jbypass_dataset.py` writes `jbypass_two_hop.json`: 218 items, 66 bridges, in the paper's `Fact:` format, each with both one-hop prompts. For capital questions the landmark is not in the capital, so the model can't skip the bridge.
  - `probe-swap.json` (90 items), `lens-eval-multihop.json` (93 items): Anthropic's two-hop sets, copied unchanged from jacobian-lens (Apache 2.0, see `data/NOTICE`).

## What the M baseline should look like if the premise holds
- `prompt`: high under `none`, low under `bridge`, high under `wrong_bridge`, `random_tok` and `random_dir`.
- `hop2_prompt`: high under every condition.
- Lens checks: the bridge is readable (pass@10 well above 0) on clean prompts, and swaps work on a good share of items.

If `bridge` barely hurts but `topk` does, removing the bridge alone isn't enough. That's a finding in itself, and the next step is to ablate more of the J-space.

Not tested on a real model yet: the code passes a syntax check only.
