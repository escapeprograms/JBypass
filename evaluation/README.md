# evaluation

Evaluation suite for the baselines and the main experiment. It measures accuracy on two-hop and one-hop prompts under each ablation condition, checks that the J-lens setup reproduces the paper, analyzes how a model gets around the ablation (lens readout and linear probes), and checks perplexity.
Primary metric (proposal): exact match of a short greedy continuation, counting synonyms ("8" or "eight") and skipping filler like "a" or "the". The paper's stricter metric (greedy next token equals the answer) is recorded alongside it. On a 1.7B model that metric is harsh, because filler or a bare space before a digit often wins the first token.

Depends on `jspace/`.

## Run order
```
python -m evaluation.run_lens_checks --tag M                       # 1. does the lens behave like the paper?
python -m evaluation.make_splits --out results/splits.json         # 2. keep items M solves, split by bridge
python -m evaluation.run_eval --dataset jbypass --tag M            # 3. baseline table on M
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt
python -m evaluation.run_mechanism --tag M                         # 4. bridge readout by position + probes
python -m evaluation.run_perplexity --tag M                        # 5. held-out perplexity
```
Every script takes `--adapter path/to/adapter` to evaluate a fine-tuned model instead of M, and `--lens_path` to ablate with a different lens (e.g. one from `jspace.fit_lens`). `run_eval` and `run_mechanism` take `--split_file results/splits.json --split test_seen` (or `test_unseen`). `run_mechanism` takes `--extra_lens recomputed=path.pt` to compare a recomputed lens with the original side by side.
For a quick first try, add `--limit 10 --conditions none,bridge`. Results go to `results/`.

## Conditions (`conditions.py`)
| name | removes | role |
|---|---|---|
| `none` | nothing | clean accuracy |
| `bridge` | J-lens vectors of the item's bridge, over the band | the intervention |
| `wrong_bridge` | same, for `swap_to`, another bridge of the same kind | specificity control |
| `random_tok` | same number of random word tokens' J-lens vectors | Mrand-style control |
| `random_dir` | same number of Gaussian directions | "does any ablation hurt?" control |
| `topk_light`, `topk_medium`, `topk_heavy` | paper's ablation, top 10 active per position, over the first third, two thirds, or all of the band | positive control |

## Prompt types
- `prompt`: two-hop (main metric).
- `hop2_prompt`: bridge to answer, with the bridge written in the prompt. This is the fair one-hop control.
- `hop1_prompt`: subject to bridge. The answer is the bridge itself, so `bridge` ablation breaks it by construction. It only shows that the ablation has an effect.

## Files
- `config.py`: `EvalConfig`: dataset paths, conditions, prompt types, seed, answer length, readout k, probe and perplexity settings, output dir.
- `data_loading.py`
  - `TwoHopItem`: the one item type every dataset is loaded into.
  - `load_jbypass`, `load_paper_probe_swap`, `load_paper_multihop`, `load_dataset(name, path)`.
- `scoring.py`
  - `next_token_logits(js, prompt)`: final-position logits under any active intervention.
  - `greedy_continuation(js, prompt, make_intervention, n_tokens)`: greedy decoding that rebuilds the intervention at every step from the current text, so prompt-dependent ablations see the right sequence.
  - `matches(continuation, answers)`: exact match with synonyms and filler skipping.
  - `answer_token_ids(tokenizer, answers)`: first token of every spelling of each answer (for the paper's metric).
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
- `mechanism.py`
  - `bridge_ranks_all_positions(js, item, condition, rng)`: the bridge's J-lens rank per layer at the final prompt token and the best rank over the question span (every token after `Fact:`, which includes the bridge's description).
  - `summarize_ranks`: pass@k over all layers (paper's metric), pass@k within the band (where ablations act), and median rank per layer.
  - `collect_final_token`: residuals at the final token for every layer, under a condition (for the probes).
- `probes.py`
  - `probe_accuracy_by_layer(X, labels, groups, ...)`: cross-validated multinomial logistic regression that predicts the bridge from the residual at each layer. Folds are grouped by subject, so all questions about one landmark land in the same fold and the probe can't score by recognizing the landmark.
  - `strip_jspace(js, X, k)`: replaces each residual with its non-J-space part (`jspace.decompose`).
  - `chance_level`: majority-class accuracy.
- `perplexity.py`: `perplexity(js, texts, max_len)`: token-weighted perplexity.
- `cli_common.py`: shared CLI flags (`--adapter`, `--lens_path`, `--device`, `--limit`), item loading, JSON output.
- `run_eval.py`, `run_lens_checks.py`, `make_splits.py`, `run_mechanism.py`, `run_perplexity.py`: command-line entry points.
- `data/`
  - Prompts are stripped of trailing spaces on load. Qwen turns a trailing space into a lone token that derails the model, and the paper's prompts end with `"is "`.
  - `build_jbypass_dataset.py` writes `jbypass_two_hop.json`: 218 items, 66 bridges, in the paper's `Fact:` format, each with both one-hop prompts. For capital questions the landmark is not in the capital, so the model can't skip the bridge.
  - `probe-swap.json` (90 items), `lens-eval-multihop.json` (93 items): Anthropic's two-hop sets, copied unchanged from jacobian-lens (Apache 2.0, see `data/NOTICE`).

## Coverage of the proposal's Evaluation section
| proposal | where |
|---|---|
| exact-match two-hop accuracy under bridge ablation, synonyms count | `run_eval`, `bridge` condition, answer alias lists |
| seen-bridge and unseen-bridge test groups reported separately | `make_splits`, `--split test_seen / test_unseen` |
| compare Mexp with Mctrl, Mrand and M | `--adapter` on every script |
| M fails two-hop but passes one-hop under ablation | `hop2_prompt` rows |
| one-hop accuracy for each test item | `hop1_prompt`, `hop2_prompt`; per-item records in the output JSON |
| random ablation should not destroy performance | `random_tok`, `random_dir` |
| recompute the J-lens after training and ablate with it | `jspace.fit_lens`, then `--lens_path` |
| bridge readout at the description tokens and final token, original vs recomputed lens | `run_mechanism --extra_lens` |
| linear probe on the full activation and its non-J-space part | `run_mechanism` (probes) |
| sanity: bridge readout near zero under ablation | `run_lens_checks` and `run_mechanism`, `bridge` condition |
| sanity: perplexity on held-out text | `run_perplexity` |
| averages over seeds | each run is tagged; average the JSON outputs across seed runs |

## What the M baseline should look like if the premise holds
- `prompt`: high under `none`, low under `bridge`, high under `wrong_bridge`, `random_tok` and `random_dir`.
- `hop2_prompt`: high under every condition.
- Lens checks: the bridge is readable (pass@10 well above 0) on clean prompts, and swaps work on a good share of items.

If `bridge` barely hurts but `topk` does, removing the bridge alone isn't enough. That's a finding in itself, and the next step is to ablate more of the J-space.

## Tested
`tests/test_jspace_eval.py` runs without a model: loaders, split leakage, answer matching, probe separability and subject leakage, decomposition recovery, projection, and swap math.

Every script was also run end to end on Qwen3-1.7B with the prefit lens, on small item counts on an M-series Mac (`--device mps`). Smoke numbers, not results:
- Lens checks (30 items): bridge readout pass@10 is 0.77 on our dataset and 0.27 on the paper's multihop set. Bridge ablation drops it to 0.07 and 0.00. Swap success is 0.10 (1 of 10 solved). The paper reports 54% to 70% on Claude.
- run_eval (8 items), two-hop / hop2 exact match:
  - none: 1.00 / 1.00
  - bridge: 0.75 / 1.00
  - wrong_bridge, random_tok, random_dir: 1.00 / 1.00
  - topk_light: 0.62 / 0.75
  - topk_medium: 0.12 / 0.50
  - topk_heavy: 0.00 / 0.00
- run_mechanism (24 items), inside the band: the bridge readout drops from 0.88 to 0.00 under ablation. Probe accuracy is near chance in early layers and about 0.88 at layer 24 (chance 0.33).
