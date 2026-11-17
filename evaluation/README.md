# evaluation

Evaluation suite for the baselines and the main experiment. It measures accuracy on two-hop and one-hop prompts under each ablation condition, checks that the J-lens setup reproduces the paper, analyzes how a model gets around the ablation (lens readout and linear probes), and checks perplexity.
Primary metric (proposal): exact match of a short greedy continuation, counting synonyms ("8" or "eight") and skipping filler like "a" or "the". The paper's stricter metric (greedy next token equals the answer) is recorded alongside it. On a 1.7B model that metric is harsh, because filler or a bare space before a digit often wins the first token.

Depends on `jspace/` and `model/`: models are loaded from the profile by `model.loading.load_model`, decoding and losses are `model.inference`, and ablations are `model.hooks.Intervention`s. `finetune/` reuses the dataset, the conditions and `readable_items`.

## Run order
```
python -m evaluation.run_lens_checks --tag M                       # 1. does the lens behave like the paper?
python -m evaluation.make_splits --out results/splits.json         # 2. keep items M solves, split by bridge
python -m evaluation.run_eval --dataset jbypass --tag M            # 3. baseline table on M
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt
python -m evaluation.run_mechanism --tag M                         # 4. bridge readout by position + probes
python -m evaluation.run_perplexity --tag M                        # 5. held-out perplexity
python -m evaluation.show_table "results/eval_*.json"              # reprint saved eval tables (no model)
```
Every script takes `--config` (run.yaml) and `--model` (profile; default from run.yaml), `--device` (default = the profile's device; `--device mps` on a Mac), `--adapter path/to/adapter` to evaluate a fine-tuned model instead of M (an adapter saved by `finetune.run_train` / `model.lora.save_adapter`, loaded unmerged), and `--lens_path` to ablate with a different lens (e.g. one from `jspace.fit_lens`; required for a profile without `prefit_lens`). `run_eval` and `run_mechanism` take `--split_file results/splits.json --split test_seen` (or `test_unseen`). `run_mechanism` takes `--extra_lens recomputed=path.pt` to compare a recomputed lens with the original side by side.
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
  - `TwoHopItem`: the one item type every dataset is loaded into. `bridge_zh`: simplified Chinese spellings of the bridge, full name first; `()` for datasets without them (the paper sets).
  - `load_jbypass`, `load_paper_probe_swap`, `load_paper_multihop`, `load_dataset(name, path)`. All read the JSON as UTF-8; `load_jbypass` requires the `bridge_zh` key.
- `scoring.py`
  - `greedy_continuation(lm, prompt, interventions_for, n_tokens)`: thin wrapper over `model.inference.greedy_tokens`; `interventions_for(enc)` rebuilds the interventions at every step from the current token ids, so sequence-dependent ablations see the right sequence. Decodes the generated token ids (no re-tokenising of text) and stops early at EOS. Returns (continuation text, first generated token id).
  - `matches(continuation, answers)`: exact match with synonyms and filler skipping.
  - `FILLER_WORDS`: `FILLER_PREFIXES` without their trailing space ("a", "the", ...).
  - `is_answer_token(text)`: a decoded token is not whitespace, a bare quote or a filler word.
  - `answer_index(tokenizer, response_ids)`: index of the first response token `is_answer_token` accepts (0 if none); the per-token mirror of `matches` that the viewer uses to stop the heatmap at the answer. Unlike `matches` it treats " a" as filler without a following space and does not stop at a newline; `correct` still comes from `matches`.
  - `answer_token_ids(tokenizer, answers)`: first token of every spelling of each answer (for the paper's metric).
- `conditions.py`
  - `Builder = (js, item, enc, rng) -> Sequence[Intervention]`; one builder per condition, registered in `CONDITIONS`; `()` for the clean run. `enc` is the current one-row token sequence (`model.inference.Encoded`). The top-k builders run a clean pass on it for the exclude mask.
  - `item_rng(seed, item, condition)`: the per-item random source; one seeding rule for eval, mechanism and training.
  - `interventions_for_item(js, item, condition, seed)`: an `InterventionsFor` (`enc -> interventions`) that calls the condition's builder with a fresh `item_rng` on every call, so random ablations stay fixed across decoding steps.
- `runner.py`
  - `evaluate(js, items, conditions, fields, seed, n_tokens)`: one record per item, prompt type, and condition.
  - `score_one(js, item, field, condition, seed, n_tokens)`: one record; decodes with `conditions.interventions_for_item`, so the builder runs every step and random ablations stay fixed.
  - `solved_items`: items the clean model gets right on every prompt type.
  - `readable_items`: drops bridges with no single-token spelling.
  - `accuracy_table`, `format_table`.
- `lens_checks.py`
  - `bridge_ranks` / `bridge_readout`: the paper's lens-eval pass@k for the bridge, clean or with the bridge ablated (the ablation is passed to `jspace.token_ranks`).
  - `swap_success`: the paper's swap check. On Claude the paper reports 54% to 70%.
- `splits.py`: `split_by_bridge(items, unseen_frac, seed)` into `train`, `test_seen`, `test_unseen`.
- `mechanism.py`
  - `question_start(js, prompt)`: index of the first token after the `Fact:` prefix, in jlens's encoding.
  - `bridge_ranks_all_positions(js, item, condition, rng)`: the bridge's J-lens rank per layer at the final prompt token and the best rank over the question span (every token after `Fact:`, which includes the bridge's description), via `jspace.lens_logits` under the condition.
  - `summarize_ranks`: pass@k over all layers (paper's metric), pass@k within the band (where ablations act), and median rank per layer.
  - `collect_final_token`: residuals at the final token for every layer, under a condition (for the probes), via `model.inference.forward_residuals`, so it records the edited residual.
- `probes.py`
  - `probe_accuracy_by_layer(X, labels, groups, ...)`: cross-validated multinomial logistic regression that predicts the bridge from the residual at each layer. Folds are grouped by subject, so all questions about one landmark land in the same fold and the probe can't score by recognizing the landmark.
  - `strip_jspace(js, X, k)`: replaces each residual with its non-J-space part (`jspace.decompose`).
  - `chance_level`: majority-class accuracy.
- `perplexity.py`: `perplexity(lm, texts, max_len, batch_size)`: token-weighted perplexity over all texts, each truncated to `max_len` tokens, texts under 2 tokens skipped, computed in left-padded batches with `model.inference.sequence_nll`.
- `cli_common.py`
  - `add_model_args(parser)`: `--config`, `--model`, `--device` (from `model.cli`), `--adapter`, `--lens_path`, `--limit`.
  - `load_frozen_model(profile, adapter=None)`: the profile's model plus the adapter if given (unmerged, `model.lora.load_adapter`), with every weight frozen (`model.loading.freeze`; the probes' decomposition relies on it).
  - `load_frozen_jspace(profile, adapter=None, lens_path=None)`: the above plus the lens (`lens_path`, or the profile's prefit lens), as a `JSpace` with the default `JSpaceConfig` (`TODO(debt-002)`).
  - `load_model_from_args(args)`: `load_frozen_model` with the profile and `--adapter` from the flags. Used by `run_perplexity`, which needs no lens.
  - `load_from_args(args)`: `load_frozen_jspace` with the profile, `--adapter` and `--lens_path` from the flags.
  - `load_items(name, cfg, split_file, split, limit)`, `write_json(obj, path)`.
- `run_eval.py`, `run_lens_checks.py`, `make_splits.py`, `run_mechanism.py`, `run_perplexity.py`: command-line entry points.
- `show_table.py`: command-line tool that reprints `run_eval` result tables without loading a model.
  - `expand(patterns) -> list[Path]`: paths or glob patterns (expanded here, since PowerShell doesn't), each pattern's matches sorted; raises `FileNotFoundError` if a pattern matches nothing.
  - `load_result(path) -> dict`: reads a result JSON as UTF-8.
  - `describe(path, result) -> str`: header (file name, tag, dataset, split, adapter or "base model", distinct items) plus the exact-match and next-token tables, rebuilt from `records` with `runner.accuracy_table` / `format_table`, so old and new files read the same.
- `data/`
  - Prompts are stripped of trailing spaces on load. Qwen turns a trailing space into a lone token that derails the model, and the paper's prompts end with `"is "`.
  - `build_jbypass_dataset.py` writes `jbypass_two_hop.json`: 1032 items, 156 bridges (89 countries, 39 US states, 12 animals, 4 planets, 7 elements, 5 people), in the paper's `Fact:` format, each with both one-hop prompts. All tables live in this file.
    - Idea: a two-hop prompt is the hop-2 template with the bridge replaced by a reference to the subject (`"the country where Mont Saint-Michel is located"`). The dataset is the cross product bridge x its subjects x its relations.
    - `RELATIONS`: hop-2 template per category and relation (`{b}` = bridge phrase or subject reference, `{B}` = capitalised).
    - `KINDS`: how a subject points at its bridge, as (hop-1 template, reference template): `place` ("is located in"), `person` ("was born in"), `company` ("is headquartered in"), `description` ("the lightest element is"), `known_as` ("the planet known as the Red Planet is").
    - `Bridge`: `name` (the ablated word), `attrs` (`{relation: [answers]}`, first = training target), `subjects` (`[(kind, text)]`), `phrase` (how hop 2 names it, e.g. "a spider"), `lead` (hop-1 lead-in, e.g. " the", " Albert"). Built by `country`, `state`, `described`, `animal`; `None` for a country attribute omits that relation.
    - `compose(category, bridge, kind, subject, relation)`: one item. `build_items()`: the cross product. `validate(items)`: raises on duplicate names, a bridge in two categories, or a bridge named in its own two-hop prompt. `add_swaps(items, seed)`: `swap_to` = another bridge of the same category.
    - `BRIDGE_ZH`: `{bridge name: [simplified Chinese spellings]}` for every bridge, full name first, then common short forms (`"California": ["加利福尼亚", "加州"]`). `check_bridge_zh(bridges, table)`: raises naming bridges without an entry, keys that are not bridges, and empty / all-ASCII spellings; `main()` calls it first.
    - Rules: no subject is in, born in, or headquartered in the bridge's capital, or best known for its ties to it (Rosa Parks and Montgomery), so the model can't skip the bridge for capital questions; every bridge has simplified Chinese spellings, full name first (they need not be single tokens; consumers filter); a subject never contains its bridge's name; every new bridge name is a single Qwen token (Curie, kangaroo and octopus are not, and are dropped by `readable_items`); a relation is omitted when its answer contains the bridge name (Singapore, Guatemala capitals) or is disputed.
    - Row keys: `name` (`{category}-{bridge}-{relation}-{subject slug}`, no index, so table edits don't rename other items), `category`, `relation`, `bridge`, `answers`, `subject`, `subject_kind`, `hop1_prompt`, `prompt`, `hop2_prompt`, `bridge_zh`, `swap_to`. `load_jbypass` ignores `relation`, `subject` and `subject_kind`.
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
`tests/test_jspace_eval.py` runs without a model: loaders, split leakage, the dataset generator's invariants, answer matching, probe separability and subject leakage, decomposition recovery, projection, and swap math. `tests/test_evaluation_model.py` runs decoding, the condition builders and perplexity on a tiny random model with a synthetic lens.

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
