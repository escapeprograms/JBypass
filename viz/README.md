# viz

Interactive J-space viewer. Pick a model (a base profile with a prefit lens, or a saved adapter in `runs/`), an example (or a random one), a prompt field and one ablation condition (`none` = clean run); the model generates, and the J-lens readout of every layer over the prompt and the response is shown as jlens's heatmap page (position x layer grid, top-n on click, rank charts for tracked tokens, with the bridge and the answers pinned).

Depends on `evaluation/` (conditions, items, scoring), `jspace/` (readout) and `model/` (loading, inference). Only `app.py` and `run_app.py` import gradio; the rest is plain Python and tested in `tests/test_viz.py`. Settings live in `configs/viz.yaml`.

## How one run works
1. `run.run_example`: encode the prompt, greedy-decode under the condition (same decoding and scoring as eval) -> `prompt_ids + response_ids`.
2. `slice.run_slice`: rebuild the condition's interventions once on the full sequence, one forward pass, capture the post-intervention residual at every readout layer.
3. Per layer: transport by J_l, unembed -> top-n ids and tracked-token ranks -> `jlens.vis.SliceData` (same outputs as `jlens.vis.compute_slice`, but on our own ids).
4. `page.slice_page` -> self-contained HTML (`build_page(mode="embed")`), shown via `page.iframe`; `slice.replay_mismatches` checks the final row's top-1 reproduces every generated token.

Rows at ablated layers show the post-edit residual (under `bridge` the bridge is projected out there by construction). Adapters are read with the base model's prefit lens.

## Files
- `__init__.py` — package docstring only; imports nothing (so the core works without gradio).
- `config.py` — viewer settings.
  - `VizConfig` — frozen: `runs_dir`, `split_file` (or None), `top_n`, `max_tracked` (or None = all), `page_height`, `server_name`, `server_port`.
  - `load_viz_config(path=DEFAULT_VIZ_CONFIG)` — read the YAML; `runs_dir` / `split_file` resolved against the file's folder; `split_file: null` -> None.
  - `DEFAULT_VIZ_CONFIG` — path of `configs/viz.yaml`.
- `catalog.py` — what can be chosen (file access only); invalid combinations are filtered out, never caught.
  - `ModelChoice(profile, adapter=None)` — frozen; `.label` is `"qwen3-1.7b"` or `"qwen3-1.7b + <adapter folder>"`.
  - `lens_profiles(profiles_dir)` — sorted profile names with a `prefit_lens`.
  - `model_choices(profiles_dir, runs_dir)` — each lens profile, then each `runs_dir/*/` whose saved profile (`model.lora.PROFILE_FILE`) is a lens profile; a missing `runs_dir` gives just the profiles.
  - `ALL_SPLITS` — `"all"`, the split choice meaning no split filter.
  - `split_names(items, split_file)` — `ALL_SPLITS` plus each split in the file naming at least one item; `[ALL_SPLITS]` when `split_file` is None.
  - `examples(js, dataset, split, eval_cfg, split_file)` — the dataset's items in the split whose bridge (and `swap_to`) has a single-token spelling (`readable_items`).
  - `valid_fields(item, fields)` — the prompt fields that are not None on the item.
  - `valid_conditions(item, conditions)` — drops `wrong_bridge` when the item has no `swap_to`.
  - `pick_random(items, rng)` — one item from a `random.Random`.
- `session.py` — at most one model in memory.
  - `Session(load)` — `.get(choice)` returns the cached JSpace for the current choice, else releases it and calls `load(choice)`; `.release()` drops the model, runs `gc.collect()` and empties the CUDA cache. Switching adapters reloads the whole model.
  - `make_loader(run, device)` — `choice -> load_frozen_jspace(profile, choice.adapter)` with the profile from `run.profiles_dir`, on `device` (None = the profile's device), using the profile's prefit lens.
- `run.py` — one generation.
  - `RunResult` — frozen, plain Python values only: `item, field, condition, seed, prompt, prompt_ids, response_ids, response_text, answers, correct`.
  - `run_example(js, item, field, condition, seed, max_new_tokens)` — greedy-decode the item's field under the condition; text decoded and scored exactly as `evaluation.runner.score_one` does.
- `slice.py` — token ids -> `jlens.vis.SliceData` (tensors only, no HTML).
  - `pinned_tokens(js, result)` — sorted ids of the bridge's single-token spellings plus the answers' first tokens.
  - `slice_from_ids(js, enc, interventions, top_n, max_tracked, pinned)` — one forward over a single unpadded row with the interventions; per readout layer (`readout_layers`: fitted layers + final) top-n ids, then tracked tokens (jlens's sum-of-1/(rank+1) rule capped by `max_tracked`, plus `pinned`), then their full-vocab ranks. Logits are made one layer at a time.
  - `run_slice(js, result, top_n, max_tracked)` — `slice_from_ids` on `prompt_ids + response_ids` with the condition's interventions rebuilt on that sequence and the bridge/answers pinned.
  - `replay_mismatches(slice_data, result)` — response indices whose generated token is not the final layer's top-1 at the previous position; `[]` means the readout reproduces generation (bf16 near-ties on GPU may break this, so it is reported, not raised).
  - `RANK_CHUNK` — positions per full-vocab sort when ranking.
- `page.py` — the only HTML (and, through jlens, network) code.
  - `slice_page(slice_data, result, title)` — `build_page(mode="embed")` HTML; `title` names the model, the description gives the condition and field and notes that ablated rows show the post-edit residual. d3 is fetched from a CDN once per process.
  - `iframe(page, height)` — the page in a `srcdoc` iframe (HTML-escaped).
  - `summary(result, mismatches)` — markdown: prompt, bridge, expected answers, response (via `one_line`), correct/incorrect, which positions are the response, and the replay check.
  - `one_line(text)` — newlines shown as ⏎ (as the heatmap shows them) so the response stays one markdown line.
  - `item_card(item, field)` — markdown card for the chosen example before a run: name, category, the field's prompt, bridge, swap bridge, expected answers.

## App (`app.py`, `run_app.py`)
Launch from the repo root: `python -m viz.run_app` (flags: `--config`, `--model`, `--device`, `--viz_config`), then open `http://127.0.0.1:7860` (`server_name` / `server_port` in `configs/viz.yaml`). The first run in a process downloads d3 from a CDN (jlens inlines it into the page), so it needs internet once. Needs `gradio==6.32.0` (in `environment.yml`).

UI flow: on page open the default model loads (the example list needs its tokenizer, so this happens on `demo.load`, with Gradio's progress indicator on the summary), then the dataset's splits, the split's examples, the first example's valid fields and conditions, and its card are filled in. Changing a dropdown re-runs its own step and every later one (model -> dataset -> split -> example -> card); **Random example** picks an example and continues from the example step. **Run** generates and shows the summary and the heatmap. All events that touch the model share one queue slot (`MODEL_QUEUE`), so two tabs never load or run at once.

Reading the result: the summary's last line is the replay check: whether the final layer's top-1 at each position, from the single forward over prompt + response, reproduces the generated token. On GPU bf16 a near-tie can flip between generation (one growing sequence per step) and the replay (one full-length forward), so some positions may be listed; the heatmap then shows the replay's numbers at those positions. Rows at ablated layers show the post-edit residual, so under `bridge` the bridge is gone there by construction. Adapters are read with the base model's prefit lens (no refit lens per adapter). The page logs a harmless `history.replaceState` SecurityError in the browser console: jlens saves its view state in the page URL, which a `srcdoc` iframe cannot do; rendering and clicks are unaffected.

- `app.py` — Gradio wiring only (no model or tensor code).
  - `AppContext` — `run` (RunConfig), `viz` (VizConfig), `eval_cfg` (EvalConfig), `session` (the only holder of the loaded model; never put in `gr.State`), `choices` (label -> ModelChoice), `default_model` (a label), `rng` (for Random example).
  - `MODEL_QUEUE` — shared `concurrency_id` / `concurrency_limit=1` for every model-touching event.
  - `on_model(ctx, label)` — load the model via `session.get`; returns a "Loaded" note and clears the old page.
  - `on_dataset(ctx, label, dataset)` — split dropdown: `split_names` of the dataset's readable items, reset to `all`.
  - `on_split(ctx, label, dataset, split)` — the split's examples as `{name: TwoHopItem}` (kept in a per-tab `gr.State` so later events don't re-read the dataset) and the example dropdown, first selected.
  - `on_example(ctx, items, name, field, condition)` — field and ablation dropdowns limited to `valid_fields` / `valid_conditions`; the current choice is kept if still valid.
  - `on_card(items, name, field)` — the example card (`page.item_card`).
  - `on_random(ctx, items)` — example dropdown set to `pick_random`.
  - `on_run(ctx, label, items, name, field, condition)` — `run_example` -> `run_slice` -> `replay_mismatches` -> `slice_page` -> `(summary, iframe)`; `gr.Error` only if no example/field/ablation is selected.
  - `build_app(ctx)` — the `gr.Blocks`: `_controls` lays out the components, `_wire` connects them.
  - `_steps` / `_cascade` / `_tail` / `_chain` — the refresh chain as an ordered list of event kwargs (each with an `api_name`); a trigger (`gr.on`) runs from its step to the end with `.success`, so a failed step stops the chain. Only user `.input` events start chains, never `.change`.
  - Embedding: `gr.HTML(js_on_load=None, apply_default_css=False)` holding `page.iframe`'s `<iframe srcdoc>`. Gradio 6 renders `value` raw through `innerHTML` (no sanitising); the `${value}` template inserts it as a JS variable, so `${...}` and backticks inside the page are never re-evaluated. The iframe's inline height sets its size.
- `run_app.py` — entry point.
  - `parse_args(argv=None)` — `model.cli.add_profile_args` (`--config --model --device`) plus `--viz_config`.
  - `build_context(args)` — `resolve_profile`, `load_viz_config`, `model_choices` keyed by label, default = the base choice of `--model` (or run.yaml's model; must have a prefit lens), `EvalConfig()` defaults (debt-002), `Session(make_loader(run, args.device))`, `random.Random(eval_cfg.seed)`.
  - `main()` — `build_app(build_context(...)).launch(server_name, server_port, show_error=True)`.
