# viz

Interactive J-space viewer. Pick a model (a base profile with a prefit lens, or a saved adapter in `runs/`), an example (or a random one), a prompt field and one ablation condition (`none` = clean run); the model generates, and the J-lens readout of every layer over the prompt and the response is shown as jlens's heatmap page (position x layer grid, top-n on click, rank charts for tracked tokens, with the bridge (English spellings and its single-token Chinese spellings) and the answers pinned).

Depends on `evaluation/` (conditions, items, scoring), `jspace/` (readout) and `model/` (loading, inference). Only `app.py` and `run_app.py` import gradio; the rest is plain Python and tested in `tests/test_viz.py`. Settings live in `configs/viz.yaml`.

## How one run works
1. `run.run_example`: encode the prompt, greedy-decode under the condition (same decoding and scoring as eval) -> `prompt_ids + response_ids`, plus `answer_index` (the first response token that is not whitespace, a quote or a filler word); the shown sequence is `prompt_ids + shown_ids`, i.e. the prompt and the response up to and including the answer token.
2. `slice.run_slice`: rebuild the condition's interventions once on the shown sequence, one forward pass, capture the post-intervention residual at every readout layer.
3. Per layer: transport by J_l, unembed -> top-n ids and tracked-token ranks -> `jlens.vis.SliceData` (same outputs as `jlens.vis.compute_slice`, but on our own ids).
4. `page.slice_page` -> self-contained HTML (`build_page(mode="embed")`); when a dictionary is loaded, `gloss.gloss_payload` + `page.with_glosses` add the translation box; shown via `page.iframe`. `slice.replay_mismatches` checks the final row's top-1 reproduces every shown generated token.

Rows at ablated layers show the post-edit residual (under `bridge` the bridge is projected out there by construction). Adapters are read with the base model's prefit lens.

## Translation box (click a Chinese token)
Clicking a token in the By Layer / By Pos lists, a pinned chip, the prompt strip, or a grid cell opens a box at the bottom right with the token, then per word its pinyin (tone numbers, as stored) and numbered senses. Other scripts (Kana, Hangul, Cyrillic, ...) show "no dictionary entry (CC-CEDICT covers Chinese only)"; Chinese characters with no entry show "no dictionary entry"; English, punctuation and undecodable tokens hide the box. × closes it. jlens's own click behaviour (pin toggle, selection) is unchanged: the box only listens. Not translated: the hover tooltip, the grid's cell text and the x-axis labels. Longest-match splitting is not real word segmentation, so long fragments can split oddly.

Translations come from CC-CEDICT, read once at startup from `dictionary_file` in `configs/viz.yaml` and embedded per page (only the page's foreign tokens; tens of KB per page). **Download it by hand:** save `cedict_1_0_ts_utf-8_mdbg.txt.gz` from https://www.mdbg.net/chinese/dictionary?page=cc-cedict into `downloads/` at the repo root (the site prohibits scripted access, so there is no download command). With the file missing, startup stops with an error saying where to put it; `dictionary_file: null` turns the box off. License and attribution: `NOTICE`.

## Files
- `__init__.py` — package docstring only; imports nothing (so the core works without gradio).
- `config.py` — viewer settings.
  - `VizConfig` — frozen: `runs_dir`, `split_file` (or None), `dictionary_file` (or None = no translations), `top_n`, `max_tracked` (or None = all), `page_height`, `server_name`, `server_port`.
  - `load_viz_config(path=DEFAULT_VIZ_CONFIG)` — read the YAML; `runs_dir` / `split_file` / `dictionary_file` resolved against the file's folder; `null` -> None (`_optional_path`).
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
  - `RunResult` — frozen, plain Python values only: `item, field, condition, seed, prompt, prompt_ids, response_ids, response_text, answers, correct, answer_index` (`answer_index` indexes `response_ids` at the token scoring compares against, from `evaluation.scoring.answer_index`). Property `shown_ids` = `response_ids[:answer_index + 1]`, what the heatmap shows after the prompt.
  - `run_example(js, item, field, condition, seed, max_new_tokens)` — greedy-decode the item's field under the condition; text decoded and scored exactly as `evaluation.runner.score_one` does.
- `slice.py` — token ids -> `jlens.vis.SliceData` (tensors only, no HTML).
  - `pinned_tokens(js, result)` — sorted ids of the bridge's single-token English variants (`token_variants`), the single-token Chinese spellings from the item's `bridge_zh` (`jspace.single_token_ids`), and the answers' first tokens. Multi-token Chinese spellings are not pinned (pinning their first character would mislead).
  - `slice_from_ids(js, enc, interventions, top_n, max_tracked, pinned)` — one forward over a single unpadded row with the interventions; per readout layer (`readout_layers`: fitted layers + final) top-n ids, then tracked tokens (jlens's sum-of-1/(rank+1) rule capped by `max_tracked`, plus `pinned`), then their full-vocab ranks. Logits are made one layer at a time.
  - `run_slice(js, result, top_n, max_tracked)` — `slice_from_ids` on `prompt_ids + shown_ids` (prompt + response up to the answer token) with the condition's interventions rebuilt on that sequence and the bridge/answers pinned.
  - `replay_mismatches(slice_data, result)` — indices into `shown_ids` whose generated token is not the final layer's top-1 at the previous position; `[]` means the readout reproduces generation (bf16 near-ties on GPU may break this, so it is reported, not raised).
  - `RANK_CHUNK` — positions per full-vocab sort when ranking.
- `cedict.py` — CC-CEDICT, standard library only.
  - `CEDICT_PAGE`, `CEDICT_FILE` — where the dictionary is downloaded (by hand) and the file's name.
  - `Entry(traditional, simplified, pinyin, senses)` — frozen; `pinyin` verbatim, `senses` the `/`-separated senses (`;` stays inside a sense).
  - `Segment(word, entries)` — frozen; `entries == ()` means no dictionary entry.
  - `Cedict(index, max_len, n_entries)` — headword (traditional and simplified) -> its entries in file order, proper nouns (capitalised pinyin) last; longest headword; parsed lines.
  - `parse_line(line)` — one line, either format (`繁 简 [pin1 yin1] /a/b/` or `繁 简 [[pin1yin1]] /a; b/c/`); `ValueError` otherwise.
  - `parse_cedict(lines)` — skips blank and `#` lines; any other bad line raises with its line number; no entries raises.
  - `segment(cedict, text)` — longest match at each position; unmatched characters merge into one entry-less segment; whitespace dropped.
  - `load_cedict(path)` — the only file access (gzip, `utf-8-sig`); a missing file raises naming the path, the download page and `dictionary_file: null`.
- `gloss.py` — which tokens get a translation; the JSON the box reads.
  - `is_foreign(text)` — any letter above U+024F (`LAST_LATIN`), so accented Latin is not foreign.
  - `token_gloss(cedict, text)` — `{"text": stripped token, "segments": [{"word", "entries": [{"reading", "senses"}]}]}`.
  - `gloss_payload(slice_data, cedict)` — `{"context_ids": prompt-strip ids (from ctx_offset), "glosses": {str(id): token_gloss}}` for every foreign token in `vocab_fragment` (a foreign token with no hits still gets a key, so the box can say so).
- `static/gloss.js`, `static/gloss.css` — the box (injected into the page; package data). The script reads token ids from the page's own elements: list cells and pinned chips by `data-tid` (capture-phase listener, since jlens stops propagation there; a chip's × is ignored), prompt-strip tokens by index into `context_ids`, grid cells by reading the active By Layer row's first cell after jlens's own grid handler has re-rendered (bubble phase; ignored unless jlens's hover outline shows the pointer is on a cell). It builds the box with `textContent` only and never stops or cancels an event. Depends on jlens page internals (TODO(debt) at its top; `test_jlens_page_contract` guards it). The CSS sets the box size as custom properties at the top; `z-index` 900, below the page tooltip.
- `NOTICE` — CC-CEDICT source and CC BY-SA 4.0 attribution.
- `page.py` — the only HTML (and, through jlens, network) code.
  - `slice_page(slice_data, result, title)` — `build_page(mode="embed")` HTML; `title` names the model, the description gives the condition and field, says positions cover the prompt and the response up to the answer token, and notes that ablated rows show the post-edit residual. The text passed to `build_page` is the joined `context_token_strs` (jlens never renders it). d3 is fetched from a CDN once per process.
  - `with_glosses(page, payload)` — inserts `<style>` (gloss.css), `<script type="application/json" id="jb-gloss-data">` (the payload, `<` escaped as `<`) and `<script>` (gloss.js) before the last `</body>`; raises if there is none.
  - `GLOSS_DATA_ID` — `"jb-gloss-data"`, the data script's id.
  - `iframe(page, height)` — the page in a `srcdoc` iframe (HTML-escaped).
  - `summary(result, mismatches)` — markdown: prompt, bridge (via `bridge_label`), expected answers, response (via `one_line`), correct/incorrect, the shown-positions line (`_shown_line`), and the replay check.
  - `_shown_line(result)` — "Shown response tokens are positions a-b; position b is the answer token (response token k of n)."
  - `one_line(text)` — newlines shown as ⏎ (as the heatmap shows them) so the response stays one markdown line.
  - `item_card(item, field)` — markdown card for the chosen example before a run: name, category, the field's prompt, bridge (via `bridge_label`), swap bridge, expected answers.
  - `bridge_label(item)` — the bridge followed by its `bridge_zh` spellings in parentheses, e.g. `France (法国, 法兰西)`; just `France` when `bridge_zh` is empty.

## App (`app.py`, `run_app.py`)
Launch from the repo root: `python -m viz.run_app` (flags: `--config`, `--model`, `--device`, `--viz_config`), then open `http://127.0.0.1:7860` (`server_name` / `server_port` in `configs/viz.yaml`). The first run in a process downloads d3 from a CDN (jlens inlines it into the page), so it needs internet once. Needs `gradio==6.32.0` (in `environment.yml`).

At startup it prints the dictionary line (see `load_dictionary`), or stops if `dictionary_file` is set but the file is missing.

UI flow: on page open the default model loads (the example list needs its tokenizer, so this happens on `demo.load`, with Gradio's progress indicator on the summary), then the dataset's splits, the split's examples, the first example's valid fields and conditions, and its card are filled in. Changing a dropdown re-runs its own step and every later one (model -> dataset -> split -> example -> card); **Random example** picks an example and continues from the example step. **Run** generates and shows the summary and the heatmap. All events that touch the model share one queue slot (`MODEL_QUEUE`), so two tabs never load or run at once.

Reading the result: generation still runs the full `max_new_tokens`, and the summary's response line and correct/incorrect use the full response (as eval does), but the heatmap stops at the answer token: the first generated token that, decoded, is not whitespace, a bare quote, EOS or a filler word from `evaluation.scoring.FILLER_PREFIXES` (" a", " the", " called", ...); if none qualifies it is token 0. The summary names its position. jlens pins at most 8 ids on page load (lowest ids first); with several English variants, answers and Chinese spellings some may start unpinned, but they are still tracked and can be pinned by clicking. The summary's last line is the replay check: whether the final layer's top-1 at each position, from the single forward over prompt + shown response, reproduces the generated token. On GPU bf16 a near-tie can flip between generation (one growing sequence per step) and the replay (one full-length forward), so some positions may be listed; the heatmap then shows the replay's numbers at those positions. Rows at ablated layers show the post-edit residual, so under `bridge` the bridge is gone there by construction. Adapters are read with the base model's prefit lens (no refit lens per adapter). The page logs a harmless `history.replaceState` SecurityError in the browser console: jlens saves its view state in the page URL, which a `srcdoc` iframe cannot do; rendering and clicks are unaffected.

- `app.py` — Gradio wiring only (no model or tensor code).
  - `AppContext` — `run` (RunConfig), `viz` (VizConfig), `eval_cfg` (EvalConfig), `session` (the only holder of the loaded model; never put in `gr.State`), `choices` (label -> ModelChoice), `default_model` (a label), `rng` (for Random example), `dictionary` (Cedict, or None = no translation box).
  - `MODEL_QUEUE` — shared `concurrency_id` / `concurrency_limit=1` for every model-touching event.
  - `on_model(ctx, label)` — load the model via `session.get`; returns a "Loaded" note and clears the old page.
  - `on_dataset(ctx, label, dataset)` — split dropdown: `split_names` of the dataset's readable items, reset to `all`.
  - `on_split(ctx, label, dataset, split)` — the split's examples as `{name: TwoHopItem}` (kept in a per-tab `gr.State` so later events don't re-read the dataset) and the example dropdown, first selected.
  - `on_example(ctx, items, name, field, condition)` — field and ablation dropdowns limited to `valid_fields` / `valid_conditions`; the current choice is kept if still valid.
  - `on_card(items, name, field)` — the example card (`page.item_card`).
  - `on_random(ctx, items)` — example dropdown set to `pick_random`.
  - `on_run(ctx, label, items, name, field, condition)` — `run_example` -> `run_slice` -> `replay_mismatches` -> `_page` -> `(summary, iframe)`; `gr.Error` only if no example/field/ablation is selected.
  - `_page(ctx, slice_data, result, title)` — `slice_page`, wrapped in `with_glosses(page, gloss_payload(...))` when `ctx.dictionary` is loaded.
  - `build_app(ctx)` — the `gr.Blocks`: `_controls` lays out the components, `_wire` connects them.
  - `_steps` / `_cascade` / `_tail` / `_chain` — the refresh chain as an ordered list of event kwargs (each with an `api_name`); a trigger (`gr.on`) runs from its step to the end with `.success`, so a failed step stops the chain. Only user `.input` events start chains, never `.change`.
  - Embedding: `gr.HTML(js_on_load=None, apply_default_css=False)` holding `page.iframe`'s `<iframe srcdoc>`. Gradio 6 renders `value` raw through `innerHTML` (no sanitising); the `${value}` template inserts it as a JS variable, so `${...}` and backticks inside the page are never re-evaluated. The iframe's inline height sets its size.
- `run_app.py` — entry point.
  - `parse_args(argv=None)` — `model.cli.add_profile_args` (`--config --model --device`) plus `--viz_config`.
  - `build_context(args)` — `resolve_profile`, `load_viz_config`, `model_choices` keyed by label, default = the base choice of `--model` (or run.yaml's model; must have a prefit lens), `EvalConfig()` defaults (debt-002), `Session(make_loader(run, args.device))`, `random.Random(eval_cfg.seed)`, `load_dictionary(viz.dictionary_file)`.
  - `load_dictionary(path)` — `load_cedict(path)` and prints `CC-CEDICT: <entries> entries, <headwords> headwords <- <path>`; `None` prints `Translations off (dictionary_file: null)` and returns None.
  - `main()` — `build_app(build_context(...)).launch(server_name, server_port, show_error=True)`.
