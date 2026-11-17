# jspace

Our J-space tools, built on Anthropic's official [jacobian-lens](https://github.com/anthropics/jacobian-lens) package (`jlens`).
We don't reimplement the lens. We load the prefit lens named in the model profile (Qwen3-1.7B: the one Neuronpedia fit with `jlens.fit`), and add only what the paper describes but the repo doesn't ship: J-lens vectors, ablation, and swap.

This module does not load models or register hooks. It takes a `model.loading.LanguageModel` (loaded by `model.loading.load_model`), and its interventions are plain `model.hooks.Intervention`s passed to `model.inference` functions or `model.hooks.apply_interventions`.
Used by `evaluation/` and `finetune/`.

## Install
`jlens` is pinned in `environment.yml` (`pip install "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"`).

## How it maps to the paper
| paper | here |
|---|---|
| `J_l = E[dh_final / dh_l]` (fit over wikitext, 128 tokens) | the profile's `prefit_lens` (Qwen3-1.7B: `neuronpedia/jacobian-lens`, `qwen3-1.7b/.../Qwen3-1.7B_jacobian_lens.pt`) |
| "rows of `W_U J_l` are the J-lens vectors" | `vectors.lens_vectors` |
| lens readout `W_U norm(J_l h)` | official `lens.apply`, wrapped by `readout.lens_logits` / `readout.token_ranks`; `readout.residual_logits` reads residuals you already captured |
| ablation: "at each token position, across a band of layers, we identify the k=10 most strongly activated J-lens vectors and zero out the residual stream's projection onto each" (skipping tokens in the clean pass's top 10) | `interventions.TopKAblation` |
| light / medium / heavy ablation "differ in the range of layers" | `config.ablation_strengths`: first third, two thirds, or all of the band |
| swap: `c = pinv(V) h`, `h + V(swap(c) - c)` with `V = [v_s v_t]` | `interventions.LensSwap` |
| workspace band ~L38 to ~L92 of 100 | `config.band_start/band_end` (the paper gives no Qwen band; we reuse the fractions) |
| our proposal: ablate the bridge's J-space component | `interventions.ProjectOut` over all one-token spellings of the bridge |
| (ours) lower the bridge readout only where it is abnormally high, instead of pinning it at 0 | `clamp.ReadoutClamp` with caps from `baseline.ReadoutStats` |

## Files
- `config.py`: `JSpaceConfig`: band fractions, top-k ablation settings, leading positions never edited, readout clamp settings (`clamp_z`: cap = mean + z * std, default 2; `readout_stats_dir`: folder of `<profile>.pt` statistics, default `stats/`; `stats_batch_size`: texts per pass when collecting them, default 4), decomposition k, lens-fitting settings. Model identity and lens location live in the model profile.
- `loading.py` (all file and network access)
  - `JSpace(lm, lens_model, lens, band, cfg, readout_stats=None)`: the shared `LanguageModel`, a jlens view of it, the lens, the band layers, the config, and the reference readout statistics (`ReadoutStats`, needed by `ReadoutClamp`) or `None`.
  - `lens_view(lm)`: `jlens.from_hf` on `lm.base_model`, then restores every parameter's `requires_grad` and every module's train/eval mode (from_hf freezes and calls `.eval()`). Safe before or after `attach_lora`. The view shares `lm.blocks`, so our hooks fire inside `lens.apply` and `jlens.fit`. Never compiled.
  - `load_lens(lm, lens_path=None)`: the `.pt` at `lens_path`, else the profile's `prefit_lens` from the Hub; J matrices moved to `lm.device`. Raises if neither exists, or if the lens's `d_model`, layers or matrix shapes do not fit `lm`.
  - `load_jspace(lm, cfg=JSpaceConfig(), lens_path=None, stats_path=None)`: bundles the above into a `JSpace`. Readout stats: an explicit `stats_path` is loaded (a missing file raises); with `None`, the default file is loaded if it exists, else `readout_stats` is `None`.
  - `default_stats_path(lm, cfg)`: `cfg.readout_stats_dir / "<profile name>.pt"`. Keyed by profile, so an adapter run uses its base model's statistics.
- `vectors.py`
  - `workspace_band(n_layers, start, end)`: layer indices in the band.
  - `band_prefix(band, fraction)`: the first part of the band (for ablation strengths).
  - `token_variants(tokenizer, word)`: ids of every single-token spelling (space or no space, case); builds the variants and delegates to `single_token_ids`.
  - `single_token_ids(tokenizer, spellings)`: ids of the given spellings that tokenize to exactly one token, first-seen order, deduplicated; no case or space variants added (used for the Chinese bridge spellings).
  - `lens_vectors(lens, W_U, ids, layers)`: `{layer: [k, d]}` rows of `W_U J_l`.
  - `lens_activations(lens, W_U, h, layer)`: activation of every J-lens vector on `h`, `[..., vocab]`.
- `interventions.py`: J-space edits as `model.hooks.Intervention`s. Each edits every real token of a row except its first `cfg.ablation_skip_first`; left padding is never edited.
  - `LayerIntervention(js, layers)`: base class. `__call__(resid, ctx)` runs `edit` in fp32 on the residual from the first editable column onward, casts back to the residual's dtype, and keeps the original values at padding and leading positions. Subclasses implement `edit(h, layer)`.
  - `editable_positions(attention_mask, skip_first)`: `[batch, seq]` bool of positions that may be edited. `first_editable_column(keep)`: first column any row may edit.
  - `ProjectOut(js, {layer: [k, d]})`: removes the span of fixed vectors (joint orthogonal projection).
  - `TopKAblation(js, layers, k, exclude_mask)`: the paper's ablation; `exclude_mask` is `[batch, seq, vocab]`, aligned with the full sequence.
  - `LensSwap(js, source, target, alpha=1)`: the paper's swap formula.
- `baseline.py`: reference statistics of the lens readout (no file paths or dataset knowledge; texts come in as strings).
  - `ReadoutStats(layers, mean, std, count, model, n_texts)` (frozen): `mean`, `std` `[len(layers), vocab]` fp32 on CPU, `count` `[vocab]` positions per token, `model` the profile name. `caps(token_ids, layers, z)`: `{layer: [k]}` `mean + z * std`; raises `ValueError` for a layer without statistics or a token with count 0.
  - `collect_readout_stats(js, texts, exclude_ids, batch_size)`: clean forward passes in batches; at each band layer, accumulates (fp64) the lens logit of every token over editable positions (`editable_positions`, so the first token and padding never count), skipping `exclude_ids[i]` for text `i`. `std = sqrt(E[x^2] - mean^2)`. Helpers `_exclusion_mask`, `_accumulate`, `_finish`.
  - `save_readout_stats(stats, path)`: `torch.save` of the fields (creates the folder). `load_readout_stats(path, lm, band)`: raises `ValueError` if the file is for another profile, another vocab size, or lacks a band layer.
- `clamp.py`: why it exists: `ProjectOut` sets the bridge readout to exactly 0, a value never seen in clean runs, and a model fine-tuned under it learned to exploit that signal. The clamp only lowers readouts above `mean + z * std`; every other position stays bit-identical.
  - `set_readout(h, V, V_pinv, targets)`: minimum-norm edit `h + (targets - h V^T) pinv(V)^T` giving `h V^T = targets` (zero targets = `ProjectOut`).
  - `ReadoutClamp(js, vectors, caps)`: `LayerIntervention`; `vectors` `{layer: [k, d]}`, `caps` `{layer: [k]}` (e.g. `ReadoutStats.caps`). `edit` sets each readout to `min(readout, cap)`; the part of `h` orthogonal to the vectors is untouched.
- `readout.py`
  - `lens_logits(js, prompt, interventions=(), layers=None, positions=None)`: `{layer: [n_positions, vocab]}` from `lens.apply`, with the interventions applied (an all-ones mask shaped like jlens's encoding of the prompt).
  - `token_ranks(js, prompt, ids, position=-1, layers=None, interventions=())`: best rank of those tokens in the J-lens readout, per layer.
  - `readout_layers(js)`: the fitted layers plus the model's final layer (the layers `residual_logits` accepts).
  - `residual_logits(js, resid, layer)`: fp32 `[..., vocab]` J-lens logits for residuals `[..., d]` you already have (e.g. from `model.inference.forward_residuals`): `lens.transport` if the layer has a Jacobian, then `lens_model.unembed` (casts to the head's dtype). The final layer needs no Jacobian, so its row equals the model's output logits; any other unfitted layer raises `ValueError`. Matches `jlens.vis.compute_slice`.
- `decompose.py`: the paper's J-space as a sparse nonnegative combination of at most 25 J-lens vectors, found by gradient pursuit.
  - `jspace_decompose(lens, W_U, h, layer, k)`: returns the J-space part of `h`, the chosen token ids, and their coefficients.
  - `non_jspace_part(lens, W_U, h, layer, k)`: `h` minus its J-space part (used by the probes in `evaluation/`).
- `fit_lens.py`: CLI that fits a new lens with the official `jlens.fit` on WikiText (paper settings: 128 tokens, up to 1000 prompts) on the profile's model, optionally with an unmerged LoRA adapter. This is the "recomputed J-lens" from the proposal, and the way to get a lens for a model without a prefit one (Qwen3-0.6B). Takes `--config`, `--model`, `--device`, `--adapter`, `--out`, `--n_prompts`, `--dim_batch`.
  ```
  python -m jspace.fit_lens --adapter path/to/adapter --out lenses/recomputed.pt
  ```

## Usage
```python
from model.config import load_model_profile, load_run_config
from model.loading import load_model
from model.inference import next_token_logits
from jspace import load_jspace, lens_vectors, token_variants, ProjectOut, ReadoutClamp

run = load_run_config("configs/run.yaml")
js = load_jspace(load_model(load_model_profile(run.model, run.profiles_dir)))
ids = token_variants(js.lm.tokenizer, "spider")
ablate = ProjectOut(js, lens_vectors(js.lens, js.lm.unembedding_matrix, ids, js.band))
logits = next_token_logits(js.lm, ["Fact: The number of legs on the animal that spins webs is"], [ablate])

# clamp instead of zeroing (needs stats/<profile>.pt, or load_jspace(..., stats_path=...))
vecs = lens_vectors(js.lens, js.lm.unembedding_matrix, ids, js.band)
clamp = ReadoutClamp(js, vecs, js.readout_stats.caps(ids, js.band, js.cfg.clamp_z))
```

## Open choices (the paper doesn't specify these)
- `TopKAblation`: "most strongly activated" is the J-lens vector activation `W_U[v] J_l h`. The clean top 10 are excluded per position.
- Vectors are not normalized, because ProjectOut and TopKAblation build an orthonormal basis anyway.
- Gradient pursuit selects atoms by raw (not norm-scaled) correlation with the residual.
- Model is `Qwen/Qwen3-1.7B` (the lens was fit on it), prompted with raw `Fact: ...` text and no chat template, so thinking mode never activates.
- The first token of each row is never edited. Qwen's first position is an attention sink with huge activations, and the official fitting code also skips early positions. Editing it breaks the model outright (output like `' tại tại вот'`), whereas the paper reports that ablation leaves fluency intact.
- jlens tokenises prompts itself inside `lens.apply` and `jlens.fit` (debt 001). For Qwen it equals `lm.token_ids` (no BOS token; a test checks it).
- Fitting needs a GPU for a real lens: one prompt takes about 3 minutes on an M-series Mac with `--dim_batch 8`, and `--dim_batch 64` runs out of memory there.
