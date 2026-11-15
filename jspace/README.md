# jspace

Our J-space tools, built on Anthropic's official [jacobian-lens](https://github.com/anthropics/jacobian-lens) package (`jlens`).
We don't reimplement the lens. We load the prefit Qwen3-1.7B lens that Neuronpedia fit with `jlens.fit`, and add only what the paper describes but the repo doesn't ship: J-lens vectors, ablation, and swap.

Used by `evaluation/`, and later by training (ablation during LoRA).

## Install
```
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens" peft
```

## How it maps to the paper
| paper | here |
|---|---|
| `J_l = E[dh_final / dh_l]` (fit over wikitext, 128 tokens) | prefit lens from `neuronpedia/jacobian-lens`, `qwen3-1.7b/.../Qwen3-1.7B_jacobian_lens.pt` |
| "rows of `W_U J_l` are the J-lens vectors" | `vectors.lens_vectors` |
| lens readout `W_U norm(J_l h)` | official `lens.apply`, wrapped by `readout.token_ranks` |
| ablation: "at each token position, across a band of layers, we identify the k=10 most strongly activated J-lens vectors and zero out the residual stream's projection onto each" (skipping tokens in the clean pass's top 10) | `interventions.TopKAblation` |
| swap: `c = pinv(V) h`, `h + V(swap(c) - c)` with `V = [v_s v_t]` | `interventions.LensSwap` |
| workspace band ~L38 to ~L92 of 100 | `config.band_start/band_end` (the paper gives no Qwen band; we reuse the fractions) |
| our proposal: ablate the bridge's J-space component | `interventions.ProjectOut` over all one-token spellings of the bridge |

## Files
- `config.py`: `JSpaceConfig`: model name, lens location (Hub or a local `lens_path`), band fractions, top-k settings.
- `loading.py`
  - `JSpace`: bundle of the HF model, jlens `LensModel`, tokenizer, lens, `W_U`, band layers, config.
  - `load_jspace(cfg, adapter_path=None)`: loads the model, merges a LoRA adapter if given (for Mctrl or Mexp), and loads the lens.
  - `load_lens(cfg)`: local `.pt` if `lens_path` is set, otherwise the Hub.
- `vectors.py`
  - `workspace_band(n_layers, start, end)`: layer indices in the band.
  - `token_variants(tokenizer, word)`: ids of every single-token spelling (space or no space, case).
  - `lens_vectors(lens, W_U, ids, layers)`: `{layer: [k, d]}` rows of `W_U J_l`.
  - `lens_activations(lens, W_U, h, layer)`: activation of every J-lens vector on `h`, `[..., vocab]`.
- `interventions.py`: context managers that hook the decoder layers and edit the residual at every position.
  - `LayerIntervention`: base class (hook plumbing). Subclasses implement `edit(h, layer)`.
  - `ProjectOut(js, {layer: [k, d]})`: removes the span of fixed vectors (joint orthogonal projection).
  - `TopKAblation(js, layers, k, exclude_mask)`: the paper's ablation (batch 1).
  - `LensSwap(js, source, target, alpha=1)`: the paper's swap formula.
- `readout.py`
  - `token_ranks(js, prompt, ids, position=-1)`: best rank of those tokens in the J-lens readout, per layer. Works inside an active intervention.

## Usage
```python
from jspace import load_jspace, lens_vectors, token_variants, ProjectOut
js = load_jspace()
ids = token_variants(js.tokenizer, "spider")
with ProjectOut(js, lens_vectors(js.lens, js.W_U, ids, js.band)):
    logits = js.hf_model(input_ids=js.model.encode("Fact: The number of legs on the animal that spins webs is")).logits
```

## Open choices (the paper doesn't specify these)
- `TopKAblation`: "most strongly activated" is the J-lens vector activation `W_U[v] J_l h`. The clean top 10 are excluded per position.
- Vectors are not normalized, because ProjectOut and TopKAblation build an orthonormal basis anyway.
- Model is `Qwen/Qwen3-1.7B` (the lens was fit on it), prompted with raw `Fact: ...` text and no chat template, so thinking mode never activates.
- Not tested on a real model yet: the code passes a syntax check only.
