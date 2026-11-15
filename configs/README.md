# configs

- `run.yaml` — the run: which model profile (`model`), where profiles live (`profiles_dir`, relative to this folder), and generation, LoRA and training settings. Memory-bound values (batch size) live here.
- `models/<name>.yaml` — one model profile per file; the only place model-specific values live (HF id, dtype, device, block and final-norm paths, chat-template kwargs, LoRA target modules). The file name must equal its `name`.
  - `qwen3-1.7b.yaml` — default model.
  - `qwen3-0.6b.yaml` — second profile, proves switching is config-only.
