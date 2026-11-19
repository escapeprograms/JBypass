# finetune

LoRA fine-tuning under an eval ablation condition: the experiment itself. It composes the
other modules: the model and training loop from `model/`, the lens from `jspace/`, and the
dataset, splits and condition builders from `evaluation/`. The saved adapter is what every
eval script loads with `--adapter`.

| `--condition` | model in the proposal |
|---|---|
| `none` | Mctrl (fine-tuned, no ablation) |
| `bridge` | Mexp (fine-tuned with the bridge's J-lens vectors projected out over the band) |
| `random_tok`, `random_dir` | Mrand (fine-tuned under a random ablation of the same size) |
| `bridge_clamp` | the clamp experiment: Mexp with the bridge readout capped at its reference level mu + z*sigma instead of zeroed, so the model cannot key on an exactly-zero readout. Needs readout statistics (`python -m evaluation.run_readout_stats`, or `--stats_path`) |

```
python -m evaluation.make_splits --out results/splits.json
python -m finetune.run_train --condition bridge --split_file results/splits.json --out runs/mexp
python -m evaluation.run_eval --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_readout_stats                     # once per profile, before bridge_clamp
python -m finetune.run_train --condition bridge_clamp --split_file results/splits.json --out runs/mclamp
```
Steps, batch size, learning rate, LoRA rank and the shuffling seed come from `configs/run.yaml` (`--config` for another file); so do the band, `clamp_z` and the readout statistics folder (its optional `jspace:` section, e.g. `--config configs/exp3.yaml` for layers 8-26 with `stats/exp3/`). `--model`, `--device`, `--lens_path` and `--stats_path` work as in the eval scripts.

## Files
- `config.py`
  - `TRAIN_CONDITIONS`: the eval conditions training may use (argparse `choices` of `--condition`): `none`, `bridge`, `random_tok`, `random_dir`, `bridge_clamp`. Top-k ablation is not enabled until it has a training test.
  - `FinetuneConfig`: dataset name, split to train on (`train`), name of the info file written beside the adapter. Still dataclass defaults (`TODO(debt-002)`).
- `ablated.py`
  - `item_text(lm, item)`: `(item.prompt, " " + item.answers[0])`, the raw `Fact: ...` prompt the lens was fit on and eval scores. Passed to `model.training.train` as `to_text`.
  - `condition_factory(js, condition, seed)`: an `InterventionFactory` that, per batch, calls the condition's builder for each row's item (seeded by `evaluation.conditions.item_rng(seed, item, condition)`, the same seed eval uses) and wraps the results in one `model.hooks.PerRow`.
- `run_train.py`: `main()`: resolve profile → `load_model` → `load_jspace` with `jspace_config(run)` (lens, band and readout stats) → `attach_lora` → the split's items → `readable_items` → `train` with `condition_factory` → `save_adapter` plus `train_info.json` (condition, seed, ablation seed, split file, split, `stats_path` actually loaded or null, `clamp_z`, `band` (the layer indices trained under), per-step losses). Prints each step's loss and the peak VRAM. `parse_args()` holds the flags.
  - `used_stats_path(js, stats_path) -> str | None`: the stats file `js` was loaded with (`--stats_path`, else the default path), `None` if it has no stats.

## Plug-in points
- Top-k ablation during training: add its names to `TRAIN_CONDITIONS` once a test covers it (each builder already gets its row's tokens).
- Periodic lens refits: an `on_step` callback that calls `jlens.fit(js.lens_model, ...)` and swaps `js.lens`.
