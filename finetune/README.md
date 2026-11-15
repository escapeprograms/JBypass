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

```
python -m evaluation.make_splits --out results/splits.json
python -m finetune.run_train --condition bridge --split_file results/splits.json --out runs/mexp
python -m evaluation.run_eval --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
```
Steps, batch size, learning rate, LoRA rank and the shuffling seed come from `configs/run.yaml` (`--config` for another file). `--model`, `--device` and `--lens_path` work as in the eval scripts.

## Files
- `config.py`
  - `TRAIN_CONDITIONS`: the eval conditions training may use (argparse `choices` of `--condition`). Top-k ablation is not enabled until it has a training test.
  - `FinetuneConfig`: dataset name, split to train on (`train`), name of the info file written beside the adapter.
- `ablated.py`
  - `item_text(lm, item)`: `(item.prompt, " " + item.answers[0])`, the raw `Fact: ...` prompt the lens was fit on and eval scores. Passed to `model.training.train` as `to_text`.
  - `condition_factory(js, condition, seed)`: an `InterventionFactory` that, per batch, calls the condition's builder for each row's item (seeded by `evaluation.conditions.item_rng(seed, item, condition)`, the same seed eval uses) and wraps the results in one `model.hooks.PerRow`.
- `run_train.py`: `main()`: resolve profile → `load_model` → `load_jspace` → `attach_lora` → the split's items → `readable_items` → `train` with `condition_factory` → `save_adapter` plus `train_info.json` (condition, seed, ablation seed, split file, split, per-step losses). Prints each step's loss and the peak VRAM. `parse_args()` holds the flags.

## Plug-in points
- Top-k ablation during training: add its names to `TRAIN_CONDITIONS` once a test covers it (each builder already gets its row's tokens).
- Periodic lens refits: an `on_step` callback that calls `jlens.fit(js.lens_model, ...)` and swaps `js.lens`.
