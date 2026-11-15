# Commands

Every command runs from the repo root in the `jbypass` env. On Windows, without activating the env:
```powershell
$py = "C:\Users\archi\miniconda3\envs\jbypass\python.exe"   # then use `& $py -m ...` in place of `python -m ...`
```
With the env active (`conda activate jbypass`), plain `python -m ...` works.

**Flags every command takes:** `--config configs/run.yaml`, `--model <profile>` (`qwen3-1.7b` default, or `qwen3-0.6b`), `--device <cuda|cpu|mps>`.
**Flags every eval command also takes:** `--adapter runs/<name>` (evaluate a fine-tuned model instead of base M), `--lens_path <file.pt>` (ablate with a different lens), `--limit N` (first N items, for quick runs).

## 1. Setup (once)
```powershell
conda env create -f environment.yml
conda activate jbypass
pip install -e .
python -m model.run_download                    # weights for the model in configs/run.yaml
python -m model.run_download --model qwen3-0.6b # another profile
```

## 2. Checks
```powershell
python -m pytest tests                          # unit tests, CPU, tiny random models
python -m model.run_smoke                       # GPU end-to-end check of loading, hooks, LoRA
python -m evaluation.run_lens_checks --tag M    # does the lens behave like the paper? -> results/lens_checks_M.json
```

## 3. Data and splits
```powershell
python -m evaluation.data.build_jbypass_dataset            # rebuild evaluation/data/jbypass_two_hop.json (already committed)
python -m evaluation.make_splits --out results/splits.json # items M solves, split by bridge into train / test_seen / test_unseen
```
`--unseen_frac 0.3` sets the share of bridges held out for `test_unseen`.

## 4. Baselines on base model M
```powershell
python -m evaluation.run_eval --dataset jbypass --tag M                           # all conditions, all prompt types
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt  # Anthropic's dataset
python -m evaluation.run_mechanism --tag M                                        # bridge readout by position + linear probes
python -m evaluation.run_perplexity --tag M                                       # held-out perplexity
```
Quick first try: add `--limit 10 --conditions none,bridge`.

## 5. Train
**Set `train.steps` in `configs/run.yaml` first.** It is 4 (smoke-test length). Steps, batch size, learning rate, LoRA rank and seed all come from that file.
```powershell
python -m finetune.run_train --condition bridge     --split_file results/splits.json --out runs/mexp    # Mexp
python -m finetune.run_train --condition none       --split_file results/splits.json --out runs/mctrl   # Mctrl
python -m finetune.run_train --condition random_tok --split_file results/splits.json --out runs/mrand   # Mrand (or random_dir)
```
Each run saves the LoRA adapter plus `train_info.json` (per-step losses) in `--out`.

## 6. Evaluate fine-tuned models
Run each block with `--split test_seen` and again with `--split test_unseen`, for each adapter (`mexp`, `mctrl`, `mrand`):
```powershell
python -m evaluation.run_eval      --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_mechanism --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_perplexity --adapter runs/mexp --tag mexp
```

## 7. Recompute the lens after training
```powershell
python -m jspace.fit_lens --adapter runs/mexp --out lenses/mexp.pt                # full fit
python -m jspace.fit_lens --adapter runs/mexp --n_prompts 100 --out lenses/quick.pt  # quick fit
```
Compare it with the original lens side by side, or ablate with it:
```powershell
python -m evaluation.run_mechanism --adapter runs/mexp --extra_lens recomputed=lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_eval      --adapter runs/mexp --lens_path lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp_relens
```

## Outputs
All eval results go to `results/`, named by `--tag`:
| command | file |
|---|---|
| `run_eval` | `eval_<tag>_<dataset>_<split or all>.json` |
| `run_mechanism` | `mechanism_<tag>_<dataset>_<split or all>.json` |
| `run_perplexity` | `perplexity_<tag>.json` |
| `run_lens_checks` | `lens_checks_<tag>.json` |

Conditions for `--conditions` (eval): `none, bridge, wrong_bridge, random_tok, random_dir, topk_light, topk_medium, topk_heavy`. Training `--condition` takes `none, bridge, random_tok, random_dir`.
