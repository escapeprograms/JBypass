# Commands

Every command runs from the repo root in the `jbypass` env. On Windows, without activating the env:
```powershell
$py = "C:\Users\archi\miniconda3\envs\jbypass\python.exe"   # then use `& $py -m ...` in place of `python -m ...`
```
With the env active (`conda activate jbypass`), plain `python -m ...` works. Add `--help` to any command to list its flags.

**Model names used below:** M is the base model with no fine-tuning. Mexp is fine-tuned while the bridge is ablated (the experiment). Mctrl is fine-tuned with no ablation (control). Mrand is fine-tuned under a random ablation of the same size (control).

## Shared flags

**Every command** takes these:
| flag | default | what it does |
|---|---|---|
| `--config <path>` | `configs/run.yaml` | Run settings file: which model, generation length, LoRA rank, training steps, batch size, learning rate, seed. |
| `--model <name>` | `model:` in run.yaml (`qwen3-1.7b`) | Model profile from `configs/models/<name>.yaml`. Overrides run.yaml. Options: `qwen3-1.7b`, `qwen3-0.6b`. |
| `--device <name>` | the profile's `device` (`cuda`) | Where to run: `cuda`, `cpu` or `mps` (Mac). |

**Every eval command** (`evaluation.*`) also takes these:
| flag | default | what it does |
|---|---|---|
| `--tag <name>` | `M` | A label for the run. It is printed in the log and used in the output filename (`results/eval_<tag>_...json`), so runs don't overwrite each other. Use `M` for the base model and the adapter's name for a fine-tuned one (`mexp`, `mctrl`, ...). It does not change what gets computed. |
| `--adapter <dir>` | none (base model M) | Loads a LoRA adapter saved by `finetune.run_train`, so you evaluate Mexp, Mctrl or Mrand instead of M. |
| `--lens_path <file.pt>` | the profile's prefit lens (downloaded from Neuronpedia) | Uses a different J-lens for the ablations and readouts, e.g. one refit with `jspace.fit_lens`. |
| `--limit <N>` | all items | Uses only the first N items of each dataset. For quick test runs. |

**Splits flags** (`run_eval`, `run_mechanism`):
| flag | default | what it does |
|---|---|---|
| `--split_file <path>` | none (whole dataset) | JSON file written by `evaluation.make_splits`. |
| `--split <name>` | none | Which part of the split file to use: `train`, `test_seen` (new questions about bridges seen in training) or `test_unseen` (bridges never seen in training). Needs `--split_file`. |

---

## 1. Setup (once)
```powershell
conda env create -f environment.yml
conda activate jbypass
pip install -e .
python -m model.run_download                    # downloads the model named in configs/run.yaml
python -m model.run_download --model qwen3-0.6b # downloads another profile
```
`model.run_download` saves the weights and tokenizer to the Hugging Face cache. It takes only the shared flags.

## 2. Checks
```powershell
python -m pytest tests                          # unit tests on CPU with tiny random models; no download needed
python -m model.run_smoke                       # end-to-end GPU check: generation, hooks, residual capture, a few LoRA steps
python -m evaluation.run_lens_checks --tag M    # does the lens reproduce the paper's results on this model?
```
- `run_smoke` takes only the shared flags. It stops with an error at the first check that fails.
- `run_lens_checks` reports the bridge's lens readout (pass@k), clean and with the bridge ablated, plus the paper's swap check. It writes `results/lens_checks_<tag>.json`.

## 3. Data and splits
```powershell
python -m evaluation.data.build_jbypass_dataset               # rebuilds evaluation/data/jbypass_two_hop.json (already in the repo)
python -m evaluation.make_splits --out results/splits.json    # train / test splits
```
`make_splits` keeps the jbypass items that M answers correctly on all three prompt types, then splits them by bridge.
| flag | default | what it does |
|---|---|---|
| `--out <path>` | `results/splits.json` | Where to write the split file. |
| `--unseen_frac <float>` | `0.3` | Share of bridges held out entirely for `test_unseen`. The remaining bridges' items are split into `train` and `test_seen`. |

It also takes the eval flags (`--adapter`, `--lens_path`, `--limit`). Normally you run it on base M with no extra flags.

## 4. Baselines on base model M
```powershell
python -m evaluation.run_eval --dataset jbypass --tag M
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt
python -m evaluation.run_mechanism --tag M
python -m evaluation.run_perplexity --tag M
```
Quick first try: add `--limit 10 --conditions none,bridge`.

### `evaluation.run_eval`: accuracy under each ablation
Prints and saves a table of accuracy for each condition and prompt type. Output: `results/eval_<tag>_<dataset>_<split or all>.json`.
| flag | default | what it does |
|---|---|---|
| `--dataset <name>` | `jbypass` | `jbypass` (our 1032 items), `paper_probe_swap` or `paper_multihop` (Anthropic's sets). |
| `--conditions <a,b,...>` | all eight | Comma-separated ablations to run, with no spaces (see the table below). |
| `--fields <a,b,...>` | `prompt,hop2_prompt,hop1_prompt` | Which prompt types to score. `prompt` is the two-hop question (main metric). `hop2_prompt` gives the bridge and asks for the answer (fair one-hop control). `hop1_prompt` asks for the bridge itself. Anthropic's datasets only have `prompt`, so pass `--fields prompt` for them. |
| `--all_items` | off | Keeps items that M gets wrong with no ablation. By default those are dropped, so accuracy measures what the ablation breaks. |

It also takes the shared, eval and split flags.

**Ablation conditions** (`--conditions` values):
| name | what it removes |
|---|---|
| `none` | nothing (clean accuracy) |
| `bridge` | the J-lens vectors of the item's bridge, over the ablation band (the intervention) |
| `wrong_bridge` | the same, for a different bridge of the same kind (checks that the ablation is specific) |
| `random_tok` | J-lens vectors of the same number of random word tokens |
| `random_dir` | the same number of random Gaussian directions |
| `topk_light`, `topk_medium`, `topk_heavy` | the paper's ablation: the top 10 active J-lens vectors at each position, over the first third, two thirds, or all of the band |

### `evaluation.show_table`: reprint saved eval tables
```powershell
python -m evaluation.show_table results/eval_M_jbypass_all.json "results/eval_mexp_*.json"
```
Prints the exact-match and next-token tables of saved `run_eval` results, rebuilt from their per-item records, without loading a model. Takes one or more files or glob patterns (quote patterns; the script expands them). No other flags. Writes nothing.

### `evaluation.run_mechanism`: how the model gets around the ablation
For each condition it saves the bridge's J-lens rank at each layer and position, plus the accuracy of linear probes that predict the bridge from the residual stream. Output: `results/mechanism_<tag>_<dataset>_<split or all>.json`.
| flag | default | what it does |
|---|---|---|
| `--dataset <name>` | `jbypass` | As in `run_eval`. |
| `--conditions <a,b,...>` | `none,bridge` | Conditions to analyze under. |
| `--extra_lens <label>=<path>` | none | Also runs the analysis with another lens and saves it under `<label>`, next to the original lens's results. You can repeat it, e.g. `--extra_lens recomputed=lenses/mexp.pt`. |

It also takes the shared, eval and split flags.

### `evaluation.run_perplexity`: did fine-tuning damage general language modeling?
Perplexity on held-out WikiText. Output: `results/perplexity_<tag>.json`. Takes only the shared and eval flags. `--limit` sets the number of texts, and `--lens_path` is accepted but ignored.

## 5. Train
**Set `train.steps` in `configs/run.yaml` first.** It is 4 (smoke-test length). Steps, batch size, learning rate, LoRA rank and seed all come from that file. To try settings without editing it, copy it and pass `--config <copy>`.
```powershell
python -m finetune.run_train --condition bridge     --split_file results/splits.json --out runs/mexp    # Mexp
python -m finetune.run_train --condition none       --split_file results/splits.json --out runs/mctrl   # Mctrl
python -m finetune.run_train --condition random_tok --split_file results/splits.json --out runs/mrand   # Mrand
```
`finetune.run_train` trains a LoRA adapter on the `train` split, with the chosen ablation applied during every forward pass.
| flag | default | what it does |
|---|---|---|
| `--condition <name>` | required | The ablation applied during training: `bridge` (Mexp), `none` (Mctrl), `random_tok` or `random_dir` (Mrand). |
| `--split_file <path>` | required | Split file from `make_splits`. Training always uses its `train` split. |
| `--out <dir>` | required | Folder to save the adapter in, along with `train_info.json` (condition, seeds, per-step losses). Pass this folder to the eval commands as `--adapter`. |
| `--lens_path <file.pt>` | the profile's prefit lens | The lens whose vectors are ablated during training. |

It also takes the shared flags. It prints the loss at every step and the peak GPU memory at the end.

## 6. Evaluate fine-tuned models
Same commands as section 4, plus `--adapter`, the split flags, and a `--tag` naming the model. Run each one for every adapter (`mexp`, `mctrl`, `mrand`) and for both `test_seen` and `test_unseen`:
```powershell
python -m evaluation.run_eval       --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_mechanism  --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_perplexity --adapter runs/mexp --tag mexp
```
The split name is already in the output filename, so `--tag mexp` is enough for both splits. `run_perplexity` has no split, so run it once per adapter.

## 7. Recompute the lens after training
```powershell
python -m jspace.fit_lens --adapter runs/mexp --out lenses/mexp.pt                    # full fit
python -m jspace.fit_lens --adapter runs/mexp --n_prompts 100 --out lenses/quick.pt   # quick, rougher fit
```
`jspace.fit_lens` fits a new J-lens with Anthropic's `jlens.fit` on WikiText prompts, the same corpus the prefit lens used.
| flag | default | what it does |
|---|---|---|
| `--out <path>` | required | Where to save the lens `.pt`. A checkpoint `<out>.ckpt` is written while it runs. |
| `--adapter <dir>` | none (base M) | Fits the lens on the fine-tuned model instead of M. |
| `--n_prompts <N>` | `1000` | Number of WikiText prompts to fit on. Fewer is faster and rougher. |
| `--dim_batch <N>` | `32` | Number of dimensions `jlens.fit` processes at once. Lower it if you run out of GPU memory. |

It also takes the shared flags.

Then compare the new lens with the original side by side, or ablate with it:
```powershell
python -m evaluation.run_mechanism --adapter runs/mexp --extra_lens recomputed=lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_eval      --adapter runs/mexp --lens_path lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp_relens
```
The second command uses a new tag (`mexp_relens`) so it doesn't overwrite the results from section 6.

## 8. Visualize
```powershell
python -m viz.run_app                                   # then open http://127.0.0.1:7860
python -m viz.run_app --device cpu                      # no GPU (slow)
```
`viz.run_app` starts a local Gradio app for looking at one example at a time: it generates under one ablation and shows the J-lens readout of every layer over the prompt and the response (jlens's heatmap page). The model loads when the page opens (a few seconds on GPU). The first run in each process downloads d3 from a CDN, so it needs internet once. An env created before gradio was added needs `pip install gradio==6.32.0` and `pip install -e .` (a fresh `conda env create -f environment.yml` already has it).
| flag | default | what it does |
|---|---|---|
| `--config <path>` | `configs/run.yaml` | Run settings; `generation.max_new_tokens` is the response length. |
| `--model <name>` | `model:` in run.yaml | The base model selected when the page opens. Must have a prefit lens (`qwen3-1.7b`). |
| `--device <name>` | the profile's `device` | Where every model choice runs. |
| `--viz_config <path>` | `configs/viz.yaml` | Viewer settings: adapters folder, split file, top-n, tracked-token cap, iframe height, host and port. |

What you can do in the UI:
- **Model**: the base model or any adapter in `runs/` trained from it (switching reloads; only one model is in memory). Adapters are read with the base model's prefit lens.
- **Dataset / Split / Example**: splits come from `results/splits_expanded.json` (`all` = no filter); only items whose bridge the lens can read are listed. **Random example** picks one.
- **Prompt field / Ablation**: only the fields the item has and the conditions that can be built for it (`wrong_bridge` needs a swap bridge). `none` is the clean run.
- **Run**: shows the response, whether it is correct, and the heatmap: click a cell for its top tokens, click a pin to see its rank over every (position, layer); the bridge and the answers are pinned. Rows at ablated layers show the residual after the ablation.

## Outputs
| command | writes |
|---|---|
| `make_splits` | `results/splits.json` (or `--out`) |
| `run_eval` | `results/eval_<tag>_<dataset>_<split or all>.json` |
| `run_mechanism` | `results/mechanism_<tag>_<dataset>_<split or all>.json` |
| `run_perplexity` | `results/perplexity_<tag>.json` |
| `run_lens_checks` | `results/lens_checks_<tag>.json` |
| `run_train` | `<out>/` (adapter + `train_info.json`) |
| `fit_lens` | `<out>` (lens `.pt`) |
| `viz.run_app` | nothing (serves a local web page) |
