"""Settings for training under an ablation condition. Change values here, not inline.
Optimiser, LoRA and step settings come from configs/run.yaml."""

from dataclasses import dataclass

# Eval conditions a model may be trained under (names in evaluation/conditions.py):
# none = Mctrl, bridge = Mexp, random_tok / random_dir = Mrand. Top-k ablation is left
# out until it has a training test.
TRAIN_CONDITIONS = ("none", "bridge", "random_tok", "random_dir")


@dataclass(frozen=True)
class FinetuneConfig:
    dataset: str = "jbypass"            # name in EvalConfig.datasets
    split: str = "train"                # split in the --split_file to train on
    info_file: str = "train_info.json"  # written beside the adapter: condition, seeds, split, losses
