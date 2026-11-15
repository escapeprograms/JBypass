"""J-space settings: workspace band, ablation, decomposition and lens fitting. Change
values here, not inline. The model and its prefit lens location live in the model
profile (configs/models/*.yaml), not here."""

from dataclasses import dataclass


# TODO(debt-002): dataclass defaults, while run and model settings are YAML.
@dataclass(frozen=True)
class JSpaceConfig:
    # Workspace band as a fraction of depth. The paper reports ~L38 to ~L92 on a
    # 0-100 scale for Claude; it gives no band for Qwen, so we reuse the fractions.
    band_start: float = 0.38
    band_end: float = 0.92

    # Paper's J-space ablation: top-k most active J-lens vectors per position.
    ablation_top_k: int = 10
    # Strengths: the paper's light / medium / heavy "differ in the range of
    # layers". We use the first third, two thirds, or all of the band.
    ablation_strengths: tuple = (("light", 1 / 3), ("medium", 2 / 3), ("heavy", 1.0))
    # Leading real tokens of each row never edited (attention sink; see interventions.py).
    ablation_skip_first: int = 1
    # Paper excludes tokens in the clean forward pass's top-k output from ablation.
    ablation_exclude_top_k: int = 10

    # J-space decomposition: sparse nonnegative combination of at most k J-lens
    # vectors ("typically no more than 25" in the paper).
    decompose_k: int = 25

    # Refitting a lens (e.g. on a fine-tuned model). Paper: 1000 wikitext
    # sequences of 128 tokens; the README says ~100 is usable.
    fit_n_prompts: int = 1000
    fit_max_seq_len: int = 128
    fit_dim_batch: int = 32
