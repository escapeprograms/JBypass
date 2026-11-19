"""J-space settings: workspace band, ablation, decomposition and lens fitting. The
dataclass holds the defaults; a run overrides any of them in the optional `jspace:`
section of its run.yaml (`jspace_config`). The model and its prefit lens location live
in the model profile (configs/models/*.yaml), not here."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from model.config import RunConfig


@dataclass(frozen=True)
class JSpaceConfig:
    # Workspace band as a fraction of depth. The paper reports ~L38 to ~L92 on a
    # 0-100 scale for Claude; it gives no band for Qwen, so we reuse the fractions.
    band_start: float = 0.38
    band_end: float = 0.92
    # Explicit inclusive (first, last) band layer indices; when set, it replaces the
    # fractions above. None = use the fractions (layers 11-26 on the 28-layer Qwen3-1.7B).
    band_layers: tuple[int, int] | None = None

    # Paper's J-space ablation: top-k most active J-lens vectors per position.
    ablation_top_k: int = 10
    # Strengths: the paper's light / medium / heavy "differ in the range of
    # layers". We use the first third, two thirds, or all of the band.
    ablation_strengths: tuple = (("light", 1 / 3), ("medium", 2 / 3), ("heavy", 1.0))
    # Leading real tokens of each row never edited (attention sink; see interventions.py).
    ablation_skip_first: int = 1
    # Paper excludes tokens in the clean forward pass's top-k output from ablation.
    ablation_exclude_top_k: int = 10

    # Readout clamp (clamp.ReadoutClamp): cap each readout at mean + clamp_z * std of
    # its clean reference distribution (0 = clamp to the mean).
    clamp_z: float = 0.5
    # Folder of reference readout statistics, one <profile name>.pt per model.
    readout_stats_dir: Path = Path("stats")
    # Texts per forward pass when collecting them ([batch, seq, vocab] fp32 per layer).
    stats_batch_size: int = 4

    # J-space decomposition: sparse nonnegative combination of at most k J-lens
    # vectors ("typically no more than 25" in the paper).
    decompose_k: int = 25

    # Refitting a lens (e.g. on a fine-tuned model). Paper: 1000 wikitext
    # sequences of 128 tokens; the README says ~100 is usable.
    fit_n_prompts: int = 1000
    fit_max_seq_len: int = 128
    fit_dim_batch: int = 32
    # Prompts between writes of the resumable <out>.ckpt (None = only at the end).
    # Each write is ~450 MB for Qwen3-1.7B.
    fit_checkpoint_every: int | None = 10


def jspace_config(run: RunConfig) -> JSpaceConfig:
    """JSpaceConfig defaults with run.yaml's `jspace:` section applied on top. Unknown keys
    raise; relative paths in the section are relative to the YAML file's folder."""
    _check_keys(run.jspace)
    values = {key: _convert(key, value, run) for key, value in run.jspace.items()}
    return dataclasses.replace(JSpaceConfig(), **values)


def _check_keys(section: dict) -> None:
    known = {f.name for f in dataclasses.fields(JSpaceConfig)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown jspace setting(s) {unknown} in run.yaml; known: {sorted(known)}")


def _convert(key: str, value, run: RunConfig):
    """A YAML value as the type its JSpaceConfig field expects."""
    converter = _CONVERTERS.get(key)
    return value if converter is None else converter(value, run)


def _config_relative_path(value, run: RunConfig) -> Path:
    return (run.config_dir / value).resolve()


def _layer_pair(value, run: RunConfig) -> tuple[int, int] | None:
    """A YAML [first, last] list as a tuple of two ints; null stays None (= the fractions).
    Whether the layers exist is checked when the band is built (jspace.vectors.layer_range)."""
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value)):
        raise ValueError(f"jspace.band_layers must be two integer layer indices [first, last], got {value!r}")
    return (value[0], value[1])


# Fields whose YAML value needs converting: key -> (value, run) -> field value.
_CONVERTERS = {
    "readout_stats_dir": _config_relative_path,
    "band_layers": _layer_pair,
}
