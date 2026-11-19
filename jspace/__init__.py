"""J-space tools on top of Anthropic's jlens: attach a lens to a loaded model, J-lens
vectors, ablation / swap / clamp interventions, readout statistics, and readouts."""

from jspace.baseline import ReadoutStats, collect_readout_stats, load_readout_stats, save_readout_stats
from jspace.clamp import ReadoutClamp
from jspace.config import JSpaceConfig, jspace_config
from jspace.interventions import LayerIntervention, LensSwap, ProjectOut, TopKAblation
from jspace.loading import JSpace, default_stats_path, lens_view, load_jspace, load_lens, profile_stats_path
from jspace.readout import lens_logits, readout_layers, residual_logits, token_ranks
from jspace.vectors import (
    band_prefix, layer_range, lens_activations, lens_vectors, single_token_ids, token_variants,
    workspace_band,
)

__all__ = [
    "JSpace", "JSpaceConfig", "LayerIntervention", "LensSwap", "ProjectOut", "ReadoutClamp",
    "ReadoutStats", "TopKAblation", "band_prefix", "collect_readout_stats", "default_stats_path",
    "jspace_config", "layer_range", "lens_activations", "lens_logits", "lens_vectors", "lens_view", "load_jspace",
    "load_lens", "load_readout_stats", "profile_stats_path", "readout_layers", "residual_logits", "save_readout_stats",
    "single_token_ids", "token_ranks", "token_variants", "workspace_band",
]
