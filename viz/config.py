"""Viewer settings, read from configs/viz.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_VIZ_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "viz.yaml"


@dataclass(frozen=True)
class VizConfig:
    runs_dir: Path                  # folder whose subfolders are saved adapters
    split_file: Path | None         # {split: [item names]}; None offers only "all"
    top_n: int                      # top tokens per (position, layer) cell
    max_tracked: int | None         # cap on non-pinned tracked tokens; None = every top token
    page_height: int                # iframe height, pixels
    server_name: str
    server_port: int


def load_viz_config(path: str | Path = DEFAULT_VIZ_CONFIG) -> VizConfig:
    """Read viz.yaml. `runs_dir` and `split_file` are relative to the file's folder."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return VizConfig(**{
        **data,
        "runs_dir": _relative_to(path, data["runs_dir"]),
        "split_file": None if data["split_file"] is None else _relative_to(path, data["split_file"]),
    })


def _relative_to(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()
