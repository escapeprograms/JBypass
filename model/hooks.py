"""The contract with the J-space module: what an intervention is and how it attaches
to the residual stream.

Hook point "layer i" is the residual returned by decoder block i (lm.blocks[i]).
Hooks carry gradients, so interventions and captures work during training.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.hooks import RemovableHandle

from model.loading import LanguageModel


@dataclass(frozen=True)
class HookContext:
    layer: int
    attention_mask: torch.Tensor   # [batch, seq], aligned with resid; 0 marks left padding


class Intervention(Protocol):
    """Edits the residual at each of `layers`. Returns a new [batch, seq, d_model]
    tensor in resid.dtype; never edits `resid` in place."""

    layers: Sequence[int]

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor: ...


class PerRow:
    """Satisfies Intervention: rows[i] applies to batch row i only, so each example in a
    batch gets its own edits. Each row's interventions see that row's slice of the
    residual and of the attention mask."""

    def __init__(self, rows: Sequence[Sequence[Intervention]]):
        self.rows = [list(row) for row in rows]
        self.layers = sorted({layer for row in self.rows for iv in row for layer in iv.layers})

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        if resid.shape[0] != len(self.rows):
            raise ValueError(f"PerRow has {len(self.rows)} rows but the batch has {resid.shape[0]}")
        return torch.cat([self._edit_row(i, resid[i:i + 1], ctx) for i in range(len(self.rows))])

    def _edit_row(self, i: int, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        row_ctx = HookContext(ctx.layer, ctx.attention_mask[i:i + 1])
        for iv in self.rows[i]:
            if ctx.layer in iv.layers:
                resid = iv(resid, row_ctx)
        return resid


def block_residual(output) -> torch.Tensor:
    """The residual from a block's output: the tensor itself, or the first item of a tuple."""
    return output if isinstance(output, torch.Tensor) else output[0]


def with_block_residual(output, resid: torch.Tensor):
    """`output` with its residual replaced by `resid`, keeping the block's output shape."""
    return resid if isinstance(output, torch.Tensor) else (resid, *output[1:])


def _intervention_hook(intervention: Intervention, ctx: HookContext):
    def hook(module, inputs, output):
        resid = block_residual(output)
        new = intervention(resid, ctx)
        if new.shape != resid.shape or new.dtype != resid.dtype:
            raise ValueError(
                f"intervention at layer {ctx.layer} returned {tuple(new.shape)} {new.dtype}, "
                f"expected {tuple(resid.shape)} {resid.dtype}"
            )
        return with_block_residual(output, new)
    return hook


def _capture_hook(store: dict[int, torch.Tensor], layer: int):
    def hook(module, inputs, output):
        store[layer] = block_residual(output)
    return hook


@contextmanager
def _registered(handles: list[RemovableHandle]) -> Iterator[None]:
    """Removes the hook handles on exit, including on exception."""
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def apply_interventions(
    lm: LanguageModel, interventions: Sequence[Intervention], attention_mask: torch.Tensor
) -> Iterator[None]:
    """Apply each intervention at its layers for forward passes inside the block.
    Interventions at the same layer run in the order given."""
    handles = [
        lm.blocks[layer].register_forward_hook(_intervention_hook(iv, HookContext(layer, attention_mask)))
        for iv in interventions
        for layer in iv.layers
    ]
    with _registered(handles):
        yield


@contextmanager
def capture_residuals(lm: LanguageModel, layers: Sequence[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Yield {layer: resid} filled by forward passes inside the block (the latest pass wins).
    Values are in-graph (not detached). Opened inside apply_interventions, it records the
    post-intervention value."""
    store: dict[int, torch.Tensor] = {}
    handles = [lm.blocks[layer].register_forward_hook(_capture_hook(store, layer)) for layer in layers]
    with _registered(handles):
        yield store
