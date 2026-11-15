"""Dummy interventions for tests only (no J-space maths)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from model.hooks import HookContext


@dataclass
class Identity:
    layers: Sequence[int]
    seen: list[torch.Tensor] = field(default_factory=list)

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        self.seen.append(resid)
        return resid * 1


@dataclass
class ZeroOut:
    layers: Sequence[int]

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        return torch.zeros_like(resid)


@dataclass
class Replace:
    """Sets the residual to `fn(resid)` once the sequence reaches `from_len` tokens."""

    layers: Sequence[int]
    fn: Callable[[torch.Tensor], torch.Tensor]
    from_len: int = 0

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        return self.fn(resid) if resid.shape[1] >= self.from_len else resid * 1
