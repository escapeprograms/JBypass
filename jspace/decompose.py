"""Split residuals into a J-space part and a non-J-space part.

Paper (Methods, "The J-Space"): the J-space is the set of points expressible as
a sparse nonnegative combination of J-lens vectors (k <= 25), found "using
gradient pursuit". We implement nonnegative gradient pursuit: greedily add the
atom most correlated with the residual, take an exact line-search step along
the gradient restricted to the chosen atoms, clip coefficients at zero.
"""

import torch

from jspace.vectors import lens_activations


def jspace_decompose(lens, W_U: torch.Tensor, h: torch.Tensor, layer: int, k: int):
    """h: [n, d]. Returns (jspace_part [n, d], atom ids [n, k], coeffs [n, k]).
    TODO(debt): atoms are not unit-normalized for selection; the paper doesn't say."""
    h = h.float()
    J = lens.jacobians[layer].to(h.device)
    n = h.shape[0]
    support = torch.empty(n, 0, dtype=torch.long, device=h.device)
    coeffs = torch.empty(n, 0, device=h.device)
    residual = h.clone()
    for _ in range(k):
        support, coeffs = _add_best_atom(lens, W_U, residual, layer, support, coeffs)
        atoms = W_U[support].float() @ J  # [n, s, d]
        coeffs = _gradient_step(atoms, residual, coeffs)
        residual = h - torch.einsum("ns,nsd->nd", coeffs, atoms)
    return h - residual, support, coeffs


def non_jspace_part(lens, W_U, h, layer: int, k: int) -> torch.Tensor:
    """Residual left after removing its J-space reconstruction."""
    jspace_part, _, _ = jspace_decompose(lens, W_U, h, layer, k)
    return h.float() - jspace_part


def _add_best_atom(lens, W_U, residual, layer, support, coeffs):
    corr = lens_activations(lens, W_U, residual, layer)  # [n, V]
    if support.shape[1]:
        corr.scatter_(-1, support, float("-inf"))
    best = corr.argmax(-1, keepdim=True)
    zero = torch.zeros_like(best, dtype=coeffs.dtype)
    return torch.cat([support, best], 1), torch.cat([coeffs, zero], 1)


def _gradient_step(atoms, residual, coeffs):
    """One gradient pursuit step on the support, with exact line search, then clip >= 0."""
    grad = torch.einsum("nsd,nd->ns", atoms, residual)  # descent direction in coeff space
    direction = torch.einsum("ns,nsd->nd", grad, atoms)
    step = (residual * direction).sum(-1) / (direction * direction).sum(-1).clamp_min(1e-12)
    return (coeffs + step[:, None] * grad).clamp_min(0)
