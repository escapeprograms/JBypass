"""Linear probes for the bridge entity (proposal, Evaluation section):
"train a linear probe to predict the bridge entity at each layer, from both the
full activation and its non J-space part".

If the probe finds the bridge but the original J-lens does not, the concept
likely moved to new directions; if neither finds it, the model may have skipped
the bridge. Accuracy is cross-validated by subject group: all questions about
the same subject (e.g. one landmark asked four ways) share a fold, so the probe
cannot score by recognizing the subject instead of the bridge. Every class
appears in train and test.
"""

from collections import Counter

import torch

from jspace.decompose import non_jspace_part


def probe_accuracy_by_layer(X: torch.Tensor, labels: list[str], groups: list[str], n_folds: int,
                            steps: int, weight_decay: float) -> dict[int, float]:
    """X: [n, n_layers, d]. groups: subject key per item. Uses only bridges with
    >= n_folds distinct subjects."""
    keep, y = _usable(labels, groups, n_folds)
    X, folds = X[keep], _folds([groups[i] for i in keep], y, n_folds)
    return {l: _cv_accuracy(X[:, l], y, folds, n_folds, steps, weight_decay)
            for l in range(X.shape[1])}


def strip_jspace(js, X: torch.Tensor, k: int) -> torch.Tensor:
    """Replace each [n, n_layers, d] residual with its non-J-space part."""
    layers = sorted(js.lens.source_layers)
    out = X.clone()
    W_U = js.lm.unembedding_matrix
    for l in layers:
        out[:, l] = non_jspace_part(js.lens, W_U, X[:, l].to(js.lm.device), l, k).cpu()
    return out


def chance_level(labels: list[str], groups: list[str], n_folds: int) -> float:
    _, y = _usable(labels, groups, n_folds)
    return Counter(y.tolist()).most_common(1)[0][1] / len(y)


def _usable(labels, groups, n_folds):
    subjects = {}
    for b, g in zip(labels, groups):
        subjects.setdefault(b, set()).add(g)
    classes = sorted(c for c, gs in subjects.items() if len(gs) >= n_folds)
    keep = [i for i, b in enumerate(labels) if b in classes]
    y = torch.tensor([classes.index(labels[i]) for i in keep], dtype=torch.long)
    return keep, y


def _folds(groups: list[str], y: torch.Tensor, n_folds: int) -> torch.Tensor:
    """Fold id per item: within each class, the i-th distinct subject goes to fold i % n_folds."""
    order: dict[int, list[str]] = {}
    for g, c in zip(groups, y.tolist()):
        order.setdefault(c, [])
        if g not in order[c]:
            order[c].append(g)
    return torch.tensor([order[c].index(g) % n_folds for g, c in zip(groups, y.tolist())])


def _cv_accuracy(X, y, folds, n_folds, steps, weight_decay) -> float:
    correct = 0
    for f in range(n_folds):
        train, test = folds != f, folds == f
        model = _fit_logreg(X[train], y[train], int(y.max()) + 1, steps, weight_decay)
        correct += int((model(X[test]).argmax(-1) == y[test]).sum())
    return correct / len(y)


def _fit_logreg(X, y, n_classes, steps, weight_decay):
    """Multinomial logistic regression on standardized features (full-batch Adam)."""
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    linear = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=weight_decay)
    Xs = (X - mean) / std
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(linear(Xs), y).backward()
        opt.step()
    return lambda Z: linear((Z - mean) / std)
