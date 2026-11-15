import pytest
import torch

from jbypass.model.hooks import apply_interventions, block_residual, capture_residuals, with_block_residual

from conftest import D_MODEL, N_LAYERS, random_ids
from dummies import Identity, ZeroOut


def logits_of(lm, ids, mask):
    return lm.hf_model(input_ids=ids, attention_mask=mask).logits


def hook_count(lm):
    return sum(len(block._forward_hooks) for block in lm.blocks)


def test_identity_leaves_logits_unchanged_and_zeroing_changes_them(lm):
    ids, mask = random_ids(lm)
    base = logits_of(lm, ids, mask)
    with apply_interventions(lm, [Identity(layers=[1])], mask):
        same = logits_of(lm, ids, mask)
    with apply_interventions(lm, [ZeroOut(layers=[1])], mask):
        zeroed = logits_of(lm, ids, mask)
    torch.testing.assert_close(same, base)
    assert not torch.allclose(zeroed, base)


def test_intervention_sees_context(lm):
    ids, mask = random_ids(lm)
    seen = []
    class Spy:
        layers = [0, 2]
        def __call__(self, resid, ctx):
            seen.append(ctx)
            return resid * 1
    with apply_interventions(lm, [Spy()], mask):
        logits_of(lm, ids, mask)
    assert [c.layer for c in seen] == [0, 2]
    assert seen[0].attention_mask is mask


def test_hooks_removed_on_exit_and_on_exception(lm):
    ids, mask = random_ids(lm)
    base = logits_of(lm, ids, mask)
    with apply_interventions(lm, [ZeroOut(layers=[0, 1])], mask), capture_residuals(lm, [2]):
        assert hook_count(lm) == 3
    assert hook_count(lm) == 0
    with pytest.raises(RuntimeError):
        with apply_interventions(lm, [ZeroOut(layers=[0])], mask), capture_residuals(lm, [1]):
            raise RuntimeError("boom")
    assert hook_count(lm) == 0
    torch.testing.assert_close(logits_of(lm, ids, mask), base)


def test_wrong_shape_from_intervention_raises(lm):
    ids, mask = random_ids(lm)
    class Bad:
        layers = [0]
        def __call__(self, resid, ctx):
            return resid[:, :-1]
    with pytest.raises(ValueError), apply_interventions(lm, [Bad()], mask):
        logits_of(lm, ids, mask)


def test_captured_shapes_and_last_layer_unembeds_to_logits(lm):
    ids, mask = random_ids(lm, batch=2, seq=5)
    with capture_residuals(lm, range(lm.n_layers)) as resids:
        logits = logits_of(lm, ids, mask)
    assert sorted(resids) == list(range(N_LAYERS))
    assert all(r.shape == (2, 5, D_MODEL) for r in resids.values())
    torch.testing.assert_close(lm.unembed(resids[lm.n_layers - 1]), logits)


def test_capture_inside_interventions_sees_post_intervention_value(lm):
    ids, mask = random_ids(lm)
    with apply_interventions(lm, [ZeroOut(layers=[1])], mask), capture_residuals(lm, [1]) as resids:
        logits_of(lm, ids, mask)
    assert torch.count_nonzero(resids[1]) == 0


def test_gradients_reach_intervention_input(lm):
    ids, mask = random_ids(lm)
    spy = Identity(layers=[1])
    with apply_interventions(lm, [spy], mask), capture_residuals(lm, [0]) as resids:
        loss = logits_of(lm, ids, mask).sum()
    grad_in, grad_captured = torch.autograd.grad(loss, [spy.seen[0], resids[0]])
    assert grad_in.abs().sum() > 0
    assert grad_captured.abs().sum() > 0


def test_block_output_normalisation_handles_tensor_and_tuple():
    resid, extra, new = torch.ones(1, 2, 3), torch.zeros(1), torch.full((1, 2, 3), 5.0)
    assert block_residual(resid) is resid
    assert block_residual((resid, extra)) is resid
    assert with_block_residual(resid, new) is new
    rebuilt = with_block_residual((resid, extra), new)
    assert isinstance(rebuilt, tuple) and rebuilt[0] is new and rebuilt[1] is extra
