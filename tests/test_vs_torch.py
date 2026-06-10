import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nabla import Tensor, concat, cross_entropy, stack


def assert_np_close(actual, expected, name):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, err_msg=name)


def test_shape_numeric_matmul_program_matches_torch():
    rng = np.random.default_rng(10)
    x_np = rng.standard_normal((2, 3, 4))
    w_np = rng.standard_normal((4, 5))
    b_np = rng.standard_normal((5,))
    weights_np = rng.standard_normal((3, 2, 5))
    idx_np = np.array([0, 0, 2])

    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)
    weights = Tensor(weights_np, requires_grad=False)
    y = (x.reshape(6, 4) @ w + b).reshape(2, 3, 5).transpose(1, 0, 2)
    picked = y[idx_np]
    loss = (
        picked.logsumexp(axis=-1).sum()
        + (picked.softmax(axis=-1) * weights).sum()
        + y.max(axis=0).sum()
        - 0.25 * y.min(axis=1).sum()
    )
    loss.backward()

    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    weights_t = torch.tensor(weights_np, dtype=torch.float64)
    idx_t = torch.tensor(idx_np, dtype=torch.long)
    yt = (xt.reshape(6, 4) @ wt + bt).reshape(2, 3, 5).permute(1, 0, 2)
    picked_t = yt[idx_t]
    loss_t = (
        torch.logsumexp(picked_t, dim=-1).sum()
        + (torch.softmax(picked_t, dim=-1) * weights_t).sum()
        + torch.amax(yt, dim=0).sum()
        - 0.25 * torch.amin(yt, dim=1).sum()
    )
    loss_t.backward()

    assert_np_close(loss.data, loss_t.detach().numpy(), "loss")
    assert_np_close(x.grad, xt.grad.numpy(), "x grad")
    assert_np_close(w.grad, wt.grad.numpy(), "w grad")
    assert_np_close(b.grad, bt.grad.numpy(), "b grad")


def test_concat_stack_indexing_program_matches_torch():
    rng = np.random.default_rng(11)
    a_np = rng.standard_normal((2, 3))
    b_np = rng.standard_normal((2, 3))
    scale_np = rng.standard_normal((2, 2, 3))

    a = Tensor(a_np)
    b = Tensor(b_np)
    scale = Tensor(scale_np, requires_grad=False)
    joined = concat([a.sigmoid(), b.tanh()], axis=1)
    grouped = stack([joined[:, :3], joined[:, 3:]], axis=1)
    loss = (grouped.transpose(1, 0, 2).logsumexp(axis=2).sum() + (grouped * scale).mean())
    loss.backward()

    at = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    scale_t = torch.tensor(scale_np, dtype=torch.float64)
    joined_t = torch.cat([torch.sigmoid(at), torch.tanh(bt)], dim=1)
    grouped_t = torch.stack([joined_t[:, :3], joined_t[:, 3:]], dim=1)
    loss_t = torch.logsumexp(grouped_t.permute(1, 0, 2), dim=2).sum() + (grouped_t * scale_t).mean()
    loss_t.backward()

    assert_np_close(loss.data, loss_t.detach().numpy(), "loss")
    assert_np_close(a.grad, at.grad.numpy(), "a grad")
    assert_np_close(b.grad, bt.grad.numpy(), "b grad")


def test_cross_entropy_matches_torch():
    rng = np.random.default_rng(12)
    logits_np = rng.standard_normal((5, 4))
    targets_np = np.array([0, 3, 1, 2, 1])

    logits = Tensor(logits_np)
    loss = cross_entropy(logits, targets_np)
    loss.backward()

    logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    targets_t = torch.tensor(targets_np, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t, targets_t)
    loss_t.backward()

    assert_np_close(loss.data, loss_t.detach().numpy(), "cross entropy loss")
    assert_np_close(logits.grad, logits_t.grad.numpy(), "cross entropy grad")
