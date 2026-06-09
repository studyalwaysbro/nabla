"""Gradient checking by central finite differences.

For any scalar function f of a tensor, the i-th partial derivative is
approximately ``(f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``. We compare that
numeric estimate against the analytic gradient the engine produces. If a VJP in
tensor.py is wrong, the relative error blows past the tolerance and the test
fails. This is what makes "I implemented backprop from scratch" a claim you can
actually trust.
"""

import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nabla import (
    Adam,
    AdamW,
    MLP,
    SGD,
    Tensor,
    clip_grad_norm_,
    cross_entropy,
    mse_loss,
    no_grad,
)


def numeric_grad(f, t: Tensor, eps=1e-6) -> np.ndarray:
    """Central-difference gradient of scalar ``f()`` w.r.t. tensor ``t``."""
    g = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = t.data[idx]
        t.data[idx] = orig + eps
        plus = float(f().data)
        t.data[idx] = orig - eps
        minus = float(f().data)
        t.data[idx] = orig
        g[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return g


def analytic_grad(f, *tensors):
    for t in tensors:
        t.zero_grad()
    f().backward()
    return [t.grad.copy() for t in tensors]


def assert_close(a, b, name, tol=1e-5):
    rel = np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)
    assert rel.max() < tol, f"{name}: max rel err {rel.max():.2e}\n{a}\n!=\n{b}"


def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


rng = np.random.default_rng(0)


def test_elementwise_chain():
    x = Tensor(rng.standard_normal((3, 4)))
    y = Tensor(rng.standard_normal((3, 4)))
    f = lambda: ((x * y).tanh() + x.exp()).sum()
    (gx, gy), = [analytic_grad(f, x, y)]
    assert_close(gx, numeric_grad(f, x), "elementwise dx")
    assert_close(gy, numeric_grad(f, y), "elementwise dy")


def test_broadcasting_add_and_mul():
    x = Tensor(rng.standard_normal((4, 5)))     # matrix
    b = Tensor(rng.standard_normal((5,)))       # broadcast row vector
    f = lambda: ((x + b) * (x * b)).sum()
    gx, gb = analytic_grad(f, x, b)
    assert_close(gx, numeric_grad(f, x), "broadcast dx")
    assert_close(gb, numeric_grad(f, b), "broadcast db")   # the unbroadcast path


def test_broadcasting_crossed_singleton_axes():
    row = Tensor(rng.standard_normal((1, 4)))
    col = Tensor(rng.standard_normal((3, 1)))
    f = lambda: ((row + col) * (row - col)).sum()
    grow, gcol = analytic_grad(f, row, col)
    assert_close(grow, numeric_grad(f, row), "crossed broadcast drow")
    assert_close(gcol, numeric_grad(f, col), "crossed broadcast dcol")


def test_matmul():
    a = Tensor(rng.standard_normal((3, 4)))
    w = Tensor(rng.standard_normal((4, 2)))
    f = lambda: (a @ w).tanh().sum()
    ga, gw = analytic_grad(f, a, w)
    assert_close(ga, numeric_grad(f, a), "matmul dA")
    assert_close(gw, numeric_grad(f, w), "matmul dW")


def test_batched_matmul_gradcheck():
    a = Tensor(rng.standard_normal((2, 3, 4)))
    w = Tensor(rng.standard_normal((2, 4, 5)))
    f = lambda: (a @ w).tanh().sum()
    ga, gw = analytic_grad(f, a, w)
    assert_close(ga, numeric_grad(f, a), "batched matmul dA")
    assert_close(gw, numeric_grad(f, w), "batched matmul dW")


def test_batched_matmul_broadcasts_shared_rhs_gradcheck():
    a = Tensor(rng.standard_normal((2, 3, 4)))
    w = Tensor(rng.standard_normal((4, 5)))
    f = lambda: (a @ w).tanh().sum()
    ga, gw = analytic_grad(f, a, w)
    assert_close(ga, numeric_grad(f, a), "broadcast matmul dA")
    assert_close(gw, numeric_grad(f, w), "broadcast matmul dW")


def test_matmul_rejects_vector_operands():
    assert_raises(ValueError, lambda: Tensor(rng.standard_normal((3,))) @ Tensor(rng.standard_normal((3, 2))))
    assert_raises(ValueError, lambda: Tensor(rng.standard_normal((2, 3))) @ Tensor(rng.standard_normal((3,))))


def test_mean_and_pow():
    x = Tensor(rng.standard_normal((6,)) + 3.0)   # keep positive for **0.5
    f = lambda: ((x ** 2).mean() + (x ** 0.5).sum() + x.log().sum())
    gx, = analytic_grad(f, x)
    assert_close(gx, numeric_grad(f, x), "mean/pow/log dx")


def test_mean_tuple_axis():
    x = Tensor(rng.standard_normal((2, 3, 4)))
    f = lambda: x.mean(axis=(0, 2)).sum() + x.mean(axis=-1).sum()
    gx, = analytic_grad(f, x)
    assert_close(gx, numeric_grad(f, x), "tuple-axis mean dx")


def test_sub_and_division():
    x = Tensor(np.array([[1.2, 1.6, 2.1], [0.8, 1.1, 1.7]]))
    y = Tensor(np.array([[2.0, 1.5, 2.5]]))
    f = lambda: (((x - y) / y) + (3.0 / x) + (5.0 - x)).sum()
    gx, gy = analytic_grad(f, x, y)
    assert_close(gx, numeric_grad(f, x), "sub/div dx")
    assert_close(gy, numeric_grad(f, y), "sub/div dy")


def test_pow_zero_and_domain_guards():
    x = Tensor(np.array([0.0, 2.0]))
    f = lambda: (x ** 0).sum()
    gx, = analytic_grad(f, x)
    assert_close(gx, numeric_grad(f, x), "pow zero dx")

    assert_raises(ValueError, lambda: Tensor([-1.0]).log())
    assert_raises(ValueError, lambda: Tensor([-1.0]) ** 0.5)
    assert_raises(ZeroDivisionError, lambda: Tensor([1.0]) / Tensor([0.0]))


def test_repeated_backward_accumulates_without_zero_grad():
    x = Tensor(np.array([0.2, -0.4, 0.7]))
    w = Tensor(np.array([1.5, -0.5, 2.0]))
    h = x * w
    y = (h.tanh() + h).sum()
    assert x.is_leaf
    assert w.is_leaf
    assert not h.is_leaf
    assert not y.is_leaf

    y.backward()
    gx = x.grad.copy()
    gw = w.grad.copy()
    gh = h.grad.copy()

    y.backward()
    assert_close(x.grad, 2.0 * gx, "repeat accumulate dx", tol=1e-12)
    assert_close(w.grad, 2.0 * gw, "repeat accumulate dw", tol=1e-12)
    assert_close(h.grad, 2.0 * gh, "repeat accumulate intermediate", tol=1e-12)

    def fresh_graph():
        h_fresh = x * w
        return (h_fresh.tanh() + h_fresh).sum()

    assert_close(gx, numeric_grad(fresh_graph, x), "repeat backward numeric dx")
    assert_close(gw, numeric_grad(fresh_graph, w), "repeat backward numeric dw")


def test_backward_handles_deep_graph_iteratively():
    x = Tensor(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0
    y.backward()
    assert_close(x.grad, np.array(1.0), "deep chain dx", tol=1e-12)


def test_backward_rejects_reentrant_calls():
    Tensor._active_backward_grads = {}
    try:
        assert_raises(RuntimeError, lambda: Tensor(1.0).backward())
    finally:
        Tensor._active_backward_grads = None


def test_repeated_backward_with_leaf_zero_grad_recomputes_leaf_grads():
    x = Tensor(np.array([0.2, -0.4, 0.7]))
    w = Tensor(np.array([1.5, -0.5, 2.0]))
    h = x * w
    y = (h.tanh() + h).sum()

    y.backward()
    gx = x.grad.copy()
    gw = w.grad.copy()

    x.zero_grad()
    w.zero_grad()
    y.backward()
    assert_close(x.grad, gx, "repeat after zero_grad dx", tol=1e-12)
    assert_close(w.grad, gw, "repeat after zero_grad dw", tol=1e-12)


def test_minibatch_gradient_accumulation_sums_expected_total():
    w = Tensor(np.array([[0.2], [-0.4]]))
    b = Tensor(np.array([0.1]))
    batches = [
        (
            np.array([[1.0, 0.0], [0.5, -1.0]]),
            np.array([[0.3], [-0.2]]),
        ),
        (
            np.array([[-1.0, 2.0], [0.0, 1.5], [2.0, -0.5]]),
            np.array([[0.7], [0.1], [-0.4]]),
        ),
    ]

    def loss_for(x_batch, y_batch):
        x = Tensor(x_batch, requires_grad=False)
        return mse_loss(x @ w + b, y_batch)

    expected_w = np.zeros_like(w.data)
    expected_b = np.zeros_like(b.data)
    for x_batch, y_batch in batches:
        w.zero_grad()
        b.zero_grad()
        loss_for(x_batch, y_batch).backward()
        expected_w += w.grad.copy()
        expected_b += b.grad.copy()

    opt = SGD([w, b], lr=0.05)
    opt.zero_grad()
    for x_batch, y_batch in batches:
        loss_for(x_batch, y_batch).backward()

    assert_close(w.grad, expected_w, "minibatch accumulated dw", tol=1e-12)
    assert_close(b.grad, expected_b, "minibatch accumulated db", tol=1e-12)
    opt.zero_grad()
    assert_close(w.grad, np.zeros_like(w.data), "optimizer zero_grad dw", tol=1e-12)
    assert_close(b.grad, np.zeros_like(b.data), "optimizer zero_grad db", tol=1e-12)


def test_requires_grad_false_leaf_is_constant():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=False)
    w = Tensor(np.array([0.5, -1.0, 2.0]))
    (x * w).sum().backward()
    assert x.grad is None
    assert_close(w.grad, x.data, "constant leaf skips grad")


def test_tensor_constructor_copies_array_data():
    arr = np.ones(3)
    t = Tensor(arr, requires_grad=False)
    arr[0] = 99.0
    assert_close(t.data, np.ones(3), "constructor copies data", tol=1e-12)


def test_mse_loss_rejects_broadcasting_shapes():
    pred = Tensor(np.zeros((4, 1)))
    target = np.zeros((4,))
    assert_raises(ValueError, lambda: mse_loss(pred, target))


def test_no_grad_outputs_have_no_graph():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    with no_grad():
        y = ((x * 2.0) + 1.0).sum()
        z = x.sum()
    assert not y.requires_grad
    assert y.grad is None
    assert y._prev == set()
    assert not z.requires_grad
    assert z._prev == set()
    assert_raises(RuntimeError, y.backward)


def test_detach_stops_gradient_and_copies_data():
    x = Tensor(np.array([2.0, 3.0]))
    d = x.detach()
    x.data[0] = 10.0
    assert_close(d.data, np.array([2.0, 3.0]), "detach copies data", tol=1e-12)

    x = Tensor(np.array([2.0, 3.0]))
    (x * x.detach()).sum().backward()
    assert_close(x.grad, np.array([2.0, 3.0]), "detach boundary dx", tol=1e-12)


def test_adam_decreases_quadratic_loss():
    p = Tensor(np.array([3.0, -4.0]))
    opt = Adam([p], lr=0.05)
    start = float((p * p).sum().data)
    for _ in range(8):
        opt.zero_grad()
        loss = (p * p).sum()
        loss.backward()
        opt.step()
    end = float((p * p).sum().data)
    assert end < start


def test_adam_bias_correction_first_step():
    p = Tensor(np.array([1.0, -2.0]))
    p.grad = np.array([0.5, -0.25])
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    opt.step()
    expected = np.array([1.0, -2.0]) - 0.1 * np.array([0.5, -0.25]) / (np.array([0.5, 0.25]) + 1e-8)
    assert_close(p.data, expected, "adam first step", tol=1e-12)


def test_adamw_decoupled_decay_shrinks_zero_grad_weights():
    p = Tensor(np.array([2.0, -3.0]))
    p.grad = np.zeros_like(p.data)
    opt = AdamW([p], lr=0.1, weight_decay=0.01)
    opt.step()
    assert_close(p.data, np.array([2.0, -3.0]) * (1.0 - 0.1 * 0.01), "adamw decoupled decay", tol=1e-12)


def test_clip_grad_norm_returns_preclip_norm_and_scales():
    a = Tensor(np.array([0.0, 0.0]))
    b = Tensor(np.array([0.0]))
    a.grad = np.array([3.0, 4.0])
    b.grad = np.array([12.0])
    norm = clip_grad_norm_([a, b], 5.0)
    scale = 5.0 / (13.0 + 1e-6)
    assert abs(norm - 13.0) < 1e-12
    assert_close(a.grad, np.array([3.0, 4.0]) * scale, "clip grad a", tol=1e-12)
    assert_close(b.grad, np.array([12.0]) * scale, "clip grad b", tol=1e-12)


def test_mlp_train_step_unaffected():
    local_rng = np.random.default_rng(123)
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    net = MLP([2, 4, 1], local_rng, activation="tanh")
    opt = SGD(net.parameters(), lr=0.05)
    params = net.parameters()
    before = [p.data.copy() for p in params]

    opt.zero_grad()
    loss = mse_loss(net(x), y)
    loss.backward()
    grads = [p.grad.copy() for p in params]
    assert any(np.linalg.norm(g) > 0 for g in grads)
    opt.step()

    for p, old, grad in zip(params, before, grads):
        assert_close(p.data, old - 0.05 * grad, "mlp sgd step", tol=1e-12)
    assert x.grad is None


def test_cross_entropy_gradient():
    logits = Tensor(rng.standard_normal((5, 3)))
    targets = np.array([0, 2, 1, 1, 0])
    f = lambda: cross_entropy(logits, targets)
    gx, = analytic_grad(f, logits)
    assert_close(gx, numeric_grad(f, logits), "cross_entropy dlogits", tol=1e-4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("--- all gradient checks passed ---")
