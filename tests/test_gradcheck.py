"""Gradient checking by central finite differences.

For any scalar function f of a tensor, the i-th partial derivative is
approximately ``(f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``. We compare that
numeric estimate against the analytic gradient the engine produces. If a VJP in
tensor.py is wrong, the relative error blows past the tolerance and the test
fails. This is what makes "I implemented backprop from scratch" a claim you can
actually trust.
"""

import numpy as np

from nabla import Tensor, cross_entropy


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


def test_matmul():
    a = Tensor(rng.standard_normal((3, 4)))
    w = Tensor(rng.standard_normal((4, 2)))
    f = lambda: (a @ w).tanh().sum()
    ga, gw = analytic_grad(f, a, w)
    assert_close(ga, numeric_grad(f, a), "matmul dA")
    assert_close(gw, numeric_grad(f, w), "matmul dW")


def test_mean_and_pow():
    x = Tensor(rng.standard_normal((6,)) + 3.0)   # keep positive for **0.5
    f = lambda: ((x ** 2).mean() + (x ** 0.5).sum() + x.log().sum())
    gx, = analytic_grad(f, x)
    assert_close(gx, numeric_grad(f, x), "mean/pow/log dx")


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
