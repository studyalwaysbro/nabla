"""Reverse-mode automatic differentiation over NumPy arrays.

A ``Tensor`` wraps an ``ndarray`` and records the operation that produced it.
Calling :meth:`Tensor.backward` walks the computation graph in reverse
topological order and accumulates the gradient of the output with respect to
every tensor that fed into it — one pass, exact (up to floating point), no
finite differences.

Every gradient rule here is the *vector-Jacobian product* (VJP) for that op.
They're derived by hand in ``docs/derivations.md`` and checked numerically in
``tests/test_gradcheck.py`` — if a rule below is wrong, that test fails.
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting.

    When ``a + b`` broadcasts a smaller operand up, the upstream gradient has
    the *broadcast* shape. The chain rule says the gradient w.r.t. the original
    operand is the sum over every position it was copied into — so we sum out
    the prepended axes and any axis that was size 1. This is the single most
    common place a hand-written autodiff engine is silently wrong.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None      # closure set by each op
        self._prev = set(_children)
        self._op = _op                     # for debugging / graph viz

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    # --------------------------------------------------------------- the ops
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # C = A @ B  ->  dA = dC @ B^T,  dB = A^T @ dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    __matmul__ = matmul

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Tensor(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad   # d/dx tanh = 1 - tanh^2

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad               # d/dx exp = exp

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------- python sugar
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(other) * -1.0)

    def __rsub__(self, other):
        return (self * -1.0) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    # --------------------------------------------------------- backward
    def backward(self) -> None:
        """Accumulate gradients into every ancestor's ``.grad``.

        Topologically orders the graph so each node is processed only after
        everything that depends on it, seeds the output gradient with ones
        (d(self)/d(self) = 1), then applies each VJP in reverse.
        """
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
