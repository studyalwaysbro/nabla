"""Reverse-mode automatic differentiation over NumPy arrays.

A ``Tensor`` wraps an ``ndarray`` and records the operation that produced it.
Calling :meth:`Tensor.backward` walks the computation graph in reverse
topological order and accumulates gradients into every grad-enabled tensor that
fed into it -- one pass, exact (up to floating point), no finite differences.

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


def _ensure_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)


def _requires_grad(*tensors: "Tensor") -> bool:
    return any(t.requires_grad for t in tensors)


class Tensor:
    _active_backward_grads = None

    def __init__(self, data, _children=(), _op="", requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._is_leaf = requires_grad and len(_children) == 0
        self._backward = lambda: None       # closure set by each op
        self._prev = {child for child in _children if child.requires_grad}
        self._op = _op                      # for debugging / graph viz

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_leaf(self) -> bool:
        """True for user-created, grad-tracking tensors, not op outputs."""
        return self._is_leaf

    def zero_grad(self) -> None:
        """Clear this tensor's public gradient buffer, if it tracks gradients."""
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    def _add_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        active = Tensor._active_backward_grads
        if active is not None:
            active[self] = active.get(self, np.zeros_like(self.data)) + grad
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    # --------------------------------------------------------------- the ops
    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            (self, other),
            "+",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            self._add_grad(_unbroadcast(out.grad, self.data.shape))
            other._add_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            (self, other),
            "*",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            self._add_grad(_unbroadcast(other.data * out.grad, self.data.shape))
            other._add_grad(_unbroadcast(self.data * out.grad, other.data.shape))

        out._backward = _backward
        return out

    def matmul(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            (self, other),
            "@",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            # C = A @ B  ->  dA = dC @ B^T,  dB = A^T @ dC
            self._add_grad(out.grad @ other.data.T)
            other._add_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    __matmul__ = matmul

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        if k < 0 and np.any(self.data == 0):
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        if not float(k).is_integer() and np.any(self.data < 0):
            raise ValueError("fractional powers of negative values are not real-valued")
        out = Tensor(self.data ** k, (self,), f"**{k}", requires_grad=self.requires_grad)

        def _backward():
            local_grad = np.zeros_like(self.data) if k == 0 else k * self.data ** (k - 1)
            self._add_grad(local_grad * out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            requires_grad=self.requires_grad,
        )

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._add_grad(np.broadcast_to(g, self.data.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        elif isinstance(axis, tuple):
            n = int(np.prod([self.data.shape[a] for a in axis]))
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            (self,),
            "relu",
            requires_grad=self.requires_grad,
        )

        def _backward():
            self._add_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh", requires_grad=self.requires_grad)

        def _backward():
            self._add_grad((1.0 - t * t) * out.grad)   # d/dx tanh = 1 - tanh^2

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp", requires_grad=self.requires_grad)

        def _backward():
            self._add_grad(e * out.grad)               # d/dx exp = exp

        out._backward = _backward
        return out

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("log is only defined for positive values")
        out = Tensor(np.log(self.data), (self,), "log", requires_grad=self.requires_grad)

        def _backward():
            self._add_grad((1.0 / self.data) * out.grad)

        out._backward = _backward
        return out

    # ----------------------------------------------------- python sugar
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            (self, other),
            "-",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            self._add_grad(_unbroadcast(out.grad, self.data.shape))
            other._add_grad(_unbroadcast(-out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return _ensure_tensor(other) - self

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("division by zero")
        out = Tensor(
            self.data / other.data,
            (self, other),
            "/",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            self._add_grad(_unbroadcast(out.grad / other.data, self.data.shape))
            other._add_grad(
                _unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.data.shape)
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return _ensure_tensor(other) / self

    # --------------------------------------------------------- backward
    def backward(self) -> None:
        """Accumulate gradients into every grad-enabled ancestor's ``.grad``.

        Topologically orders the graph so each node is processed only after
        everything that depends on it. ``backward()`` does not clear existing
        public gradients; repeated calls accumulate by design. Per-call scratch
        cotangents drive the VJPs so retained non-leaf ``.grad`` values remain
        inspectable without becoming stale upstream signal.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")

        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        grads = {self: np.ones_like(self.data)}
        Tensor._active_backward_grads = grads
        try:
            for v in reversed(topo):
                current_grad = grads.get(v)
                if current_grad is None:
                    continue

                if v.grad is None:
                    v.grad = np.zeros_like(v.data)
                public_grad = v.grad + current_grad
                v.grad = current_grad
                try:
                    v._backward()
                finally:
                    v.grad = public_grad
        finally:
            Tensor._active_backward_grads = None
