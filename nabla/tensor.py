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

from contextlib import contextmanager
import operator

import numpy as np

_grad_enabled = True


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


def _normalize_axes(axis, ndim: int):
    if axis is None:
        return None
    axes = axis if isinstance(axis, tuple) else (axis,)
    normalized = []
    for ax in axes:
        ax = operator.index(ax)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(f"axis {ax} is out of bounds for tensor of dimension {ndim}")
        if ax in normalized:
            raise ValueError("duplicate axis")
        normalized.append(ax)
    return tuple(normalized)


def _expand_reduction_grad(grad: np.ndarray, axis, ndim: int, keepdims: bool) -> np.ndarray:
    if axis is None or keepdims:
        return grad
    return np.expand_dims(grad, _normalize_axes(axis, ndim))


def _squeeze_reduction(data: np.ndarray, axis, keepdims: bool) -> np.ndarray:
    return data if keepdims else np.squeeze(data, axis=axis)


def _freeze_index(idx):
    if isinstance(idx, tuple):
        return tuple(_freeze_index(part) for part in idx)
    if isinstance(idx, np.ndarray):
        return idx.copy()
    if isinstance(idx, list):
        return np.array(idx)
    return idx


def _ensure_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)


def _requires_grad(*tensors: "Tensor") -> bool:
    return _grad_enabled and any(t.requires_grad for t in tensors)


@contextmanager
def no_grad():
    """Disable graph construction for op outputs created in this context."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


class Tensor:
    _active_backward_grads = None

    def __init__(self, data, _children=(), _op="", requires_grad: bool = True):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._is_leaf = requires_grad and len(_children) == 0
        self._backward = lambda: None       # closure set by each op
        self._prev = {child for child in _children if child.requires_grad} if requires_grad else set()
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

    def detach(self) -> "Tensor":
        """Return a non-grad-tracking tensor with copied data and no parents."""
        return Tensor(self.data.copy(), requires_grad=False)

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
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            self.data.reshape(shape),
            (self,),
            "reshape",
            requires_grad=_requires_grad(self),
        )

        def _backward():
            self._add_grad(out.grad.reshape(self.data.shape))

        if out.requires_grad:
            out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)

        out = Tensor(
            self.data.transpose(axes),
            (self,),
            "transpose",
            requires_grad=_requires_grad(self),
        )
        axes_normalized = _normalize_axes(axes, self.data.ndim)
        inverse = np.argsort(axes_normalized)

        def _backward():
            self._add_grad(out.grad.transpose(inverse))

        if out.requires_grad:
            out._backward = _backward
        return out

    @property
    def T(self):
        if self.data.ndim != 2:
            raise ValueError("Tensor.T is only defined for 2-D tensors; use transpose() for general axes")
        return self.transpose(1, 0)

    def __getitem__(self, idx):
        idx = _freeze_index(idx)
        out = Tensor(
            self.data[idx],
            (self,),
            "getitem",
            requires_grad=_requires_grad(self),
        )

        def _backward():
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self._add_grad(grad)

        if out.requires_grad:
            out._backward = _backward
        return out

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

        if out.requires_grad:
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

        if out.requires_grad:
            out._backward = _backward
        return out

    def matmul(self, other):
        other = _ensure_tensor(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("matmul requires operands with ndim >= 2; 1-D operands are not supported")
        out = Tensor(
            self.data @ other.data,
            (self, other),
            "@",
            requires_grad=_requires_grad(self, other),
        )

        def _backward():
            # NumPy matmul broadcasts leading batch axes; its VJP must undo that.
            self._add_grad(
                _unbroadcast(
                    out.grad @ np.swapaxes(other.data, -1, -2),
                    self.data.shape,
                )
            )
            other._add_grad(
                _unbroadcast(
                    np.swapaxes(self.data, -1, -2) @ out.grad,
                    other.data.shape,
                )
            )

        if out.requires_grad:
            out._backward = _backward
        return out

    __matmul__ = matmul

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        if k < 0 and np.any(self.data == 0):
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        if not float(k).is_integer() and np.any(self.data < 0):
            raise ValueError("fractional powers of negative values are not real-valued")
        out = Tensor(self.data ** k, (self,), f"**{k}", requires_grad=_requires_grad(self))

        def _backward():
            local_grad = np.zeros_like(self.data) if k == 0 else k * self.data ** (k - 1)
            self._add_grad(local_grad * out.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            requires_grad=_requires_grad(self),
        )

        def _backward():
            g = _expand_reduction_grad(out.grad, axis, self.data.ndim, keepdims)
            self._add_grad(np.broadcast_to(g, self.data.shape))

        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = _normalize_axes(axis, self.data.ndim)
            n = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0.0),
            (self,),
            "relu",
            requires_grad=_requires_grad(self),
        )

        def _backward():
            self._add_grad((self.data > 0) * out.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh", requires_grad=_requires_grad(self))

        def _backward():
            self._add_grad((1.0 - t * t) * out.grad)   # d/dx tanh = 1 - tanh^2

        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self):
        s = np.empty_like(self.data)
        positive = self.data >= 0
        s[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        s[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(s, (self,), "sigmoid", requires_grad=_requires_grad(self))

        def _backward():
            self._add_grad(s * (1.0 - s) * out.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp", requires_grad=_requires_grad(self))

        def _backward():
            self._add_grad(e * out.grad)               # d/dx exp = exp

        if out.requires_grad:
            out._backward = _backward
        return out

    def logsumexp(self, axis=None, keepdims=False):
        m = np.max(self.data, axis=axis, keepdims=True)
        shifted_exp = np.exp(self.data - m)
        y_keepdims = m + np.log(shifted_exp.sum(axis=axis, keepdims=True))
        out = Tensor(
            _squeeze_reduction(y_keepdims, axis, keepdims),
            (self,),
            "logsumexp",
            requires_grad=_requires_grad(self),
        )

        def _backward():
            g = _expand_reduction_grad(out.grad, axis, self.data.ndim, keepdims)
            self._add_grad(np.exp(self.data - y_keepdims) * np.broadcast_to(g, self.data.shape))

        if out.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, axis=-1):
        m = np.max(self.data, axis=axis, keepdims=True)
        shifted_exp = np.exp(self.data - m)
        p = shifted_exp / shifted_exp.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,), "softmax", requires_grad=_requires_grad(self))

        def _backward():
            dot = (out.grad * p).sum(axis=axis, keepdims=True)
            self._add_grad(p * (out.grad - dot))

        if out.requires_grad:
            out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        y_keepdims = np.max(self.data, axis=axis, keepdims=True)
        out = Tensor(
            _squeeze_reduction(y_keepdims, axis, keepdims),
            (self,),
            "max",
            requires_grad=_requires_grad(self),
        )
        mask = self.data == y_keepdims
        ties = mask.sum(axis=axis, keepdims=True)

        def _backward():
            g = _expand_reduction_grad(out.grad, axis, self.data.ndim, keepdims)
            self._add_grad(mask * np.broadcast_to(g, self.data.shape) / ties)

        if out.requires_grad:
            out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        y_keepdims = np.min(self.data, axis=axis, keepdims=True)
        out = Tensor(
            _squeeze_reduction(y_keepdims, axis, keepdims),
            (self,),
            "min",
            requires_grad=_requires_grad(self),
        )
        mask = self.data == y_keepdims
        ties = mask.sum(axis=axis, keepdims=True)

        def _backward():
            g = _expand_reduction_grad(out.grad, axis, self.data.ndim, keepdims)
            self._add_grad(mask * np.broadcast_to(g, self.data.shape) / ties)

        if out.requires_grad:
            out._backward = _backward
        return out

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("log is only defined for positive values")
        out = Tensor(np.log(self.data), (self,), "log", requires_grad=_requires_grad(self))

        def _backward():
            self._add_grad((1.0 / self.data) * out.grad)

        if out.requires_grad:
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

        if out.requires_grad:
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

        if out.requires_grad:
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

        Do not mutate any tensor's ``.data`` between the forward pass and this
        call. Some VJPs intentionally re-read saved ``ndarray`` buffers during
        backward, so raw NumPy writes can corrupt gradients. Double backward is
        not supported: VJPs operate on raw arrays rather than building a graph
        while gradients are propagated.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")
        if Tensor._active_backward_grads is not None:
            raise RuntimeError("re-entrant backward() is not supported")

        topo, visited = [], set()

        visited.add(self)
        stack = [(self, iter(self._prev))]
        while stack:
            node, children = stack[-1]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))
                    break
            else:
                topo.append(node)
                stack.pop()

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


def concat(tensors, axis=0):
    tensors = tuple(_ensure_tensor(t) for t in tensors)
    if not tensors:
        raise ValueError("concat requires at least one tensor")
    axis = operator.index(axis)
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, tensors, "concat", requires_grad=_requires_grad(*tensors))
    axis_normalized = axis if axis >= 0 else axis + data.ndim
    sizes = [t.data.shape[axis_normalized] for t in tensors]
    splits = np.cumsum(sizes)[:-1]

    def _backward():
        for tensor, grad in zip(tensors, np.split(out.grad, splits, axis=axis_normalized)):
            tensor._add_grad(grad)

    if out.requires_grad:
        out._backward = _backward
    return out


def stack(tensors, axis=0):
    tensors = tuple(_ensure_tensor(t) for t in tensors)
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    axis = operator.index(axis)
    data = np.stack([t.data for t in tensors], axis=axis)
    out = Tensor(data, tensors, "stack", requires_grad=_requires_grad(*tensors))
    axis_normalized = axis if axis >= 0 else axis + data.ndim

    def _backward():
        for i, tensor in enumerate(tensors):
            tensor._add_grad(np.take(out.grad, i, axis=axis_normalized))

    if out.requires_grad:
        out._backward = _backward
    return out
