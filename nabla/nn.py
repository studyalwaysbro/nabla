"""Tiny neural-net utilities on top of the autodiff core.

Layers and optimizers just compose or consume Tensor ops. The fused
``cross_entropy`` loss has its own VJP for numerical stability, while the rest
of the module relies on the engine in ``tensor.py``.
"""

from __future__ import annotations

import numpy as np

from nabla.tensor import Tensor


class Linear:
    """Affine layer ``y = x @ W + b`` with He initialization."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_features)          # He init, good with ReLU
        self.W = Tensor(rng.standard_normal((in_features, out_features)) * scale)
        self.b = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """Stack of Linear layers with a chosen activation between them."""

    def __init__(self, sizes, rng: np.random.Generator, activation="tanh"):
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:            # no activation on the output
                x = x.tanh() if self.activation == "tanh" else x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    """Plain stochastic gradient descent with optional momentum.

    Gradients accumulate in Tensor buffers, so call ``zero_grad()`` before each
    optimization step unless intentionally accumulating across minibatches.
    """

    def __init__(self, params, lr=0.1, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._vel = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        """Clear all parameter gradient buffers."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            self._vel[i] = self.momentum * self._vel[i] - self.lr * p.grad
            p.data += self._vel[i]


# ----------------------------------------------------------------- losses
def mse_loss(pred: Tensor, target) -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Softmax cross-entropy, fused for numerical stability.

    The gradient of softmax-cross-entropy w.r.t. the logits is the famously
    clean ``(softmax(z) - onehot(y)) / N`` — derived in docs/derivations.md.
    We compute it directly rather than composing log/exp/sum so the subtraction
    of the row-max (a stop-gradient constant) doesn't pollute the graph.
    """
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)            # stability shift (constant)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = z.shape[0]
    loss_val = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
    out = Tensor(loss_val, (logits,), "cross_entropy", requires_grad=logits.requires_grad)

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0          # softmax - onehot
        grad /= n
        logits._add_grad(grad * out.grad)

    out._backward = _backward
    return out
