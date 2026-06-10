"""nabla — a from-scratch reverse-mode automatic differentiation engine.

    from nabla import Tensor
    x = Tensor([[1., 2., 3.]])
    y = (x * x).sum()
    y.backward()
    x.grad          # -> [[2., 4., 6.]]   (d/dx of sum(x^2) = 2x)
"""

from nabla.tensor import Tensor, concat, no_grad, stack
from nabla.nn import Adam, AdamW, Dropout, LayerNorm, MLP, SGD, Linear, clip_grad_norm_, cross_entropy, mse_loss
from nabla.viz import draw_dot

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "concat",
    "stack",
    "no_grad",
    "Linear",
    "Dropout",
    "LayerNorm",
    "MLP",
    "SGD",
    "Adam",
    "AdamW",
    "clip_grad_norm_",
    "mse_loss",
    "cross_entropy",
    "draw_dot",
]
