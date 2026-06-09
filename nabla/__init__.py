"""nabla — a from-scratch reverse-mode automatic differentiation engine.

    from nabla import Tensor
    x = Tensor([[1., 2., 3.]])
    y = (x * x).sum()
    y.backward()
    x.grad          # -> [[2., 4., 6.]]   (d/dx of sum(x^2) = 2x)
"""

from nabla.tensor import Tensor
from nabla.nn import MLP, SGD, Linear, cross_entropy, mse_loss

__version__ = "0.1.0"
__all__ = ["Tensor", "Linear", "MLP", "SGD", "mse_loss", "cross_entropy"]
