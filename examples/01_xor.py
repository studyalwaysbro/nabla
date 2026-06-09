"""The smallest convincing demo: learn XOR, which a linear model provably can't.

    python examples/01_xor.py

If the network drives the loss to ~0 and gets all four cases right, the
autodiff engine is computing correct gradients through two nonlinear layers.
"""

import numpy as np

from nabla import MLP, SGD, Tensor, mse_loss

X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0.0], [1.0], [1.0], [0.0]])

rng = np.random.default_rng(1)
net = MLP([2, 8, 1], rng, activation="tanh")
opt = SGD(net.parameters(), lr=0.5, momentum=0.9)

for epoch in range(2000):
    opt.zero_grad()
    loss = mse_loss(net(X), Y)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        print(f"epoch {epoch:4d}  loss {float(loss.data):.5f}")

print("\nlearned XOR:")
for xy, p in zip(X.data, net(X).data):
    print(f"  {int(xy[0])} XOR {int(xy[1])} = {p[0]:.3f}  -> {int(round(p[0]))}")
