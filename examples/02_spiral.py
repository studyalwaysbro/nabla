"""Classify the interleaved-spirals dataset — three classes, not linearly
separable — to show the engine handles real multi-class training.

    python examples/02_spiral.py

Expect ~99% training accuracy. The same network with the nonlinearity removed
can't crack ~50%, which is the whole point of having working gradients through
the activations.
"""

import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nabla import MLP, SGD, Tensor, cross_entropy


def make_spiral(points_per_class=100, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((points_per_class * classes, 2))
    y = np.zeros(points_per_class * classes, dtype=int)
    for c in range(classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.standard_normal(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


X_np, y_np = make_spiral()
X = Tensor(X_np, requires_grad=False)

rng = np.random.default_rng(2)
net = MLP([2, 64, 64, 3], rng, activation="relu")
opt = SGD(net.parameters(), lr=0.2, momentum=0.9)

for epoch in range(2000):
    opt.zero_grad()
    logits = net(X)
    loss = cross_entropy(logits, y_np)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        pred = net(X).data.argmax(axis=1)
        acc = (pred == y_np).mean()
        print(f"epoch {epoch:4d}  loss {float(loss.data):.4f}  acc {acc:.3f}")

pred = net(X).data.argmax(axis=1)
print(f"\nfinal training accuracy: {(pred == y_np).mean():.3f}")
