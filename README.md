# nabla ∇

**A reverse-mode automatic differentiation engine, written from scratch on
NumPy.** No PyTorch, no TensorFlow, no autograd library -- just the chain rule,
implemented with hand-derived VJPs and finite-difference checks.

```python
from nabla import Tensor

x = Tensor([[1., 2., 3.]])
y = (x * x).sum()          # y = Σ xᵢ²
y.backward()
print(x.grad)              # [[2. 4. 6.]]   (∂y/∂xᵢ = 2xᵢ)
```

`Tensor(...)` defaults to `requires_grad=True` in this repo; pass
`requires_grad=False` for data and constants.

That's a toy. The engine is real enough to train a neural net:

```bash
python examples/01_xor.py      # learns XOR (impossible for a linear model)
python examples/02_spiral.py   # 3-class spiral, ~99% accuracy
```

---

## Why this repo is different from every other "tiny autograd"

Anyone can write a backward pass that *looks* right. The hard part is knowing
it's *actually* right. This repo treats correctness as the deliverable:

1. **Every gradient is derived by hand** in [`docs/derivations.md`](docs/derivations.md)
   — as vector-Jacobian products, with the softmax-cross-entropy cancellation
   and the broadcasting adjoint worked out in full, not hand-waved.

2. **Every gradient is checked against finite differences** in
   [`tests/test_gradcheck.py`](tests/test_gradcheck.py). For each op, the
   analytic gradient from `backward()` is compared to a central-difference
   estimate `(f(x+ε) − f(x−ε)) / 2ε`. If a rule is wrong, the test fails. The
   derivation and the code can't silently disagree.

3. **The broadcasting case is handled correctly** — the place hand-rolled
   engines are most often quietly broken. When an operand is broadcast during
   the forward pass, its gradient is the *sum* over every position it was copied
   into; `_unbroadcast` does exactly that, and a dedicated test guards it.

---

## What's implemented

**Core** (`nabla/tensor.py`) — one `Tensor` class, ~200 lines:
`+`, `*`, `−`, `/`, `**`, `@` (matmul), `sum`, `mean`, `relu`, `tanh`, `exp`,
`log`, all with broadcasting-aware VJPs and reverse-topological `backward()`,
plus `no_grad()` and `detach()` for inference and graph boundaries.

**Neural nets** (`nabla/nn.py`): `Linear` (He init), `MLP`, `SGD` (with
momentum), `Adam`, `AdamW`, `clip_grad_norm_`, `mse_loss`, and a fused,
numerically-stable softmax `cross_entropy`. The layers and `mse_loss` compose
`Tensor` ops; `cross_entropy` has a fused VJP for the standard softmax-loss
gradient.

---

## How reverse-mode works, in three sentences

Each operation records the grad-enabled tensors that produced it, forming a DAG:
user-created grad-tracking tensors with no parents are leaves/parameters, while
op outputs are non-leaf intermediates. `backward()` walks that DAG in reverse
topological order so every node is visited only after everything depending on
it, seeds the output cotangent with `1`, applies each op's VJP, and accumulates
the result into public `.grad` buffers instead of clearing them. Non-leaf `.grad`
is exposed for inspection, but the training contract is to call `zero_grad()` on
leaves/optimizer parameters before a new step or accumulation window; repeated
`backward()` calls accumulate into leaf `.grad` by design, enabling gradient
accumulation across minibatches and explaining why `optimizer.zero_grad()` exists.
Do not mutate `.data` between a forward pass and its `backward()` call: several
VJPs re-read raw `ndarray` buffers, so in-place NumPy writes can corrupt the
gradient. Double backward is not supported because VJPs operate on raw ndarrays,
not on a graph-building tensor layer during backward.

---

## Run it

```bash
pip install -e .
python tests/test_gradcheck.py   # or: python -m pytest
python examples/01_xor.py
python examples/02_spiral.py
```

Requires Python ≥ 3.10 and NumPy.

## Roadmap

- `Conv2d` and a max-pool VJP are deferred
- a `graphviz` dump of the computation DAG

## License

MIT
