# Gradient derivations

Every backward rule in `nabla/tensor.py` is a **vector-Jacobian product** (VJP).
Reverse-mode autodiff never forms a Jacobian explicitly — it propagates a
gradient (a "cotangent") backward through each op. For an op `y = f(x)`, given
the upstream gradient `ḡ = ∂L/∂y`, the rule computes `∂L/∂x = Jᵀ ḡ`, where `J`
is the Jacobian of `f`. Below, `ḡ` is `out.grad` in the code.

Each rule here is checked numerically in `tests/test_gradcheck.py`, so these
aren't just claims — they're falsifiable.

---

## Notation

- `L` — the scalar loss at the end of the graph.
- `ḡ_y = ∂L/∂y` — gradient flowing into op output `y` (a tensor the same shape as `y`).
- `⊙` — elementwise (Hadamard) product.
- For an elementwise op, `J` is diagonal, so `Jᵀ ḡ` collapses to an elementwise multiply.

---

## Elementwise ops (diagonal Jacobian)

For `y_i = f(x_i)`, the Jacobian is `diag(f'(x_i))`, so

```
∂L/∂x = f'(x) ⊙ ḡ_y
```

| op | `f(x)` | `f'(x)` | code |
|----|--------|---------|------|
| `exp`  | `eˣ`        | `eˣ = y`            | `self.grad += e * out.grad` |
| `log`  | `ln x`      | `1/x`               | `self.grad += (1/self.data) * out.grad` |
| `tanh` | `tanh x`    | `1 − tanh²x = 1 − y²` | `self.grad += (1 - t*t) * out.grad` |
| `relu` | `max(0,x)`  | `1[x>0]`            | `self.grad += (self.data > 0) * out.grad` |
| `pow`  | `xᵏ`        | `k·xᵏ⁻¹`            | `self.grad += (k * self.data**(k-1)) * out.grad` |

`tanh` is the one worth noticing: writing the derivative as `1 − y²` reuses the
forward output, so backward costs one multiply and no recomputation.

---

## Add and multiply, with broadcasting

Ignoring shapes first:

- `y = a + b` ⇒ `∂L/∂a = ḡ`, `∂L/∂b = ḡ`.
- `y = a ⊙ b` ⇒ `∂L/∂a = b ⊙ ḡ`, `∂L/∂b = a ⊙ ḡ` (product rule).

**The subtlety is broadcasting.** When `a` has shape `(4,5)` and `b` has shape
`(5,)`, NumPy copies `b` across all 4 rows before multiplying. `b` therefore
influences `L` through *every* row, so by the multivariable chain rule its
gradient is the **sum** of the per-row contributions:

```
∂L/∂b = Σ_rows (a ⊙ ḡ)
```

That summation is exactly what `_unbroadcast` does — it sums out any axis that
was stretched from size 1 (or prepended) during the forward broadcast, so the
returned gradient matches the operand's original shape. Forgetting this is the
classic silent bug: the code runs, the shapes happen to work after a stray
broadcast, and the gradients are quietly wrong. The broadcasting test catches it.

---

## Matrix multiply

For `C = A B` with `A:(m,k)`, `B:(k,n)`, and upstream `Ḡ = ∂L/∂C : (m,n)`:

```
∂L/∂A = Ḡ Bᵀ        (shape (m,k))
∂L/∂B = Aᵀ Ḡ        (shape (k,n))
```

Quick sanity check via a single entry: `C_ij = Σ_p A_ip B_pj`, so
`∂C_ij/∂A_ip = B_pj`. Then

```
∂L/∂A_ip = Σ_j ḡ_ij · B_pj = (Ḡ Bᵀ)_ip ✓
```

The transposes are forced by shape-matching alone, which is a handy way to
remember them: there's only one way to arrange `A`, `B`, `Ḡ` and two transposes
so the dimensions line up.

---

## Sum and mean (the broadcast adjoint)

`sum` is the adjoint of broadcasting. If `y = Σ x` (over some axis), then every
input element contributed once, so

```
∂L/∂x = broadcast(ḡ_y) to x.shape
```

`mean` is `sum / N`, so its gradient is the same broadcast scaled by `1/N`. In
code, `mean` is literally `self.sum(...) * (1/n)` and inherits `sum`'s VJP — no
separate rule needed, which is a small proof that the op set composes.

---

## Softmax cross-entropy (the clean one)

For logits `z : (N, C)`, softmax `p_k = e^{z_k} / Σ_j e^{z_j}`, and the loss for
one example with true class `y` is `L = −ln p_y`. The gradient w.r.t. the logits:

```
∂L/∂z_k = p_k − 1[k = y]
```

**Derivation.** `L = −z_y + ln Σ_j e^{z_j}`. Differentiate term by term:

- `∂(−z_y)/∂z_k = −1[k=y]`
- `∂ ln Σ_j e^{z_j} / ∂z_k = e^{z_k} / Σ_j e^{z_j} = p_k`

Add them: `∂L/∂z_k = p_k − 1[k=y]`. Averaged over a batch of `N`, that's
`(softmax(z) − onehot(y)) / N` — exactly the three lines in
`nn.cross_entropy._backward`. The two ugly forward terms (the `e^z` and the
log-sum) cancel into a subtraction. This is *why* the fused op is both faster
and more stable than composing `log`/`exp`/`sum` in the graph: the cancellation
is done analytically instead of numerically.
```
grad = probs.copy()
grad[arange(N), targets] -= 1     # p - onehot
grad /= N
```
