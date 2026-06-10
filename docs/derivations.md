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
| `exp`  | `eˣ`        | `eˣ = y`            | `self._add_grad(e * out.grad)` |
| `log`  | `ln x`      | `1/x`               | `self._add_grad((1/self.data) * out.grad)` |
| `tanh` | `tanh x`    | `1 − tanh²x = 1 − y²` | `self._add_grad((1 - t*t) * out.grad)` |
| `relu` | `max(0,x)`  | `1[x>0]`            | `self._add_grad((self.data > 0) * out.grad)` |
| `pow`  | `xᵏ`        | `k·xᵏ⁻¹`            | `self._add_grad((k * self.data**(k-1)) * out.grad)` |

`tanh` is the one worth noticing: writing the derivative as `1 − y²` reuses the
forward output, so backward costs one multiply and no recomputation.

`log` is defined for positive inputs. Fractional powers of negative inputs and
division by zero are outside this real-valued engine's domain, so the code raises
instead of silently propagating `nan` or `inf`.

---

## Add, subtract, multiply, and divide, with broadcasting

Ignoring shapes first:

- `y = a + b` ⇒ `∂L/∂a = ḡ`, `∂L/∂b = ḡ`.
- `y = a - b` ⇒ `∂L/∂a = ḡ`, `∂L/∂b = -ḡ`.
- `y = a ⊙ b` ⇒ `∂L/∂a = b ⊙ ḡ`, `∂L/∂b = a ⊙ ḡ` (product rule).
- `y = a / b` ⇒ `∂L/∂a = ḡ / b`, `∂L/∂b = -(a / b²) ⊙ ḡ`.

**The subtlety is broadcasting.** When `a` has shape `(4,5)` and `b` has shape
`(5,)`, NumPy copies `b` across all 4 rows before multiplying. `b` therefore
influences `L` through *every* row, so by the multivariable chain rule its
gradient is the **sum** of the per-row contributions:

```
∂L/∂b = Σ_rows (a ⊙ ḡ)
```

That summation is exactly what `_unbroadcast` does for each operand's local VJP
contribution -- it sums out any axis that was stretched from size 1 (or
prepended) during the forward broadcast, so the returned gradient matches the
operand's original shape. Forgetting this is the classic silent bug: the code
runs, the shapes happen to work after a stray broadcast, and the gradients are
quietly wrong. The broadcasting tests catch it.

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

## Shape ops

Shape-only transforms do not change values, only where those values live.
Their VJPs are therefore the inverse shape movement.

For `y = reshape(x, shape)`, each output element is the same storage-order entry
as one input element, so the gradient just returns to the original shape:

```
∂L/∂x = reshape(ḡ_y, x.shape)
```

For `y = transpose(x, axes)`, the forward permutes axes. The backward applies
the inverse permutation:

```
inverse = argsort(axes)
∂L/∂x = transpose(ḡ_y, inverse)
```

Indexing is the adjoint of gathering. If `y = x[idx]`, the same input position
can be gathered multiple times. The gradient for that position is the sum of all
uses:

```
dx = zeros_like(x)
add_at(dx, idx, ḡ_y)
```

The `add_at` matters. Plain `dx[idx] += ḡ_y` looks equivalent but silently loses
updates for repeated integer indices because NumPy buffers the advanced-indexed
write. A gradcheck with repeated indices guards this exact trap.

For `concat([x₀, x₁, ...], axis)`, the forward lays tensors end-to-end. The VJP
splits `ḡ_y` at the same cumulative sizes and sends each slice back to its input.
For `stack`, the forward inserts a new axis, so input `i` receives
`take(ḡ_y, i, axis)`.

---

## Sigmoid, logsumexp, and softmax

For `s = sigmoid(x) = 1 / (1 + e^{-x})`, differentiating gives:

```
∂L/∂x = s ⊙ (1 - s) ⊙ ḡ_s
```

The stable `logsumexp` reduction is:

```
m = max(x)
y = m + log Σ_j exp(x_j - m)
```

Although the shift `m` depends on `x`, it is added outside and subtracted inside
the exponential sum, so its derivative terms cancel wherever the max is
differentiable. The remaining derivative is the normalized exponential:

```
∂y/∂x_i = exp(x_i - y)
∂L/∂x_i = exp(x_i - y) · broadcast(ḡ_y)_i
```

For `p = softmax(x)`, the Jacobian is:

```
∂p_i/∂x_j = p_i (1[i=j] - p_j)
```

Multiplying by upstream gradient `ḡ_p` collapses the Jacobian into the VJP:

```
∂L/∂x = p ⊙ (ḡ_p - Σ_j ḡ_p_j p_j)
```

The cross-entropy cancellation is the same fact in loss form. For one row of
logits and target `y`:

```
CE(z, y) = logsumexp(z) - z_y
∂CE/∂z_k = softmax(z)_k - 1[k = y]
```

That is why `nn.cross_entropy` can use the fused `(softmax(z) - onehot(y)) / N`
VJP: composing `logsumexp(z) - z_y` would derive the same gradient, but the
fused loss avoids numerically materializing the unstable intermediate terms.

---

## Max and min reductions

`max` and `min` are nondifferentiable at exact ties. This engine chooses the
HIPS-autograd convention: split the subgradient evenly across every tied
extremum. That differs from PyTorch's argmax-style routing for some reductions,
but it is symmetric and makes the reduction independent of memory order.

For `y = max(x, axis)`:

```
mask = x == broadcast(y)
∂L/∂x = mask ⊙ broadcast(ḡ_y) / sum(mask, axis, keepdims=True)
```

`min` uses the same rule with the minima mask. Away from ties this is the usual
one-hot gradient to the unique extremum; at two equal maxima, each receives half
the upstream gradient.

---

## LayerNorm (fused analysis, compositional code)

`LayerNorm(dim)` normalizes over the trailing feature axes. For one normalized
group:

```
μ = mean(x)
var = mean((x - μ)²)
x̂ = (x - μ) / sqrt(var + eps)
y = gamma ⊙ x̂ + beta
```

The implementation is deliberately compositional (`mean`, subtract, multiply,
power, divide, scale, shift), but the fused VJP is a useful check on the math.
Let `ĝ = ḡ_y ⊙ gamma` and let `mean(...)` reduce over the normalized feature
axes with `keepdims=True`. Then:

```
∂L/∂x = (1 / sqrt(var + eps)) ⊙ (ĝ - mean(ĝ) - x̂ ⊙ mean(ĝ ⊙ x̂))
∂L/∂gamma = Σ_nonfeature_axes ḡ_y ⊙ x̂
∂L/∂beta = Σ_nonfeature_axes ḡ_y
```

The central-difference LayerNorm test checks the compositional graph against
this result indirectly: if any primitive VJP mishandles broadcasting or
reductions, the LayerNorm gradient fails too.

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
logits._add_grad(grad * out.grad)
```
