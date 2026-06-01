# lau-conformal-agents

**Conformal geometry for agent systems — transformations that preserve angles but not distances.** Möbius transformations, Weyl tensor, Virasoro algebra, conformal field theory, Liouville's theorem, and conformal prediction for uncertainty quantification.

165 tests · MIT license · `nalgebra` + `num-complex` + `serde`

---

## What This Does

This crate implements conformal geometry — the mathematics of angle-preserving transformations — and applies it to agent systems. Conformal maps stretch and squeeze space but never distort angles, making them a natural language for:

1. **Möbius transformations** — the conformal automorphisms of the Riemann sphere: f(z) = (az+b)/(cz+d)
2. **Conformal maps** — 2D angle-preserving maps via Cauchy-Riemann equations, Jacobian analysis
3. **Liouville's theorem** — in dimensions n ≥ 3, only Möbius transformations are conformal
4. **Conformal Laplacian** — the Yamabe operator L = −Δ + ((n−2)/(4(n−1)))R that transforms conformally
5. **Weyl tensor** — the conformally invariant curvature tensor (vanishes iff conformally flat)
6. **Conformal weights** — how geometric quantities transform under g̃ = Ω²g
7. **Conformal compactification** — stereographic projection, adding ∞ to make Rⁿ → Sⁿ
8. **Conformal Field Theory** — primary fields, stress-energy tensor, OPE, Kac determinant
9. **Virasoro algebra** — central extension of the Witt algebra, [Lₘ, Lₙ] = (m−n)Lₘ₊ₙ + (c/12)(m³−m)δₘ₊ₙ,₀
10. **Agent learning** — conformal prediction intervals, nonconformity scores, belief rescaling

---

## Key Idea

A map f is **conformal** if it preserves angles: any two curves meeting at angle θ have images that also meet at angle θ. Equivalently, f pulls back the metric as a rescaling:

**f\*g = e²ᵖ g** (or Ω²g)

The geometry is preserved up to scale. This is weaker than isometry (which preserves distances) but much stronger than a general smooth map.

---

## Install

```toml
[dependencies]
lau-conformal-agents = "0.1.0"
```

Dependencies: `nalgebra = "0.33"`, `num-complex = "0.4"` (with serde), `serde = "1"`, `serde_json = "1"`, `approx = "0.5"`.

---

## Quick Start

```rust
use lau_conformal_agents::*;
use num_complex::Complex64;

fn main() {
    // Möbius transformation: f(z) = (z + 1) / (z - 1)
    let mobius = MobiusTransformation::new(
        Complex64::new(1.0, 0.0),  // a
        Complex64::new(1.0, 0.0),  // b
        Complex64::new(1.0, 0.0),  // c
        Complex64::new(-1.0, 0.0), // d
    );

    let z = Complex64::new(2.0, 0.0);
    let w = mobius.evaluate(z).unwrap();
    println!("f({}) = {}", z, w);

    // Fixed points: solve f(z) = z
    let fixed = mobius.fixed_points();
    println!("Fixed points: {:?}", fixed);

    // Virasoro algebra with central charge c = 1
    let vir = VirasoroAlgebra::new(1.0);
    let (lie_coeff, central) = vir.commutator(3, -3);
    println!("[L_3, L_-3] = {}·L_0 + {}", lie_coeff, central);

    // Conformal prediction interval
    let cal_scores = vec![0.1, 0.3, 0.2, 0.5, 0.15, 0.4];
    let (lo, hi) = conformal_prediction_interval(&cal_scores, 0.1, 42.0);
    println!("90% prediction interval: [{:.2}, {:.2}]", lo, hi);
}
```

---

## API Reference

### Möbius Transformations (`mobius`)

| Type | Description |
|------|-------------|
| `MobiusTransformation` | f(z) = (az+b)/(cz+d), the conformal automorphisms of the Riemann sphere |

**Methods:**
- `new(a, b, c, d)` — panics if degenerate (ad−bc ≈ 0)
- `try_new(a, b, c, d)` → `Option<Self>`
- `identity()` — f(z) = z
- `translation(b)` — f(z) = z + b
- `dilation(a)` — f(z) = az
- `inversion()` — f(z) = 1/z
- `evaluate(z)` → `Option<Complex64>` — None if maps to ∞
- `compose(&other)` — f∘g
- `inverse()` — f⁻¹
- `determinant()` → Complex64 — ad − bc
- `fixed_points()` → `Vec<Complex64>` — solutions to f(z) = z
- `is_elliptic()`, `is_parabolic()`, `is_hyperbolic()`, `is_loxodromic()` — classification
- `trace_squared()` → f64
- `normalize()` — scale so ad − bc = 1
- `to_matrix()` → `DMatrix<Complex64>` — 2×2 matrix [[a,b],[c,d]]
- `cross_ratio(z1, z2, z3, z4)` → Complex64 — conformal invariant
- `circle_to_circle(center, radius, samples)` → `(Vec<Complex64>, f64)` — maps circles to circles

### Conformal Maps (`conformal_map`)

| Type | Description |
|------|-------------|
| `ConformalMap` | 2D conformal map via complex derivative at a point |

**Methods:**
- `from_complex_derivative(z, f')` — construct from complex derivative
- `conformal_factor()` → f64 — |f'(z)|
- `jacobian()` → &DMatrix<f64> — 2×2 real Jacobian
- `is_conformal()` → bool — checks Cauchy-Riemann structure
- `pullback_metric_scale()` → f64 — |f'(z)|²

| Function | Description |
|----------|-------------|
| `exponential_map(z)` | w = eᶻ (conformal) |
| `logarithm_map(w)` | z = ln(w) (principal branch) |
| `power_map(z, α)` | w = zᵅ (conformal except at branch points) |
| `is_conformal_at(f, z, tol)` | Check if complex function f is conformal at z |

### Liouville's Theorem (`liouville`)

| Type | Description |
|------|-------------|
| `Dimension` | Enum: `Two`, `Three`, `Four`, `General(n)` |
| `ConformalClassification` | Result: dimension, is_conformal, description |

| Function | Description |
|----------|-------------|
| `is_conformal_matrix_2d(M)` | Check 2×2 matrix for Cauchy-Riemann structure |
| `is_conformal_matrix_3d(M)` | Check 3×3: MᵀM = λI (scaled rotation) |
| `classify_conformal_maps(dim)` | Liouville's theorem: what maps exist in each dimension |

### Conformal Laplacian (`conformal_laplacian`)

| Type | Description |
|------|-------------|
| `ConformalLaplacian` | L = −Δ + ((n−2)/(4(n−1)))R (Yamabe operator) |

**Methods:**
- `new(dimension, scalar_curvature)`
- `curvature_coefficient()` → f64 — (n−2)/(4(n−1))
- `apply(Δf, f)` → f64 — Lf = −Δf + c(n)·R·f
- `conformal_transform(sigma, n)` → `Self` — L̃ = e^((n+2)/2)σ · L · e^((n−2)/2)σ
- `yamabe_constant_estimate(stiffness, mass)` → `Option<f64>` — smallest eigenvalue (Rayleigh quotient)

### Weyl Tensor (`weyl`)

| Type | Description |
|------|-------------|
| `WeylTensor` | The conformally invariant curvature tensor Cᵢⱼₖₗ |

**Methods:**
- `zeros(n)` — zero tensor in n dimensions
- `from_riemann(R, Ricci, R_scalar, g)` — extract Weyl from full Riemann data
- `get(i, j, k, l)` / `set(i, j, k, l, val)` — component access
- `is_zero(tol)` → bool — vanishes iff conformally flat (n ≥ 4)
- `trace_free_check(tol)` → bool — verifies trace-free property

Formula: Cᵢⱼₖₗ = Rᵢⱼₖₗ − (1/(n−2))(gᵢₖRⱼₗ − gᵢₗRⱼₖ + gⱼₗRᵢₖ − gⱼₖRᵢₗ) + (1/((n−1)(n−2)))(gᵢₖgⱼₗ − gᵢₗgⱼₖ)R

### Conformal Weight (`conformal_weight`)

| Type | Description |
|------|-------------|
| `ConformalWeight` | Enum: `Invariant` (0), `Fixed(Δ)`, `DimensionDependent(f(n))` |
| `ConformalWeights` | Common weights: metric (2), inverse metric (−2), volume form (n), Weyl (0), etc. |

**Methods:**
- `weight(dim)` → f64 — numerical conformal weight
- `transform(value, Ω, dim)` → f64 — apply Ω^Δ scaling
- `transform_vector(v, Ω, dim)` → DVector — element-wise transformation

### Conformal Compactification (`compactification`)

| Type | Description |
|------|-------------|
| `StereographicProjection` | Sⁿ → Rⁿ projection from north/south pole |
| `ConformalCompactification` | Add point at ∞ to compactify Rⁿ → Sⁿ |

**StereographicProjection methods:**
- `north(n)`, `south(n)` — projection direction
- `project(point_on_sphere)` → `Option<DVector>` — None at pole (maps to ∞)
- `inverse(point_in_Rn)` → DVector — lift back to sphere

### Conformal Field Theory (`cft`)

| Type | Description |
|------|-------------|
| `PrimaryField` | CFT operator: name, scaling dimension Δ, spin s, weights h = (Δ+s)/2, h̄ = (Δ−s)/2 |
| `StressEnergyTensor` | T_μν: trace (zero for CFT), conformal anomaly, components |

**PrimaryField methods:**
- `new(name, Δ, s)` — construct with auto-computed weights
- `two_point_function(distance)` → f64 — 1/|x−y|^{2Δ}
- `three_point_function(fields, distances, C₁₂₃)` → f64
- `satisfies_unitarity_bound(spacetime_dim)` → bool — Δ ≥ |s| + n − 2

### Virasoro Algebra (`virasoro`)

| Type | Description |
|------|-------------|
| `VirasoroAlgebra` | [Lₘ, Lₙ] = (m−n)Lₘ₊ₙ + (c/12)(m³−m)δₘ₊ₙ,₀ with central charge c |

**Methods:**
- `new(c)` — create with central charge c
- `witt()` — c = 0 (classical Witt algebra)
- `commutator(m, n)` → (i64, f64) — (coefficient of Lₘ₊ₙ, central term)
- `central_term(m)` → f64 — (c/12)(m³−m)
- `verify_jacobi(k, m, n)` → f64 — Jacobi identity violation
- `kac_determinant(level, h)` → f64 — Kac formula at level n with highest weight h
- `descendant_states(level)` → `Vec<Vec<i64>>` — partitions generating descendant states

### Agent Learning (`agent_learning`)

| Type | Description |
|------|-------------|
| `ConformalPredictionSet<T>` | Prediction set with confidence, scores, threshold |

| Function | Description |
|----------|-------------|
| `conformal_prediction_interval(scores, α, prediction)` → (lo, hi) | Distribution-free prediction interval |
| `conformal_rescale_belief(belief, factor)` → DVector | Scale-preserving belief transform |
| `belief_angle(b1, b2)` → f64 | Conformal invariant angle between beliefs |

**Nonconformity scores** (`nonconformity` module):
- `absolute_residual(prediction, observation)` — |pred − obs|
- `normalized_residual(prediction, observation, scale)` — |pred − obs| / scale
- `rank_score(probabilities, true_class)` — classification nonconformity

---

## How It Works

### Architecture

```
Complex Plane
    │
    ├─→ MöbiusTransformation ──→ conformal automorphisms of Riemann sphere
    ├─→ ConformalMap          ──→ local angle-preserving maps via Jacobian
    │
Riemannian Manifold (n-dim)
    │
    ├─→ ConformalLaplacian    ──→ Yamabe operator (conformally covariant)
    ├─→ WeylTensor            ──→ conformally invariant curvature
    ├─→ ConformalWeight        ──→ Ω^Δ transformation rules
    ├─→ ConformalCompactification ──→ Rⁿ → Sⁿ via stereographic projection
    │
CFT / Virasoro
    │
    ├─→ PrimaryField           ──→ scaling dimension, 2-pt & 3-pt functions
    ├─→ VirasoroAlgebra        ──→ [Lₘ,Lₙ] with central charge
    │
Agent Learning
    │
    └─→ ConformalPrediction   ──→ distribution-free uncertainty quantification
```

### Key Algorithms

**Möbius composition**: (a₁,b₁,c₁,d₁) ∘ (a₂,b₂,c₂,d₂) = (a₁a₂+b₁c₂, a₁b₂+b₁d₂, c₁a₂+d₁c₂, c₁b₂+d₁d₂) — matrix multiplication.

**Conformal check (2D)**: A 2×2 matrix [[a,b],[c,d]] is conformal iff a=d and b=−c (Cauchy-Riemann).

**Conformal check (3D)**: MᵀM = λI — must be a scaled orthogonal matrix.

**Weyl extraction**: Subtract Ricci and scalar curvature traces from Riemann tensor. Result is conformally invariant.

**Conformal prediction**: Sort calibration nonconformity scores, take ⌈(n+1)(1−α)/n⌉-th quantile as threshold. Prediction set = {y : score(y) ≤ threshold}. Valid under exchangeability.

---

## The Math

### Möbius Transformations

f(z) = (az + b)/(cz + d), ad − bc ≠ 0

Group isomorphic to PGL(2,ℂ) = GL(2,ℂ)/{scalars}.

Classification by tr² = (a+d)²/(ad−bc):
- **Elliptic**: tr² ∈ [0, 4) — rotation
- **Parabolic**: tr² = 4 — translation
- **Hyperbolic**: tr² > 4 — dilation
- **Loxodromic**: tr² ∉ [0, 4] — general spiral

### Conformal Maps

f: U → V is conformal if f\*g = e²ᵖ g. In 2D, this is equivalent to f being holomorphic with f' ≠ 0 (Cauchy-Riemann).

**Liouville's theorem** (n ≥ 3): The only conformal maps of Rⁿ are Möbius transformations (compositions of translations, rotations, dilations, and inversions).

### Conformal Laplacian

L_g = −Δ_g + ((n−2)/(4(n−1))) R_g

Under g̃ = e²ᵖ g: L_g̃ = e^((n+2)/2)σ · L_g · e^((n−2)/2)σ

The **Yamabe problem**: find g̃ conformal to g such that R_g̃ is constant. Solved by minimizing the Yamabe functional Q(g) = ∫ L_g u · u dV / (∫ u^{2n/(n−2)})^{(n−2)/n}.

### Weyl Tensor

Cᵢⱼₖₗ = Rᵢⱼₖₗ − (1/(n−2))(gᵢₖRⱼₗ − gᵢₗRⱼₖ + gⱼₗRᵢₖ − gⱼₖRᵢₗ) + (R/((n−1)(n−2)))(gᵢₖgⱼₗ − gᵢₗgⱼₖ)

Properties:
- Same symmetries as Riemann tensor
- Trace-free: all contractions vanish
- **Conformally invariant**: C̃ᵢⱼₖₗ = Cᵢⱼₖₗ under g̃ = Ω²g
- C = 0 iff conformally flat (for n ≥ 4)

### Virasoro Algebra

[Lₘ, Lₙ] = (m−n)Lₘ₊ₙ + (c/12)(m³−m)δₘ₊ₙ,₀

The central term c/12·(m³−m) is the **conformal anomaly**. The Kac determinant at level n:

det(Mₙ) = ∏_{r·s≤n} (h − hᵣₛ(c))^{p(n−rs)}

where hᵣₛ(c) = ((cr − s)² − (c−1)²) / (4c) are the Kac zeros.

### Conformal Prediction

Given calibration scores s₁, ..., sₙ, prediction set at level 1−α:

C(x) = {y : s(x,y) ≤ q⌈(n+1)(1−α)/n⌉}

This is **distribution-free**: P(Y ∈ C(X)) ≥ 1−α under exchangeability, no matter the underlying distribution.

---

## License

MIT
