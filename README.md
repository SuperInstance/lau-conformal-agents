# lau-conformal-agents

**Conformal geometry for agent systems** — Möbius transformations, Liouville's theorem, the Weyl tensor, the conformal Laplacian (Yamabe operator), Weyl rescalings, conformal weights, stereographic compactification, 2D conformal field theory (CFT), the Virasoro algebra, and conformal prediction for uncertainty quantification.

## What This Does

A **conformal map** preserves angles but not distances — it's a "shape-preserving" deformation. In 2D, these are the holomorphic functions with nonzero derivative (and their complex conjugates). In dimensions ≥ 3, Liouville's theorem tells us the *only* conformal maps are Möbius transformations. This crate makes conformal geometry computational:

- **Möbius transformations** — f(z) = (az+b)/(cz+d) on the Riemann sphere, with composition, inverse, fixed points, cross-ratio
- **Conformal maps** — Complex derivative → Jacobian → conformal factor, pullback metric, Cauchy-Riemann checks
- **Liouville's theorem** — Classify conformal maps by dimension; check if matrices are conformal
- **Weyl tensor** — The conformally invariant curvature tensor: C = Riemann − trace terms. Vanishes iff conformally flat
- **Conformal Laplacian** — The Yamabe operator L = −Δ + ((n−2)/(4(n−1)))R, which transforms covariantly under conformal changes
- **Weyl rescalings** — g̃ = Ω²g, compute transformed curvature, volume, Christoffel symbols
- **Conformal weights** — How fields transform under rescaling: φ̃ = Ω^Δ φ
- **Stereographic projection** — Compactify ℝⁿ to Sⁿ (add point at infinity)
- **CFT** — Primary fields, two/three-point functions, scaling dimensions, OPE
- **Virasoro algebra** — The central extension of the Witt algebra: [L_m, L_n] = (m−n)L_{m+n} + (c/12)(m³−m)δ_{m+n,0}
- **Conformal prediction** — Distribution-free uncertainty quantification using nonconformity scores

## Key Idea

Conformal geometry is the geometry of angle-preserving transformations. In physics, conformal symmetry is the symmetry of scale-invariant systems (critical points, the early universe). In agent systems, conformal maps of belief space change the scale of uncertainty while preserving the relational structure between beliefs. The conformal prediction module provides rigorous, distribution-free prediction intervals.

## Install

```toml
[dependencies]
lau-conformal-agents = "0.1.0"
```

## Quick Start

```rust
use lau_conformal_agents::*;
use num_complex::Complex64;

// Möbius transformation: inversion z ↦ 1/z
let inv = MobiusTransformation::new(
    Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
    Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
);
let z = Complex64::new(2.0, 0.0);
println!("1/2 = {:?}", inv.evaluate(z)); // Some(0.5 + 0i)

// Check if a matrix is conformal (scaled rotation)
use nalgebra::DMatrix;
let rot = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]); // 90° rotation
println!("Is conformal: {}", liouville::is_conformal_matrix_2d(&rot)); // true

// Conformal prediction
let calibration_scores = vec![0.1, 0.3, 0.2, 0.5, 0.4, 0.15, 0.35];
let (lo, hi) = agent_learning::conformal_prediction_interval(
    &calibration_scores, 0.1, 5.0
);
println!("90% prediction interval: [{:.2}, {:.2}]", lo, hi);

// Virasoro algebra with central charge c = 1
let vir = VirasoroAlgebra::new(1.0);
let (coeff, central) = vir.commutator(3, -3);
println!("[L_3, L_{-3}] = {} L_0 + {}", coeff, central);
```

## API Reference

### `mobius` — Möbius Transformations

| Type | Description |
|------|-------------|
| `MobiusTransformation` | f(z) = (az+b)/(cz+d) with complex coefficients. |

**Methods:**
- `new(a, b, c, d)`, `try_new(a, b, c, d) → Option<Self>`, `identity()`
- `evaluate(z) → Option<Complex64>` — Evaluate at z (None if z → ∞)
- `inverse() → Self` — f⁻¹(z) = (dz−b)/(−cz+a)
- `compose(&other) → Self` — Composition f∘g
- `determinant() → Complex64` — ad − bc
- `fixed_points() → Vec<Complex64>` — Points where f(z) = z
- `is_elliptic() → bool` — |tr|² ∈ [0, 4): rotation
- `is_hyperbolic() → bool` — |tr|² > 4: dilation
- `is_parabolic() → bool` — |tr|² = 4: translation
- `cross_ratio(z1, z2, z3, z4) → Complex64` — Möbius-invariant
- `translation(w)`, `dilation(λ)`, `rotation(θ)`, `inversion()` — Presets

---

### `conformal_map` — General Conformal Maps

| Type | Description |
|------|-------------|
| `ConformalMap` | A conformal map at a point: Jacobian + conformal factor. |

**Methods:**
- `from_complex_derivative(z, f_prime) → Self`
- `conformal_factor() → f64` — Scale factor |f'(z)|
- `jacobian() → &DMatrix<f64>` — 2×2 Jacobian
- `is_conformal() → bool` — Check Cauchy-Riemann (a=d, b=−c)
- `pullback_metric_scale() → f64` — e^{2σ} = |f'|²

**Predefined maps:**
- `exponential_map(z) → ConformalMap` — w = e^z
- `logarithm_map(z) → ConformalMap` — w = ln(z)
- `power_map(z, n) → ConformalMap` — w = z^n

---

### `liouville` — Liouville's Theorem

**Free functions:**
- `is_conformal_matrix_2d(&matrix) → bool` — Check Cauchy-Riemann conditions
- `is_conformal_matrix_3d(&matrix) → bool` — Check M^T M = λI
- `classify_conformal_maps(dim) → ConformalClassification` — What conformal maps exist in each dimension

| Type | Description |
|------|-------------|
| `Dimension` | Enum: Two, Three, Four, General(n). |
| `ConformalClassification` | Result: dimension, is_conformal, description. |

---

### `weyl` — Weyl Tensor (Conformal Curvature)

| Type | Description |
|------|-------------|
| `WeylTensor` | The conformally invariant part of Riemann curvature. n⁴ components. |

**Methods:**
- `zeros(n)`, `from_riemann(&riemann, &ricci, scalar_curvature, &metric)`
- `component(i, j, k, l) → f64`, `set(i, j, k, l, value)`
- `is_zero(tol) → bool` — Vanishes iff conformally flat (n ≥ 4)
- `norm() → f64` — Frobenius norm
- `schouten_tensor(&ricci, scalar_curvature, &metric) → DMatrix<f64>` — Schouten = Ricci − (R/2(n−1))g
- `is_conformally_flat(&riemann, &ricci, scalar_curvature, &metric) → bool`

---

### `conformal_laplacian` — Conformal Laplacian (Yamabe Operator)

| Type | Description |
|------|-------------|
| `ConformalLaplacian` | L_g = −Δ_g + ((n−2)/(4(n−1))) R_g. |

**Methods:**
- `new(dimension, scalar_curvature)`
- `curvature_coefficient() → f64` — (n−2)/(4(n−1))
- `apply(laplacian_f, f) → f64` — L_g f = −Δf + c(n)Rf
- `yamabe_constant_estimate(&stiffness, &mass) → Option<f64>` — Smallest eigenvalue
- `conformal_transform(sigma) → f64` — Transform L under g̃ = e^{2σ}g

---

### `weyl` (module) — Weyl Rescalings

| Type | Description |
|------|-------------|
| `WeylRescaling` | A conformal change g̃ = Ω²g. |

**Methods:**
- `new(omega)`, `from_function(sigma)` — Ω = e^σ
- `transform_metric(&g) → DMatrix<f64>` — g̃ = Ω²g
- `transform_scalar_curvature(&R, &sigma, &laplacian_sigma, dim) → f64`
- `transform_volume(&dV, dim) → f64` — dṼ = Ω^n dV
- `transform_christoffel(&gamma, &d_omega, &g_inv, dim) → DMatrix<f64>`

---

### `conformal_weight` — Conformal Weights

| Type | Description |
|------|-------------|
| `ConformalWeight` | Enum: Invariant, Fixed(Δ), DimensionDependent(f(n)). |
| `ConformalWeights` | Common weights: metric(2), inverse_metric(−2), volume_form(n), scalar_curvature(−2), christoffel(−1). |

**`ConformalWeight` methods:**
- `weight(dim) → f64` — Numerical weight in given dimension
- `transform(value, omega, dim) → f64` — Apply Ω^Δ · value

---

### `compactification` — Stereographic Projection

| Type | Description |
|------|-------------|
| `StereographicProjection` | Project S^n → ℝ^n (or inverse). |

**Methods:**
- `north(dimension)`, `south(dimension)`
- `project(&point) → Option<DVector<f64>>` — S^n → ℝ^n (None at pole)
- `inverse(&u) → DVector<f64>` — ℝ^n → S^n

**Free functions:**
- `inverse_stereographic(u, dimension) → DVector<f64>` — Map ℝ^n → S^n ⊂ ℝ^{n+1}
- `compactify(&point, dimension) → Option<DVector<f64>>` — Add point at infinity

---

### `cft` — Conformal Field Theory

| Type | Description |
|------|-------------|
| `PrimaryField` | A CFT primary: name, scaling dimension Δ, spin s, weights h = (Δ+s)/2, h̄ = (Δ−s)/2. |
| `OperatorProductExpansion` | OPE: φ_i(x)φ_j(0) ~ Σ_k C_{ijk} |x|^{Δ_k−Δ_i−Δ_j} φ_k(0). |
| `StressEnergyTensor` | Central charge c, holomorphic weight. |

**`PrimaryField`:**
- `new(name, Δ, s)`, `two_point_function(distance) → f64` — 1/|x|^{2Δ}
- `three_point_function(fields, distances, C_123) → f64`
- `scaling_dimension() → f64`, `spin() → f64`

**`OperatorProductExpansion`:**
- `new()`, `add_term(operator_index, coefficient, dimension)`
- `compute(phi_i, phi_j, distance) → Vec<(usize, f64)>`

---

### `virasoro` — Virasoro Algebra

| Type | Description |
|------|-------------|
| `VirasoroAlgebra` | The central extension of Witt: generators L_n with central charge c. |

**Methods:**
- `new(central_charge)`, `witt()` — c = 0
- `commutator(m, n) → (i64, f64)` — [L_m, L_n] = (m−n)L_{m+n} + central term
- `central_term(m) → f64` — (c/12)(m³−m) for [L_m, L_{−m}]
- `verify_jacobi(k, m, n) → f64` — Jacobi identity violation (should be 0)
- `highest_weight_state(h, level) → Vec<f64>` — L_0 eigenvalue h, descendant levels
- `character(h, q) → f64` — Virasoro character χ(q) = q^{h−c/24}/η(q)

---

### `agent_learning` — Conformal Prediction for Agents

| Type | Description |
|------|-------------|
| `ConformalPredictionSet<T>` | A prediction set with confidence, scores, and threshold. |

**Free functions:**
- `conformal_prediction_interval(&calibration_scores, α, prediction) → (lo, hi)` — Distribution-free interval
- `nonconformity::absolute_residual(pred, obs) → f64`
- `nonconformity::normalized_residual(pred, obs, scale) → f64`
- `nonconformity::rank_score(&probs, true_class) → f64`

## How It Works

1. **Möbius transformations**: Represented as 2×2 complex matrices [[a,b],[c,d]] with det ≠ 0. Composition is matrix multiplication. The group PGL(2,ℂ) acts on the Riemann sphere ℂ∪{∞}.

2. **Conformal maps**: In 2D, the complex derivative f'(z) encodes the conformal map — the Jacobian is a scaled rotation [[Re f', −Im f'],[Im f', Re f']]. The conformal factor is |f'(z)|.

3. **Liouville's theorem**: In n ≥ 3, only Möbius transformations are conformal. This is checked by verifying M^T M = λI (the map is a scaled orthogonal transformation).

4. **Weyl tensor**: The trace-free part of Riemann curvature: C = R − (Schouten terms). It's invariant under conformal changes: C̃ = C. Vanishes iff the manifold is conformally flat.

5. **Conformal Laplacian**: L = −Δ + c(n)R transforms covariantly under g̃ = Ω²g. Its Yamabe constant (smallest eigenvalue) is a conformal invariant.

6. **CFT**: Primary fields are characterized by (Δ, s). Two-point functions are power laws. Three-point functions involve structure constants. The OPE encodes the algebra of fields.

7. **Virasoro**: The infinite-dimensional Lie algebra [L_m, L_n] = (m−n)L_{m+n} + (c/12)(m³−m)δ_{m+n,0}. The central charge c is the quantum anomaly of conformal symmetry.

8. **Conformal prediction**: Given calibration nonconformity scores and a significance level α, compute the (1−α) quantile and form prediction intervals. This is distribution-free and valid for any underlying distribution.

## The Math

### Conformal Maps

A map f: (M,g) → (N,h) is conformal if f*g = e^{2σ}g for some function σ. In 2D, by the Cauchy-Riemann equations, these are exactly the holomorphic/antiholomorphic functions with nonzero derivative.

### Möbius Transformations

f(z) = (az+b)/(cz+d) form the automorphism group of the Riemann sphere. They are classified by tr²/ad−bc: elliptic (rotation), hyperbolic (dilation), parabolic (translation). The cross-ratio (z₁,z₂;z₃,z₄) is Möbius-invariant.

### Weyl Tensor

C_{ijkl} = R_{ijkl} − (1/(n−2))(g_{ik}R_{jl} − g_{il}R_{jk} + g_{jl}R_{ik} − g_{jk}R_{il}) + R/((n−1)(n−2))(g_{ik}g_{jl} − g_{il}g_{jk})

This is the only part of Riemann curvature that survives conformal transformations.

### Virasoro Algebra

The Witt algebra [L_m, L_n] = (m−n)L_{m+n} (vector fields ℓ_n = −z^{n+1}∂_z on S¹) has a unique central extension: the Virasoro algebra with central charge c. This is the symmetry algebra of 2D CFT.

### Conformal Prediction

Given exchangeable observations (Z₁,...,Zₙ,Z_{n+1}) and nonconformity scores αᵢ, the prediction set {z : α_{n+1} ≤ q_{1−α}} has coverage P(Z_{n+1} ∈ Ĉₙ) ≥ 1−α, valid for any distribution.

## License

MIT
