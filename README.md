# lau-conformal-agents

**Conformal geometry for agent systems — transformations that preserve angles but not necessarily distances.**

Conformal maps preserve local shape (angles) while allowing global deformation (scaling). For agents, conformal maps of belief space change the scale of uncertainty but preserve the structure of beliefs. This crate implements the full machinery: Möbius transformations, the Weyl tensor, conformal Laplacian, Virasoro algebra, CFT correlators, and conformal prediction for uncertainty quantification.

165 tests · MIT license · pure Rust · zero unsafe

---

## What This Does

| Module | Concept | What you get |
|---|---|---|
| `mobius` | f(z) = (az+b)/(cz+d) on the Riemann sphere | Composition, fixed points, classification, cross-ratio |
| `conformal_map` | Angle-preserving maps | Jacobian analysis, Schwarzian derivative, generalized circles |
| `conformal_weight` | How fields transform under rescaling | Invariant, fixed-weight, dimension-dependent weights |
| `weyl` | Conformal curvature tensor | Weyl tensor from Riemann/Ricci/scalar, Cotton tensor, conformal flatness |
| `liouville` | Liouville's theorem | Conformal classification by dimension, conformal Killing equation |
| `conformal_laplacian` | Yamabe operator L = −Δ + c(n)R | Scalar curvature coupling, Yamabe constant, conformal transformation law |
| `compactification` | Adding a point at infinity | Stereographic projection, one-point compactification, round metric |
| `cft` | Conformal field theory | Primary fields, OPE, stress-energy tensor, bootstrap bounds |
| `virasoro` | Central extension of the Witt algebra | Commutators, Verma modules, Kac determinant, unitarity |
| `agent_learning` | Learning with conformal geometry | Conformal prediction intervals, belief rescaling, invariant features |

---

## Key Idea

> In 2D, the conformal group is infinite-dimensional (any holomorphic function). In higher dimensions, Liouville's theorem restricts it to Möbius transformations. This crate bridges the gap: it implements the full 2D conformal algebra (Virasoro) *and* the Riemannian conformal geometry (Weyl tensor, conformal Laplacian) needed for higher dimensions, all applied to agent belief spaces.

The most practical piece is **conformal prediction**: given calibration data, produce prediction sets with guaranteed coverage probability — no distributional assumptions needed.

---

## Install

```toml
[dependencies]
lau-conformal-agents = { git = "https://github.com/SuperInstance/lau-conformal-agents" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra`, `num-complex`, `serde`, `approx`.

---

## Quick Start

### Möbius transformations

```rust
use lau_conformal_agents::{MobiusTransformation, cross_ratio};
use num_complex::Complex64;

let c = Complex64::new;

// f(z) = (z + 1) / (z - 1)
let mob = MobiusTransformation::new(c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0));

let z = c(2.0, 0.0);
let w = mob.evaluate(z);
assert_eq!(w.unwrap(), c(3.0, 0.0));

// Classification: parabolic, elliptic, hyperbolic, loxodromic
println!("type: {:?}", mob.classify());

// Cross-ratio (conformal invariant)
let cr = cross_ratio(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0));
```

### Conformal prediction (uncertainty quantification)

```rust
use lau_conformal_agents::{conformal_prediction_interval, nonconformity};

let calibration = vec![0.5, 1.2, 0.8, 1.5, 0.3, 2.1, 0.9];
let prediction = 5.0;
let alpha = 0.1; // 90% coverage

let (lower, upper) = conformal_prediction_interval(&calibration, alpha, prediction);
println!("90% prediction interval: [{:.2}, {:.2}]", lower, upper);
```

### CFT correlators

```rust
use lau_conformal_agents::{PrimaryField, StressEnergyTensor};

let phi = PrimaryField::new("φ", 1.0, 0.0);  // Δ=1, spin=0
let two_pt = phi.two_point_function(2.0);      // 1/|x-y|^{2Δ}
println!("⟨φ(x)φ(y)⟩ = {:.4}", two_pt);

let T = StressEnergyTensor::new(2, 1.0);       // d=2, c=1
let ward_weight = T.ward_identity_weight(&phi); // Conformal Ward identity
```

### Virasoro algebra

```rust
use lau_conformal_agents::{VirasoroAlgebra, VermaModule};

let vir = VirasoroAlgebra::new(1.0); // c = 1
let (lie_coeff, central) = vir.commutator(3, -3);
println!("[L₃, L₋₃] = {} L₀ + {}", lie_coeff, central);

let verma = VermaModule::new(1.0, 0.5); // c=1, h=1/2
println!("reducible? {}", verma.is_reducible(5));

let kac = vir.kac_determinant(2, 0.5);
println!("Kac det at level 2: {:.4}", kac);
```

---

## API Reference

### Möbius Transformations

**`MobiusTransformation`** — f(z) = (az + b)/(cz + d), ad − bc ≠ 0.
- `MobiusTransformation::new(a, b, c, d)` — panics if degenerate.
- `MobiusTransformation::try_new(a, b, c, d)` → `Option<Self>`.
- `MobiusTransformation::identity()` — f(z) = z.
- `mob.evaluate(z)` → `Option<Complex64>` — None if z maps to ∞.
- `mob.compose(&other)` — composition of two maps.
- `mob.inverse()` — inverse transformation.
- `mob.fixed_points()` → `Vec<Complex64>` — solve f(z) = z.
- `mob.trace()` — tr of the coefficient matrix (a + d).
- `mob.classify()` → `MobiusType` — Parabolic / Elliptic / Hyperbolic / Loxodromic.
- `mob.multiplier()` — derivative at the attracting fixed point.
- `mob.normalize()` — rescale so det = 1.

**Constructors**:
- `translation(b)` — f(z) = z + b.
- `dilation(lambda)` — f(z) = λz.
- `rotation(theta)` — f(z) = e^{iθ} z.
- `inversion()` — f(z) = 1/z.

**Utilities**:
- `cross_ratio(z1, z2, z3, z4)` — the conformal invariant (z₁z₃)(z₂z₄)/((z₁z₂)(z₃z₄)).
- `map_three_points_to_0_1_inf(z1, z2, z3)` — unique Möbius map.
- `map_three_points(src, dst)` — map any triple to any triple.

### Conformal Maps

**`ConformalMap`** — angle-preserving map with Jacobian analysis.
- `ConformalMap::from_complex_derivative(z, f_prime)` — build from f'(z).
- `map.conformal_factor()` — |f'(z)|, the local scale factor.
- `map.is_conformal()` — check Cauchy–Riemann conditions.
- `map.pullback_metric_scale()` — e^{2σ} for the metric rescaling.

**Elementary conformal maps**:
- `exponential_map(z)` → e^z.
- `logarithm_map(w)` → log w.
- `power_map(z, alpha)` → z^α.
- `conformal_inversion(z)` → 1/z.
- `compose(f, g, z)` — functional composition at a point.

**`schwarzian_derivative(f, f_prime, f_double_prime, z)`** — the S[f](z) = f'''/f' − (3/2)(f''/f')² invariant that detects non-Möbius conformal maps.

**`GeneralizedCircle`** — circles and lines in the complex plane (preserved by Möbius maps).
- `GeneralizedCircle::circle(center, radius)`.
- `GeneralizedCircle::line(a, b, c)` — ax + by + c = 0.
- `circle.contains(z, tol)` / `circle.is_line()` / `circle.is_circle()`.
- `circle_through_three_points(z1, z2, z3)`.

### Conformal Weight

**`ConformalWeight`** — how a field transforms under g̃ = Ω²g.
- `Invariant` — weight 0 (doesn't change).
- `Fixed(w)` — transforms as Ω^w.
- `DimensionDependent(f)` — weight depends on manifold dimension.
- `weight.transform(value, omega, dim)` — apply the transformation.

**`ConformalWeights`** — common weights: metric (2), inverse metric (−2), volume form (n), scalar curvature (−2).

### Weyl Tensor

**`WeylTensor`** — the trace-free part of Riemann curvature, invariant under conformal changes.
- `WeylTensor::zeros(n)` — zero tensor in n dimensions.
- `WeylTensor::from_riemann(&riemann, &ricci, scalar_curvature, &metric)` — compute from Riemannian data.
- `weyl.get(i, j, k, l)` / `weyl.set(i, j, k, l, val)`.
- `weyl.is_zero(tol)` — vanishing implies conformal flatness (n ≥ 4).
- `weyl.independent_components(n)` — count of free components.
- Symmetry verifiers: `verify_antisymmetry_ij`, `verify_antisymmetry_kl`, `verify_pair_symmetry`, `verify_trace_free`.

**`CottonTensor`** — the (n−1)-form obstruction to conformal flatness in n = 3.
- `CottonTensor::from_ricci_gradient(...)`.
- `cotton.is_zero(tol)` — vanishing implies conformal flatness in 3D.

**`is_conformally_flat_weyl(&weyl)`** / **`is_conformally_flat_cotton(&cotton)`**.

### Liouville's Theorem

- `is_conformal_matrix_2d(&matrix)` — Cauchy–Riemann check (a = d, b = −c).
- `is_conformal_matrix_3d(&matrix)` — M^T M = λI check.
- `classify_conformal_maps(dim)` → `ConformalClassification` — describes which maps are conformal in each dimension.
- `conformal_killing_equation(&jacobian, &metric)` — check the conformal Killing equation ∇ᵢξⱼ + ∇ⱼξᵢ = (2/n)(∇·ξ)gᵢⱼ.
- `conformal_group_dimension(n)` — dim of the conformal group = (n+1)(n+2)/2.

### Conformal Laplacian

**`ConformalLaplacian`** — L_g = −Δ + ((n−2)/(4(n−1))) R.
- `ConformalLaplacian::new(dimension, scalar_curvature)`.
- `lap.curvature_coefficient()` — (n−2)/(4(n−1)).
- `lap.apply(laplacian_f, f)` — compute L_g f.
- `lap.yamabe_constant_estimate(&stiffness, &mass)` — smallest eigenvalue (finite-element approximation).

### Compactification

**`StereographicProjection`** — map S^n → ℝ^n.
- `StereographicProjection::north(dimension)` / `south(dimension)`.
- `proj.project(&point_on_sphere)` → `Option<DVector<f64>>`.
- `proj.inverse(&point_on_rn)` → `DVector<f64>`.

**`ConformalCompactification`** — add ∞ to make a manifold compact.
- `compactify(&points)` — map through stereographic projection.
- `one_point_compactify(&point, dimension)`.
- `infinity_point(dimension)` — the representative point at ∞.
- `round_metric_stereographic(&u)` / `round_metric_conformal_factor(&u)`.

### CFT

**`PrimaryField`** — a field characterized by scaling dimension Δ and spin s.
- `PrimaryField::new(name, scaling_dimension, spin)`.
- `field.two_point_function(distance)` — ⟨φ(x)φ(y)⟩ = 1/|x−y|^{2Δ}.
- `field.three_point_function((f1, f2, f3), (d12, d23, d13), C_{123})`.
- `field.satisfies_unitarity_bound(spacetime_dim)` — Δ ≥ s + (n−2)/2.

**`StressEnergyTensor`** — the generator of conformal transformations.
- `StressEnergyTensor::new(spacetime_dim, central_charge)`.
- `T.two_point_coefficient(distance)` — ⟨T(x)T(y)⟩ ~ c/|x−y|^{2d}.
- `T.is_traceless(&components, tol)` — T^μ_μ = 0 in CFT.
- `T.ward_identity_weight(&field)` — the conformal Ward identity coefficient.

**`OPE`** — operator product expansion.
- `OPE::new("φ₁", "φ₂")`.
- `ope.add_term(name, coefficient, scaling_dim)`.
- `ope.coefficient_at_distance(idx, dist, Δᵢ, Δⱼ)`.

### Virasoro Algebra

**`VirasoroAlgebra`** — [Lₘ, Lₙ] = (m−n)L_{m+n} + (c/12)(m³−m)δ_{m+n,0}.
- `VirasoroAlgebra::new(central_charge)`.
- `VirasoroAlgebra::witt()` — c = 0 (no central extension).
- `vir.commutator(m, n)` → `(lie_bracket_coefficient, central_term)`.
- `vir.central_term(m)` — (c/12)(m³ − m).
- `vir.verify_jacobi(k, m, n)` — check [Lₖ,[Lₘ,Lₙ]] + cyclic = 0.
- `vir.kac_determinant(level, h)` — the Kac determinant at given level.
- `vir.effective_central_charge(h₀)` — c_eff = c − 24h₀.
- `vir.is_unitary(h)` — check the unitarity bounds.

**`VermaModule`** — highest-weight representation.
- `VermaModule::new(c, h)` — with highest weight h.
- `verma.l0_eigenvalue(level)` — h + level.
- `verma.level_dimension(level)` — number of states at that level.
- `verma.is_reducible(max_level)` — if the Kac determinant vanishes.

**`sugawara_central_charge(level_k, dim_g, dual_coxeter)`** — c from the Sugawara construction.
**`casimir_energy(c)`** — the ground-state energy −c/24.
**`character_q_expansion(c, h, max_order)`** — Virasoro character as power series in q.

### Agent Learning

**`conformal_prediction_interval(&calibration_scores, alpha, prediction)` → `(lower, upper)`** — distribution-free prediction interval with (1−α) coverage guarantee.

**Nonconformity scores**:
- `absolute_residual(prediction, observation)`.
- `normalized_residual(prediction, observation, scale)`.
- `rank_score(&probabilities, true_class)`.

**`conformal_rescale_belief(&belief, factor)`** — rescale belief vector conformally.
**`belief_angle(&b1, &b2)`** — angle between belief vectors (conformal invariant).
**`conformal_invariant_features(&beliefs)`** — extract angle-based invariants.

**`AdaptiveConformal`** — online conformal prediction that adapts to distribution shift:
- `AdaptiveConformal::new(target_alpha, gamma)`.
- `ac.update(covered)` — update the threshold after each observation.

---

## How It Works

### Möbius Transformations

The Möbius group is the set of conformal automorphisms of the Riemann sphere ℂ ∪ {∞}. Every Möbius map can be decomposed into translations, dilations, rotations, and inversion. The **cross-ratio** of four points is invariant under all Möbius maps.

### Conformal Flatness

A manifold (M, g) is **conformally flat** if there exists a function σ such that g̃ = e^{2σ}g is flat. In n ≥ 4, this is equivalent to the Weyl tensor vanishing. In n = 3, it's equivalent to the Cotton tensor vanishing. In n = 2, every metric is conformally flat (uniformization theorem).

### Virasoro Algebra

The conformal algebra in 2D has generators Lₙ (modes of the stress-energy tensor) satisfying [Lₘ, Lₙ] = (m−n)L_{m+n}. Quantization introduces the central charge c, giving the Virasoro algebra. The **Kac determinant** detects null states (reducibility) in Verma modules. **Unitarity** requires c ≥ 1 and h ≥ 0 (with exceptions at c < 1 in the minimal model series).

### CFT Correlators

In a CFT, the n-point correlation functions are fixed up to constants by conformal symmetry. The two-point function ⟨φᵢ(x)φⱼ(y)⟩ = δᵢⱼ/|x−y|^{2Δᵢ}. The three-point function ⟨φᵢφⱼφₖ⟩ = Cᵢⱼₖ/(|x₁₂|^{...}|x₂₃|^{...}|x₁₃|^{...}). The **operator product expansion** (OPE) gives the singular part of the product of two operators as they approach each other.

### Conformal Prediction

Given calibration scores s₁, ..., sₙ and a new prediction, the conformal prediction set is {y : s(x, y) ≤ q} where q is the ⌈(1−α)(n+1)⌉/n quantile of calibration scores. This guarantees P(Y ∈ C(X)) ≥ 1−α under exchangeability — no distributional assumptions needed.

---

## The Math

### Möbius Transformations

f(z) = (az+b)/(cz+d) with ad−bc ≠ 0. These form a group isomorphic to PGL(2, ℂ). The **trace** tr = a+d classifies: |tr| = 2 → parabolic, |tr| < 2 → elliptic, |tr| > 2 → hyperbolic, else → loxodromic.

### Weyl Tensor

C_{ijkl} = R_{ijkl} − (1/(n−2))(g_{ik}R_{jl} − g_{il}R_{jk} + g_{jl}R_{ik} − g_{jk}R_{il}) + R/((n−1)(n−2))(g_{ik}g_{jl} − g_{il}g_{jk}). Properties: same symmetries as Riemann, trace-free on all contractions, invariant under conformal rescaling.

### Conformal Laplacian

L_g = −Δ_g + ((n−2)/(4(n−1)))R_g transforms as L_{e^{2σ}g} φ = e^{−((n+2)/2)σ} L_g (e^{((n−2)/2)σ} φ). The **Yamabe problem** asks: does there exist a conformal metric with constant scalar curvature? The answer depends on the sign of the Yamabe constant (smallest eigenvalue of L).

### Virasoro Commutator

[Lₘ, Lₙ] = (m−n)L_{m+n} + (c/12)(m³−m)δ_{m+n,0}. The central term (c/12)(m³−m) is the **anomaly**: it vanishes classically but appears upon quantization. The central charge c measures the "number of degrees of freedom" of the CFT.

### Kac Determinant

At level N in the Verma module V(c, h), the Kac determinant is det M_N = ∏_{rs≤N} (h − h_{r,s}(c))^{p(N−rs)} where h_{r,s}(c) are the Kac zeros. When it vanishes, the module has a null state and is reducible.

### Conformal Prediction Coverage

Under exchangeability of (X₁, Y₁), ..., (Xₙ, Yₙ), (X, Y), the conformal prediction set C(X) satisfies P(Y ∈ C(X)) ≥ 1−α. The proof uses the fact that the rank of the test score among all n+1 scores is uniformly distributed.

---

## License

MIT
