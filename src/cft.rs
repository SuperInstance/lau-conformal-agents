//! Conformal Field Theory (CFT) basics.
//!
//! A conformal field theory is a quantum field theory invariant under conformal
//! transformations. Key structures: the stress-energy tensor, operator product
//! expansions (OPE), scaling dimensions, and conformal bootstrap.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A primary field (operator) in a CFT, characterized by its scaling dimension
/// and spin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryField {
    /// Name/label for the field.
    pub name: String,
    /// Scaling dimension Δ (eigenvalue of the dilatation operator).
    pub scaling_dimension: f64,
    /// Spin s (integer or half-integer for fields on the plane).
    pub spin: f64,
    /// Conformal weight h = (Δ + s)/2.
    pub conformal_weight_h: f64,
    /// Anti-holomorphic weight h̄ = (Δ - s)/2.
    pub conformal_weight_hbar: f64,
}

impl PrimaryField {
    /// Create a new primary field.
    pub fn new(name: &str, scaling_dimension: f64, spin: f64) -> Self {
        let h = (scaling_dimension + spin) / 2.0;
        let hbar = (scaling_dimension - spin) / 2.0;
        Self {
            name: name.to_string(),
            scaling_dimension,
            spin,
            conformal_weight_h: h,
            conformal_weight_hbar: hbar,
        }
    }

    /// Two-point function of a primary field with itself:
    /// ⟨φ(x)φ(y)⟩ = 1 / |x - y|^{2Δ}
    pub fn two_point_function(&self, distance: f64) -> f64 {
        if distance < 1e-15 {
            f64::INFINITY
        } else {
            1.0 / distance.powf(2.0 * self.scaling_dimension)
        }
    }

    /// Three-point function coefficient (structure constant).
    /// ⟨φ₁(x₁)φ₂(x₂)φ₃(x₃)⟩ = C_{123} / (|x₁₂|^{Δ₁+Δ₂-Δ₃} |x₂₃|^{Δ₂+Δ₃-Δ₁} |x₁₃|^{Δ₁+Δ₃-Δ₂})
    pub fn three_point_function(
        fields: (&PrimaryField, &PrimaryField, &PrimaryField),
        distances: (f64, f64, f64), // (|x₁₂|, |x₂₃|, |x₁₃|)
        structure_constant: f64,
    ) -> f64 {
        let (f1, f2, f3) = fields;
        let (d12, d23, d13) = distances;
        let d1 = f1.scaling_dimension;
        let d2 = f2.scaling_dimension;
        let d3 = f3.scaling_dimension;

        let e12 = d1 + d2 - d3;
        let e23 = d2 + d3 - d1;
        let e13 = d1 + d3 - d2;

        structure_constant / (d12.powf(e12) * d23.powf(e23) * d13.powf(e13))
    }

    /// Unitarity bound: Δ ≥ |s| + n - 2 for n-dimensional CFT (for s ≥ 1).
    pub fn satisfies_unitarity_bound(&self, spacetime_dim: usize) -> bool {
        if self.spin >= 1.0 {
            self.scaling_dimension >= self.spin + spacetime_dim as f64 - 2.0 - 1e-10
        } else {
            self.scaling_dimension >= (spacetime_dim as f64 - 2.0) / 2.0 - 1e-10
        }
    }
}

/// The stress-energy tensor T_μν in a CFT.
/// It is traceless in a CFT: T^μ_μ = 0.
/// Its conformal weight is (d, 0) in 2D.
#[derive(Debug, Clone)]
pub struct StressEnergyTensor {
    /// Dimension of spacetime.
    pub spacetime_dimension: usize,
    /// Central charge c.
    pub central_charge: f64,
}

impl StressEnergyTensor {
    /// Create a stress-energy tensor for a d-dimensional CFT with central charge c.
    pub fn new(spacetime_dimension: usize, central_charge: f64) -> Self {
        Self { spacetime_dimension, central_charge }
    }

    /// The two-point function of T_{μν}:
    /// ⟨T_{μν}(x) T_{ρσ}(0)⟩ = c / |x|^{2d} I_{μν,ρσ}(x)
    /// where I is a known tensor structure.
    pub fn two_point_coefficient(&self, distance: f64) -> f64 {
        let d = self.spacetime_dimension as f64;
        self.central_charge / distance.powf(2.0 * d)
    }

    /// Trace of the stress-energy tensor (should be zero in a CFT).
    pub fn trace(&self, components: &DMatrix<f64>) -> f64 {
        let n = self.spacetime_dimension;
        let mut trace = 0.0;
        for i in 0..n.min(components.nrows()) {
            trace += components[(i, i)];
        }
        trace
    }

    /// Check if the stress-energy tensor is traceless.
    pub fn is_traceless(&self, components: &DMatrix<f64>, tol: f64) -> bool {
        self.trace(components).abs() < tol
    }

    /// The conformal Ward identity for the stress-energy tensor:
    /// ⟨T_{μν}(x) φ₁(x₁) ... φₙ(xₙ)⟩ is determined by the conformal weights
    /// and positions of the operators.
    pub fn ward_identity_weight(&self, field: &PrimaryField) -> f64 {
        field.scaling_dimension
    }
}

/// Operator Product Expansion (OPE).
/// φᵢ(x) φⱼ(0) ~ Σₖ C_{ijk} |x|^{Δₖ-Δᵢ-Δⱼ} [φₖ(0) + descendants]
#[derive(Debug, Clone)]
pub struct OPE {
    /// The two operators being expanded.
    pub op1: String,
    pub op2: String,
    /// Coefficients: (operator_name, structure_constant, scaling_dimension).
    pub terms: Vec<(String, f64, f64)>,
}

impl OPE {
    /// Create a new OPE.
    pub fn new(op1: &str, op2: &str) -> Self {
        Self {
            op1: op1.to_string(),
            op2: op2.to_string(),
            terms: Vec::new(),
        }
    }

    /// Add a term to the OPE.
    pub fn add_term(&mut self, name: &str, coefficient: f64, scaling_dim: f64) {
        self.terms.push((name.to_string(), coefficient, scaling_dim));
    }

    /// Compute the OPE coefficient at distance |x| for a given term.
    pub fn coefficient_at_distance(&self, term_index: usize, distance: f64, delta_i: f64, delta_j: f64) -> f64 {
        if let Some((_, c, delta_k)) = self.terms.get(term_index) {
            c * distance.powf(delta_k - delta_i - delta_j)
        } else {
            0.0
        }
    }

    /// Number of terms in the OPE.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
}

/// Conformal bootstrap constraints.
/// The crossing equation relates different channels of a four-point function.
pub struct ConformalBootstrap;

impl ConformalBootstrap {
    /// The crossing equation for four identical scalars:
    /// Σₖ (-1)^ℓ V_{Δ,ℓ}(u,v) = 0
    /// where u, v are conformal cross-ratios.
    pub fn crossing_equation_block(
        delta: f64,
        ell: f64,
        u: f64,
        v: f64,
    ) -> f64 {
        // Simplified conformal block contribution (scalar blocks in mean field theory)
        let block_u = u.powf(delta / 2.0);
        let block_v = v.powf(delta / 2.0);
        let sign = if ell as i64 % 2 == 0 { 1.0 } else { -1.0 };
        sign * (block_u - block_v)
    }

    /// Compute conformal cross-ratios u, v from four points.
    pub fn cross_ratios(x1: f64, x2: f64, x3: f64, x4: f64) -> (f64, f64) {
        let x12 = x1 - x2;
        let x34 = x3 - x4;
        let x13 = x1 - x3;
        let x24 = x2 - x4;
        let x14 = x1 - x4;
        let x23 = x2 - x3;
        let u = (x12 * x34) / (x13 * x24);
        let v = (x14 * x23) / (x13 * x24);
        (u, v)
    }
}

/// Anomalous dimension: the difference between the quantum scaling dimension
/// and the classical (engineering) dimension.
#[derive(Debug, Clone)]
pub struct AnomalousDimension {
    pub classical_dimension: f64,
    pub anomalous_part: f64,
}

impl AnomalousDimension {
    pub fn new(classical: f64, anomalous: f64) -> Self {
        Self { classical_dimension: classical, anomalous_part: anomalous }
    }

    /// The full scaling dimension.
    pub fn full_dimension(&self) -> f64 {
        self.classical_dimension + self.anomalous_part
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_primary_field_creation() {
        let phi = PrimaryField::new("φ", 1.0, 0.0);
        assert_relative_eq!(phi.scaling_dimension, 1.0);
        assert_relative_eq!(phi.conformal_weight_h, 0.5);
        assert_relative_eq!(phi.conformal_weight_hbar, 0.5);
    }

    #[test]
    fn test_primary_field_spinor() {
        let psi = PrimaryField::new("ψ", 1.5, 0.5);
        assert_relative_eq!(psi.conformal_weight_h, 1.0);
        assert_relative_eq!(psi.conformal_weight_hbar, 0.5);
    }

    #[test]
    fn test_two_point_function() {
        let phi = PrimaryField::new("φ", 1.0, 0.0);
        let g2 = phi.two_point_function(2.0);
        assert_relative_eq!(g2, 0.25); // 1/2^2
    }

    #[test]
    fn test_two_point_function_zero_distance() {
        let phi = PrimaryField::new("φ", 1.0, 0.0);
        assert!(phi.two_point_function(0.0).is_infinite());
    }

    #[test]
    fn test_three_point_function() {
        let phi1 = PrimaryField::new("φ₁", 1.0, 0.0);
        let phi2 = PrimaryField::new("φ₂", 1.0, 0.0);
        let phi3 = PrimaryField::new("φ₃", 1.0, 0.0);
        // When all Δ = 1: exponents are all 1
        let result = PrimaryField::three_point_function(
            (&phi1, &phi2, &phi3),
            (1.0, 1.0, 1.0),
            1.0,
        );
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unitarity_bound_scalar() {
        let phi = PrimaryField::new("φ", 0.5, 0.0);
        // Δ=0.5 < (4-2)/2 = 1, so does NOT satisfy
        assert!(!phi.satisfies_unitarity_bound(4));
    }

    #[test]
    fn test_unitarity_bound_scalar_satisfied() {
        let phi = PrimaryField::new("φ", 1.0, 0.0);
        assert!(phi.satisfies_unitarity_bound(4)); // Δ ≥ 1.0, yes
    }

    #[test]
    fn test_unitarity_bound_vector() {
        let j = PrimaryField::new("J", 3.0, 1.0);
        assert!(j.satisfies_unitarity_bound(4)); // Δ ≥ 1+4-2 = 3, yes
    }

    #[test]
    fn test_stress_energy_traceless() {
        let mut t = DMatrix::zeros(2, 2);
        t[(0, 0)] = 1.0;
        t[(1, 1)] = -1.0;
        let se = StressEnergyTensor::new(2, 1.0);
        assert!(se.is_traceless(&t, 1e-10));
    }

    #[test]
    fn test_stress_energy_not_traceless() {
        let mut t = DMatrix::zeros(2, 2);
        t[(0, 0)] = 1.0;
        t[(1, 1)] = 1.0;
        let se = StressEnergyTensor::new(2, 1.0);
        assert!(!se.is_traceless(&t, 1e-10));
    }

    #[test]
    fn test_stress_energy_two_point() {
        let se = StressEnergyTensor::new(2, 1.0);
        let coeff = se.two_point_coefficient(1.0);
        assert_relative_eq!(coeff, 1.0);
    }

    #[test]
    fn test_ope_construction() {
        let mut ope = OPE::new("φ", "φ");
        ope.add_term("I", 1.0, 0.0);
        ope.add_term("φ", 1.0, 1.0);
        ope.add_term("T", 1.0, 2.0);
        assert_eq!(ope.num_terms(), 3);
    }

    #[test]
    fn test_ope_coefficient() {
        let mut ope = OPE::new("φ", "φ");
        ope.add_term("I", 1.0, 0.0);
        // φ(x)φ(0) ~ |x|^{0-1-1} I = |x|^{-2} I
        let coeff = ope.coefficient_at_distance(0, 2.0, 1.0, 1.0);
        assert_relative_eq!(coeff, 0.25); // 2^{-2}
    }

    #[test]
    fn test_cross_ratios() {
        let (u, v) = ConformalBootstrap::cross_ratios(0.0, 1.0, 2.0, 3.0);
        assert!(u.is_finite());
        assert!(v.is_finite());
    }

    #[test]
    fn test_crossing_block() {
        let block = ConformalBootstrap::crossing_equation_block(2.0, 0.0, 0.5, 0.5);
        assert!(block.is_finite());
    }

    #[test]
    fn test_anomalous_dimension() {
        let ad = AnomalousDimension::new(1.0, 0.1);
        assert_relative_eq!(ad.full_dimension(), 1.1);
    }

    #[test]
    fn test_anomalous_dimension_zero() {
        let ad = AnomalousDimension::new(2.0, 0.0);
        assert_relative_eq!(ad.full_dimension(), 2.0);
    }
}
