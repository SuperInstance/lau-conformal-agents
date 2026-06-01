//! Conformal weight and rescaling.
//!
//! A field φ has conformal weight Δ if under g̃ = Ω²g, the field transforms as
//! φ̃ = Ω^Δ φ. The conformal weight determines how quantities transform under
//! conformal changes of metric.

use nalgebra::DMatrix;

/// Conformal weight of a geometric quantity.
#[derive(Debug, Clone, Copy)]
pub enum ConformalWeight {
    /// The field is invariant (weight 0).
    Invariant,
    /// Fixed conformal weight Δ: transforms as φ̃ = Ω^Δ φ.
    Fixed(f64),
    /// Depends on the dimension: Δ = f(n).
    DimensionDependent(fn(usize) -> f64),
}

impl ConformalWeight {
    /// Get the numerical weight in a given dimension.
    pub fn weight(&self, dim: usize) -> f64 {
        match self {
            ConformalWeight::Invariant => 0.0,
            ConformalWeight::Fixed(w) => *w,
            ConformalWeight::DimensionDependent(f) => f(dim),
        }
    }

    /// Transform a field value under conformal rescaling g̃ = Ω²g.
    pub fn transform(&self, value: f64, omega: f64, dim: usize) -> f64 {
        let w = self.weight(dim);
        omega.powf(w) * value
    }
}

/// Common conformal weights for geometric quantities.
pub struct ConformalWeights;

impl ConformalWeights {
    /// Metric tensor: g̃_{ij} = Ω² g_{ij}, weight = 2.
    pub fn metric() -> ConformalWeight {
        ConformalWeight::Fixed(2.0)
    }

    /// Inverse metric: g̃^{ij} = Ω^{-2} g^{ij}, weight = -2.
    pub fn inverse_metric() -> ConformalWeight {
        ConformalWeight::Fixed(-2.0)
    }

    /// Volume form: dṼ = Ω^n dV, weight = n.
    pub fn volume_form() -> ConformalWeight {
        ConformalWeight::DimensionDependent(|n| n as f64)
    }

    /// Scalar curvature: transforms in a complex way, but the transformation formula involves
    /// the conformal Laplacian. Weight is not a simple power law.
    pub fn scalar_curvature() -> ConformalWeight {
        ConformalWeight::Fixed(-2.0) // leading order
    }

    /// Christoffel symbols: not a tensor, but transforms with weight -1.
    pub fn christoffel() -> ConformalWeight {
        ConformalWeight::Fixed(-1.0)
    }

    /// Weyl tensor: invariant under conformal transformations.
    pub fn weyl_tensor() -> ConformalWeight {
        ConformalWeight::Invariant
    }

    /// Ricci tensor: R̃_{ij} has weight 0 at leading order (not a simple power).
    pub fn ricci_tensor() -> ConformalWeight {
        ConformalWeight::Fixed(0.0) // approximate
    }

    /// A scalar field of weight Δ.
    pub fn scalar_field(delta: f64) -> ConformalWeight {
        ConformalWeight::Fixed(delta)
    }

    /// Conformal Laplacian eigenvalue.
    pub fn yamabe_eigenvalue() -> ConformalWeight {
        ConformalWeight::DimensionDependent(|n| -(2.0 + 4.0 / (n as f64 - 2.0)))
    }
}

/// Perform a conformal rescaling of a field.
pub fn conformal_rescale(
    field_value: f64,
    weight: &ConformalWeight,
    omega: f64,
    dim: usize,
) -> f64 {
    weight.transform(field_value, omega, dim)
}

/// Rescale a matrix field (e.g., metric tensor) under conformal change.
pub fn rescale_matrix(
    matrix: &DMatrix<f64>,
    weight: f64,
    omega: f64,
) -> DMatrix<f64> {
    matrix.scale(omega.powf(weight))
}

/// The conformal factor relating two metrics: g̃ = e^{2σ} g.
/// Here σ = ln(Ω), so Ω = e^σ.
pub fn conformal_factor_from_sigma(sigma: f64) -> f64 {
    sigma.exp()
}

/// Recover σ from Ω.
pub fn sigma_from_conformal_factor(omega: f64) -> f64 {
    omega.ln()
}

/// Chain two conformal rescalings: g̃ = Ω₁² g, g̃̃ = Ω₂² g̃.
/// The combined factor is (Ω₁Ω₂).
pub fn chain_conformal_factors(omega1: f64, omega2: f64) -> f64 {
    omega1 * omega2
}

/// A conformal class of metrics [g] = {e^{2σ} g : σ ∈ C^∞}.
#[derive(Debug, Clone)]
pub struct ConformalClass {
    /// The representative metric.
    pub representative: DMatrix<f64>,
    /// Dimension.
    pub dimension: usize,
}

impl ConformalClass {
    /// Create a conformal class from a representative metric.
    pub fn new(metric: DMatrix<f64>) -> Self {
        let n = metric.nrows();
        Self { representative: metric, dimension: n }
    }

    /// Generate a new metric in the same conformal class.
    pub fn rescale(&self, sigma: f64) -> DMatrix<f64> {
        let omega_sq = (2.0 * sigma).exp();
        self.representative.scale(omega_sq)
    }

    /// Check if two metrics are in the same conformal class.
    pub fn are_conformally_related(&self, other: &DMatrix<f64>, tol: f64) -> bool {
        // g̃ = λ g for some positive λ
        let n = self.dimension;
        if other.nrows() != n || other.ncols() != n {
            return false;
        }
        // Find λ = g̃_{ij}/g_{ij} (should be the same for all i,j)
        let mut lambda = None;
        for i in 0..n {
            for j in 0..n {
                if self.representative[(i, j)].abs() > tol {
                    let l = other[(i, j)] / self.representative[(i, j)];
                    match lambda {
                        None => lambda = Some(l),
                        Some(l0) => {
                            if (l - l0).abs() > tol * l0.abs().max(1.0) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        lambda.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_invariant_weight() {
        let w = ConformalWeight::Invariant;
        assert_relative_eq!(w.weight(3), 0.0);
        assert_relative_eq!(w.transform(5.0, 2.0, 3), 5.0);
    }

    #[test]
    fn test_fixed_weight() {
        let w = ConformalWeight::Fixed(2.0);
        assert_relative_eq!(w.weight(3), 2.0);
        assert_relative_eq!(w.transform(3.0, 2.0, 3), 12.0); // 2^2 * 3
    }

    #[test]
    fn test_dimension_dependent_weight() {
        let w = ConformalWeight::DimensionDependent(|n| n as f64);
        assert_relative_eq!(w.weight(3), 3.0);
        assert_relative_eq!(w.transform(1.0, 2.0, 3), 8.0); // 2^3 * 1
    }

    #[test]
    fn test_metric_weight() {
        let w = ConformalWeights::metric();
        assert_relative_eq!(w.weight(3), 2.0);
    }

    #[test]
    fn test_inverse_metric_weight() {
        let w = ConformalWeights::inverse_metric();
        assert_relative_eq!(w.weight(3), -2.0);
    }

    #[test]
    fn test_volume_form_weight() {
        let w = ConformalWeights::volume_form();
        assert_relative_eq!(w.weight(3), 3.0);
    }

    #[test]
    fn test_weyl_invariant() {
        let w = ConformalWeights::weyl_tensor();
        assert_relative_eq!(w.weight(4), 0.0);
        assert_relative_eq!(w.transform(7.0, 3.0, 4), 7.0);
    }

    #[test]
    fn test_conformal_rescale() {
        let result = conformal_rescale(5.0, &ConformalWeight::Fixed(1.0), 3.0, 3);
        assert_relative_eq!(result, 15.0);
    }

    #[test]
    fn test_rescale_matrix() {
        let m = DMatrix::identity(3, 3);
        let result = rescale_matrix(&m, 2.0, 2.0);
        assert_relative_eq!(result[(0, 0)], 4.0);
    }

    #[test]
    fn test_conformal_factor_roundtrip() {
        let sigma = 1.5;
        let omega = conformal_factor_from_sigma(sigma);
        let sigma_back = sigma_from_conformal_factor(omega);
        assert_relative_eq!(sigma, sigma_back, epsilon = 1e-10);
    }

    #[test]
    fn test_chain_factors() {
        let combined = chain_conformal_factors(2.0, 3.0);
        assert_relative_eq!(combined, 6.0);
    }

    #[test]
    fn test_conformal_class_rescale() {
        let metric = DMatrix::identity(3, 3);
        let cc = ConformalClass::new(metric);
        let rescaled = cc.rescale(0.0);
        assert_relative_eq!(rescaled[(0, 0)], 1.0);
    }

    #[test]
    fn test_conformal_class_related() {
        let metric = DMatrix::identity(3, 3);
        let cc = ConformalClass::new(metric);
        let scaled = DMatrix::from_diagonal_element(3, 3, 4.0);
        assert!(cc.are_conformally_related(&scaled, 1e-10));
    }

    #[test]
    fn test_conformal_class_unrelated() {
        let metric = DMatrix::identity(2, 2);
        let cc = ConformalClass::new(metric);
        let other = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        assert!(!cc.are_conformally_related(&other, 1e-10));
    }

    #[test]
    fn test_scalar_field_weight() {
        let w = ConformalWeights::scalar_field(-1.0);
        assert_relative_eq!(w.weight(4), -1.0);
        assert_relative_eq!(w.transform(6.0, 2.0, 4), 3.0); // 2^{-1} * 6
    }
}
