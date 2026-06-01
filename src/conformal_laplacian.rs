//! Conformal Laplacian (Yamabe operator).
//!
//! The conformal Laplacian is L_g = -Δ_g + ((n-2)/(4(n-1))) R_g,
//! where R_g is the scalar curvature. Under a conformal change g̃ = e^{2σ} g,
//! the conformal Laplacian transforms as:
//!   L_{g̃} = e^{-((n+2)/2)σ} L_g e^{((n-2)/2)σ}

use nalgebra::DMatrix;

/// The conformal Laplacian (Yamabe operator).
/// L_g = -Δ + ((n-2)/(4(n-1))) R
#[derive(Debug, Clone)]
pub struct ConformalLaplacian {
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Scalar curvature R.
    pub scalar_curvature: f64,
}

impl ConformalLaplacian {
    /// Create a new conformal Laplacian for a manifold of given dimension and scalar curvature.
    pub fn new(dimension: usize, scalar_curvature: f64) -> Self {
        Self { dimension, scalar_curvature }
    }

    /// The coefficient (n-2)/(4(n-1)) of the scalar curvature term.
    pub fn curvature_coefficient(&self) -> f64 {
        if self.dimension <= 1 {
            0.0
        } else if self.dimension == 2 {
            // Special case: coefficient is 0 for n=2 (use Gaussian curvature directly)
            0.0
        } else {
            (self.dimension as f64 - 2.0) / (4.0 * (self.dimension as f64 - 1.0))
        }
    }

    /// Apply the conformal Laplacian to a function f with given Laplacian value.
    /// L_g f = -Δ_g f + c(n) R_g f
    pub fn apply(&self, laplacian_f: f64, f: f64) -> f64 {
        -laplacian_f + self.curvature_coefficient() * self.scalar_curvature * f
    }

    /// The Yamabe constant: the infimum of ∫ L_g u · u dV over all u with ∫ u^{2n/(n-2)} = 1.
    /// For computational purposes, we provide a finite-dimensional approximation.
    pub fn yamabe_constant_estimate(
        &self,
        stiffness_matrix: &DMatrix<f64>,
        mass_matrix: &DMatrix<f64>,
    ) -> Option<f64> {
        // The Yamabe constant is the smallest eigenvalue of the generalized eigenvalue problem
        // L u = λ u, where L is the discretized conformal Laplacian
        // This is approximated by the Rayleigh quotient
        let n = stiffness_matrix.nrows();
        if n == 0 || n != mass_matrix.nrows() {
            return None;
        }
        // Use power iteration for the smallest eigenvalue
        let num_iterations = 100;
        let mut v = DMatrix::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());

        for _ in 0..num_iterations {
            // Solve mass_matrix * w = stiffness_matrix * v
            // Approximate: use the fact that for diagonal mass matrix, this is easy
            let mut mv = stiffness_matrix * &v;
            // Simple diagonal scaling as approximation
            for i in 0..n {
                if mass_matrix[(i, i)].abs() > 1e-15 {
                    mv[(i, 0)] /= mass_matrix[(i, i)];
                }
            }
            // Normalize
            let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                v = mv / norm;
            }
        }

        // Rayleigh quotient
        let mv = stiffness_matrix * &v;
        let vv = &v.transpose() * &mv;
        let vmv = &v.transpose() * (mass_matrix * &v);
        if vmv[(0, 0)].abs() < 1e-15 {
            None
        } else {
            Some(vv[(0, 0)] / vmv[(0, 0)])
        }
    }

    /// The conformal factor exponent for the transformation of L under g̃ = e^{2σ}g.
    /// L_{g̃} u = e^{-((n+2)/2)σ} L_g (e^{((n-2)/2)σ} u)
    pub fn conformal_transformation_exponents(&self) -> (f64, f64) {
        let n = self.dimension as f64;
        // The exponent for the multiplication on the left
        let left_exp = -(n + 2.0) / 2.0;
        // The exponent for the multiplication on the right
        let right_exp = (n - 2.0) / 2.0;
        (left_exp, right_exp)
    }
}

/// Transform the scalar curvature under a conformal change g̃ = e^{2σ} g.
/// R̃ = e^{-2σ}(R - 2(n-1)Δσ - (n-2)(n-1)|∇σ|^2)
pub fn transform_scalar_curvature(
    n: usize,
    r: f64,
    sigma: f64,
    laplacian_sigma: f64,
    grad_sigma_norm_sq: f64,
) -> f64 {
    let n_f = n as f64;
    (-2.0 * sigma).exp() * (r - 2.0 * (n_f - 1.0) * laplacian_sigma - (n_f - 2.0) * (n_f - 1.0) * grad_sigma_norm_sq)
}

/// The Yamabe equation: find σ such that g̃ = u^{4/(n-2)} g has constant scalar curvature.
/// -Δ_g u + c(n) R_g u = λ u^{(n+2)/(n-2)}
pub fn yamabe_equation_rhs(n: usize, u: f64, lambda: f64) -> f64 {
    let n_f = n as f64;
    let exponent = (n_f + 2.0) / (n_f - 2.0);
    lambda * u.powf(exponent)
}

/// Discretized conformal Laplacian as a matrix (on a triangulated surface).
pub fn discretized_conformal_laplacian(
    n_vertices: usize,
    cotangent_weights: &DMatrix<f64>,
    area_weights: &DMatrix<f64>,
    scalar_curvature: f64,
) -> DMatrix<f64> {
    // L = -L_cot + c * R * A
    // where L_cot is the cotangent Laplacian and A is the mass matrix
    let c = if n_vertices <= 2 { 0.0 } else { ((n_vertices as f64) - 2.0) / (4.0 * ((n_vertices as f64) - 1.0)) };
    -cotangent_weights + c * scalar_curvature * area_weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_curvature_coefficient_3d() {
        let cl = ConformalLaplacian::new(3, 1.0);
        // (3-2)/(4*(3-1)) = 1/8
        assert_relative_eq!(cl.curvature_coefficient(), 0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_curvature_coefficient_4d() {
        let cl = ConformalLaplacian::new(4, 1.0);
        // (4-2)/(4*(4-1)) = 2/12 = 1/6
        assert_relative_eq!(cl.curvature_coefficient(), 1.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curvature_coefficient_2d() {
        let cl = ConformalLaplacian::new(2, 1.0);
        assert_relative_eq!(cl.curvature_coefficient(), 0.0);
    }

    #[test]
    fn test_apply_flat() {
        // Flat space: R = 0, so L = -Δ
        let cl = ConformalLaplacian::new(3, 0.0);
        let result = cl.apply(5.0, 1.0);
        assert_relative_eq!(result, -5.0);
    }

    #[test]
    fn test_apply_positive_curvature() {
        let cl = ConformalLaplacian::new(3, 1.0);
        // L f = -Δf + (1/8) * 1 * f
        let result = cl.apply(0.0, 2.0);
        assert_relative_eq!(result, 0.25);
    }

    #[test]
    fn test_conformal_exponents_3d() {
        let cl = ConformalLaplacian::new(3, 0.0);
        let (left, right) = cl.conformal_transformation_exponents();
        // left = -(3+2)/2 = -2.5, right = (3-2)/2 = 0.5
        assert_relative_eq!(left, -2.5);
        assert_relative_eq!(right, 0.5);
    }

    #[test]
    fn test_conformal_exponents_4d() {
        let cl = ConformalLaplacian::new(4, 0.0);
        let (left, right) = cl.conformal_transformation_exponents();
        assert_relative_eq!(left, -3.0);
        assert_relative_eq!(right, 1.0);
    }

    #[test]
    fn test_transform_scalar_curvature_zero_sigma() {
        // g̃ = g when σ = 0
        let r = transform_scalar_curvature(3, 5.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(r, 5.0);
    }

    #[test]
    fn test_yamabe_equation_rhs_3d() {
        // (n+2)/(n-2) = 5 when n=3
        let rhs = yamabe_equation_rhs(3, 2.0, 1.0);
        assert_relative_eq!(rhs, 32.0); // 1.0 * 2^5 = 32
    }

    #[test]
    fn test_yamabe_equation_rhs_6d() {
        // (n+2)/(n-2) = 8/4 = 2 when n=6
        let rhs = yamabe_equation_rhs(6, 3.0, 2.0);
        assert_relative_eq!(rhs, 18.0); // 2.0 * 3^2 = 18
    }

    #[test]
    fn test_discretized_flat() {
        let n = 3;
        let cot = DMatrix::identity(n, n);
        let area = DMatrix::identity(n, n);
        let L = discretized_conformal_laplacian(n, &cot, &area, 0.0);
        assert_relative_eq!(L[(0, 0)], -1.0);
    }

    #[test]
    fn test_curvature_coefficient_1d() {
        let cl = ConformalLaplacian::new(1, 1.0);
        assert_relative_eq!(cl.curvature_coefficient(), 0.0);
    }

    #[test]
    fn test_apply_negative_curvature() {
        let cl = ConformalLaplacian::new(3, -2.0);
        let result = cl.apply(0.0, 1.0);
        assert_relative_eq!(result, -0.25); // 0.125 * (-2) * 1
    }

    #[test]
    fn test_yamabe_constant_simple() {
        let cl = ConformalLaplacian::new(3, 0.0);
        let stiff = DMatrix::from_diagonal_element(3, 3, 2.0);
        let mass = DMatrix::identity(3, 3);
        let yc = cl.yamabe_constant_estimate(&stiff, &mass);
        assert!(yc.is_some());
        assert_relative_eq!(yc.unwrap(), 2.0, epsilon = 0.1);
    }
}
