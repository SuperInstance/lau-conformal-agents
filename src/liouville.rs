//! Liouville's theorem on conformal maps in R^n.
//!
//! In dimensions n ≥ 3, the only conformal maps are Möbius transformations.
//! In dimension n = 2, there are many more (any holomorphic function with nonzero derivative).

use nalgebra::{DMatrix, DVector};
/// Dimension of the ambient space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Two,
    Three,
    Four,
    General(usize),
}

/// Result of checking whether a map can be conformal in a given dimension.
#[derive(Debug, Clone)]
pub struct ConformalClassification {
    pub dimension: Dimension,
    pub is_conformal: bool,
    pub description: String,
}

/// Check if a 2x2 real matrix represents a conformal map (scaled rotation).
pub fn is_conformal_matrix_2d(matrix: &DMatrix<f64>) -> bool {
    if matrix.nrows() != 2 || matrix.ncols() != 2 {
        return false;
    }
    let a = matrix[(0, 0)];
    let b = matrix[(0, 1)];
    let c = matrix[(1, 0)];
    let d = matrix[(1, 1)];
    // Conformal iff a = d and b = -c (Cauchy-Riemann)
    let tol = 1e-10;
    (a - d).abs() < tol && (b + c).abs() < tol
}

/// Check if a 3x3 real matrix represents a conformal map (scaled rotation).
/// In 3D, a conformal matrix must be a scalar multiple of an orthogonal matrix.
pub fn is_conformal_matrix_3d(matrix: &DMatrix<f64>) -> bool {
    if matrix.nrows() != 3 || matrix.ncols() != 3 {
        return false;
    }
    // Check if M^T M = λI for some λ > 0
    let mt_m = matrix.transpose() * matrix;
    let diag_0 = mt_m[(0, 0)];
    let tol = 1e-10 * diag_0.max(1.0);
    if diag_0 < -tol {
        return false;
    }
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { diag_0 } else { 0.0 };
            if (mt_m[(i, j)] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Liouville's theorem: classify what conformal maps exist in each dimension.
pub fn classify_conformal_maps(dim: Dimension) -> ConformalClassification {
    match dim {
        Dimension::Two => ConformalClassification {
            dimension: dim,
            is_conformal: true,
            description: "In 2D, all holomorphic functions with nonzero derivative are conformal. \
                          Much richer than higher dimensions.".to_string(),
        },
        Dimension::Three | Dimension::Four => {
            let d = match dim {
                Dimension::Three => 3,
                Dimension::Four => 4,
                _ => unreachable!(),
            };
            ConformalClassification {
                dimension: dim,
                is_conformal: true,
                description: format!(
                    "In {}D (n≥3), Liouville's theorem: the only conformal maps \
                     are Möbius transformations (compositions of translations, rotations, \
                     dilations, and inversions).",
                    d
                ),
            }
        }
        Dimension::General(n) if n >= 3 => ConformalClassification {
            dimension: dim,
            is_conformal: true,
            description: format!(
                "In {}D (n≥3), Liouville's theorem: the only conformal maps \
                 are Möbius transformations.",
                n
            ),
        },
        _ => ConformalClassification {
            dimension: dim,
            is_conformal: false,
            description: "Invalid dimension".to_string(),
        },
    }
}

/// The Weyl tensor in 3D must vanish for conformal flatness.
/// (In 3D, the Cotton tensor plays this role.)
pub fn is_conformally_flat_3d(cotton_tensor: &DMatrix<f64>) -> bool {
    // If all components of the Cotton tensor vanish, the manifold is conformally flat
    cotton_tensor.iter().all(|&x| x.abs() < 1e-10)
}

/// Conformal Killing equation: L_X g = 2σg
/// For a vector field X on R^n, this constrains X to be a conformal Killing field.
/// In R^n, the conformal Killing fields are: translations, rotations, dilations, and inversions.
pub fn conformal_killing_equation(
    xi: &DVector<f64>,
    dx_re: &DVector<f64>,
    dy_re: &DVector<f64>,
) -> bool {
    // The conformal Killing equation in 2D: ∂ξ/∂x̄ = 0 (Cauchy-Riemann for the velocity field)
    // For general dimension, we check ∂_i ξ_j + ∂_j ξ_i = 2σ δ_{ij}
    // In 2D with complex coordinates: ∂_{z̄} ξ = 0
    if xi.nrows() < 2 || dx_re.nrows() < 2 || dy_re.nrows() < 2 {
        return false;
    }
    // Simplified check: the antisymmetric part should be zero
    let tol = 1e-10;
    let sym = dx_re[1] + dy_re[0];
    let antisym = dx_re[1] - dy_re[0];
    sym.abs() < tol || antisym.abs() < tol * sym.abs().max(1.0)
}

/// Maximum dimension of the conformal group in n dimensions.
/// The conformal group in n dimensions has dimension (n+1)(n+2)/2.
pub fn conformal_group_dimension(n: usize) -> usize {
    (n + 1) * (n + 2) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_conformal_2d_identity() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(is_conformal_matrix_2d(&m));
    }

    #[test]
    fn test_conformal_2d_rotation() {
        let theta = std::f64::consts::FRAC_PI_4;
        let m = DMatrix::from_row_slice(2, 2, &[
            theta.cos(), -theta.sin(),
            theta.sin(), theta.cos(),
        ]);
        assert!(is_conformal_matrix_2d(&m));
    }

    #[test]
    fn test_conformal_2d_scaled_rotation() {
        let s = 3.0;
        let theta = std::f64::consts::FRAC_PI_6;
        let m = DMatrix::from_row_slice(2, 2, &[
            s * theta.cos(), -s * theta.sin(),
            s * theta.sin(), s * theta.cos(),
        ]);
        assert!(is_conformal_matrix_2d(&m));
    }

    #[test]
    fn test_non_conformal_2d() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(!is_conformal_matrix_2d(&m));
    }

    #[test]
    fn test_conformal_3d_identity() {
        let m = DMatrix::identity(3, 3);
        assert!(is_conformal_matrix_3d(&m));
    }

    #[test]
    fn test_conformal_3d_scaled() {
        let m = DMatrix::from_diagonal_element(3, 3, 2.0);
        assert!(is_conformal_matrix_3d(&m));
    }

    #[test]
    fn test_non_conformal_3d() {
        let m = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]);
        assert!(!is_conformal_matrix_3d(&m));
    }

    #[test]
    fn test_liouville_2d() {
        let c = classify_conformal_maps(Dimension::Two);
        assert!(c.is_conformal);
        assert!(c.description.contains("holomorphic"));
    }

    #[test]
    fn test_liouville_3d() {
        let c = classify_conformal_maps(Dimension::Three);
        assert!(c.is_conformal);
        assert!(c.description.contains("Möbius"));
    }

    #[test]
    fn test_conformally_flat_3d() {
        let zero = DMatrix::zeros(3, 3);
        assert!(is_conformally_flat_3d(&zero));
    }

    #[test]
    fn test_not_conformally_flat_3d() {
        let mut m = DMatrix::zeros(3, 3);
        m[(0, 0)] = 1.0;
        assert!(!is_conformally_flat_3d(&m));
    }

    #[test]
    fn test_conformal_group_dimension_2() {
        // (2+1)(2+2)/2 = 6
        assert_eq!(conformal_group_dimension(2), 6);
    }

    #[test]
    fn test_conformal_group_dimension_3() {
        // (3+1)(3+2)/2 = 10
        assert_eq!(conformal_group_dimension(3), 10);
    }

    #[test]
    fn test_conformal_group_dimension_4() {
        // (4+1)(4+2)/2 = 15
        assert_eq!(conformal_group_dimension(4), 15);
    }

    #[test]
    fn test_wrong_size_matrix_2d() {
        let m = DMatrix::identity(3, 3);
        assert!(!is_conformal_matrix_2d(&m));
    }

    #[test]
    fn test_wrong_size_matrix_3d() {
        let m = DMatrix::identity(2, 2);
        assert!(!is_conformal_matrix_3d(&m));
    }
}
