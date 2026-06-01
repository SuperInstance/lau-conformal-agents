//! Weyl tensor — the conformal curvature tensor.
//!
//! The Weyl tensor C is the trace-free part of the Riemann curvature tensor.
//! It is invariant under conformal transformations: C̃_{ijkl} = C_{ijkl} for g̃ = Ω²g.
//! A manifold is conformally flat iff the Weyl tensor vanishes (for n ≥ 4).

use nalgebra::DMatrix;

/// The Weyl tensor in component form for an n-dimensional manifold.
/// In n dimensions, the Weyl tensor has n²(n²-1)/12 independent components.
#[derive(Debug, Clone)]
pub struct WeylTensor {
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Components stored as C[i][j][k][l] in a flat array.
    /// Index: i * n³ + j * n² + k * n + l
    components: Vec<f64>,
}

impl WeylTensor {
    /// Create a zero Weyl tensor.
    pub fn zeros(n: usize) -> Self {
        let size = n * n * n * n;
        Self {
            dimension: n,
            components: vec![0.0; size],
        }
    }

    /// Create from the Riemann tensor, Ricci tensor, and scalar curvature.
    /// C_{ijkl} = R_{ijkl} - (1/(n-2))(g_{ik}R_{jl} - g_{il}R_{jk} + g_{jl}R_{ik} - g_{jk}R_{il})
    ///            + (1/((n-1)(n-2)))(g_{ik}g_{jl} - g_{il}g_{jk})R
    pub fn from_riemann(
        riemann: &[f64],
        ricci: &DMatrix<f64>,
        scalar_curvature: f64,
        metric: &DMatrix<f64>,
    ) -> Self {
        let n = ricci.nrows();
        let mut components = vec![0.0; n * n * n * n];

        let coeff1 = if n >= 3 { 1.0 / (n as f64 - 2.0) } else { 0.0 };
        let coeff2 = if n >= 3 { 1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0)) } else { 0.0 };

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let idx = i * n * n * n + j * n * n + k * n + l;
                        let rijkl = riemann.get(idx).copied().unwrap_or(0.0);

                        let term1 = coeff1 * (
                            metric[(i, k)] * ricci[(j, l)]
                            - metric[(i, l)] * ricci[(j, k)]
                            + metric[(j, l)] * ricci[(i, k)]
                            - metric[(j, k)] * ricci[(i, l)]
                        );

                        let term2 = coeff2 * (
                            metric[(i, k)] * metric[(j, l)]
                            - metric[(i, l)] * metric[(j, k)]
                        ) * scalar_curvature;

                        components[idx] = rijkl - term1 + term2;
                    }
                }
            }
        }

        Self { dimension: n, components }
    }

    /// Get component C_{ijkl}.
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let n = self.dimension;
        self.components[i * n * n * n + j * n * n + k * n + l]
    }

    /// Set component C_{ijkl}.
    pub fn set(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        let n = self.dimension;
        self.components[i * n * n * n + j * n * n + k * n + l] = val;
    }

    /// Number of independent components in n dimensions.
    pub fn independent_components(n: usize) -> usize {
        n * n * (n * n - 1) / 12
    }

    /// Check if the Weyl tensor is zero (conformally flat for n ≥ 4).
    pub fn is_zero(&self, tol: f64) -> bool {
        self.components.iter().all(|&x| x.abs() < tol)
    }

    /// The Weyl tensor has the same symmetries as the Riemann tensor:
    /// C_{ijkl} = -C_{jikl} = -C_{ijlk} = C_{klij}
    /// Verify antisymmetry in first two indices.
    pub fn verify_antisymmetry_ij(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let c1 = self.get(i, j, k, l);
                        let c2 = self.get(j, i, k, l);
                        if (c1 + c2).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Verify antisymmetry in last two indices.
    pub fn verify_antisymmetry_kl(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let c1 = self.get(i, j, k, l);
                        let c2 = self.get(i, j, l, k);
                        if (c1 + c2).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Verify symmetry under swapping pairs: C_{ijkl} = C_{klij}.
    pub fn verify_pair_symmetry(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let c1 = self.get(i, j, k, l);
                        let c2 = self.get(k, l, i, j);
                        if (c1 - c2).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Trace-free property: C^i_{jil} = 0 (contraction on first and third indices).
    pub fn verify_trace_free(&self, metric_inv: &DMatrix<f64>, tol: f64) -> bool {
        let n = self.dimension;
        for j in 0..n {
            for l in 0..n {
                let mut trace = 0.0;
                for i in 0..n {
                    trace += metric_inv[(i, i)] * self.get(i, j, i, l);
                }
                if trace.abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

/// The Cotton tensor (important in 3D: replaces Weyl for conformal flatness).
/// C_{ijk} = ∇_k R_{ij} - ∇_j R_{ik} - (1/(2(n-1)))(g_{ij}∇_k R - g_{ik}∇_j R)
#[derive(Debug, Clone)]
pub struct CottonTensor {
    pub dimension: usize,
    components: Vec<f64>,
}

impl CottonTensor {
    /// Create from the gradient of the Ricci tensor and gradient of scalar curvature.
    pub fn from_ricci_gradient(
        n: usize,
        ricci_gradient: &DMatrix<f64>, // n x n: ∇_k R_{ij} stored as [i*n+k, j]
        scalar_curvature_gradient: &[f64], // n: ∇_k R
        metric: &DMatrix<f64>,
    ) -> Self {
        let mut components = vec![0.0; n * n * n];
        let coeff = if n >= 2 { 1.0 / (2.0 * (n as f64 - 1.0)) } else { 0.0 };

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = i * n * n + j * n + k;
                    // C_{ijk} = ∇_k R_{ij} - ∇_j R_{ik} - coeff*(g_{ij}∇_k R - g_{ik}∇_j R)
                    let dr_ij_k = ricci_gradient[(i * n + k, j)];
                    let dr_ik_j = ricci_gradient[(i * n + j, k)];
                    components[idx] = dr_ij_k - dr_ik_j
                        - coeff * (metric[(i, j)] * scalar_curvature_gradient[k]
                            - metric[(i, k)] * scalar_curvature_gradient[j]);
                }
            }
        }

        Self { dimension: n, components }
    }

    /// Get component C_{ijk}.
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        let n = self.dimension;
        self.components[i * n * n + j * n + k]
    }

    /// Check if the Cotton tensor is zero.
    pub fn is_zero(&self, tol: f64) -> bool {
        self.components.iter().all(|&x| x.abs() < tol)
    }
}

/// Check if a manifold of dimension n ≥ 4 is conformally flat (Weyl = 0).
pub fn is_conformally_flat_weyl(weyl: &WeylTensor) -> bool {
    weyl.is_zero(1e-10)
}

/// Check if a 3-manifold is conformally flat (Cotton = 0).
pub fn is_conformally_flat_cotton(cotton: &CottonTensor) -> bool {
    cotton.is_zero(1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weyl_zeros() {
        let w = WeylTensor::zeros(4);
        assert!(w.is_zero(1e-15));
    }

    #[test]
    fn test_weyl_flat_space() {
        // In flat space, Riemann = 0, Ricci = 0, R = 0
        let n = 4;
        let riemann = vec![0.0; n * n * n * n];
        let ricci = DMatrix::zeros(n, n);
        let metric = DMatrix::identity(n, n);
        let w = WeylTensor::from_riemann(&riemann, &ricci, 0.0, &metric);
        assert!(w.is_zero(1e-10));
    }

    #[test]
    fn test_independent_components_3() {
        // 3*3*(9-1)/12 = 6
        assert_eq!(WeylTensor::independent_components(3), 6);
    }

    #[test]
    fn test_independent_components_4() {
        // 4*4*(16-1)/12 = 20
        assert_eq!(WeylTensor::independent_components(4), 20);
    }

    #[test]
    fn test_weyl_get_set() {
        let mut w = WeylTensor::zeros(3);
        w.set(0, 1, 0, 1, 5.0);
        assert_relative_eq!(w.get(0, 1, 0, 1), 5.0);
    }

    #[test]
    fn test_weyl_antisymmetry_ij() {
        let mut w = WeylTensor::zeros(3);
        w.set(0, 1, 0, 1, 2.0);
        w.set(1, 0, 0, 1, -2.0);
        assert!(w.verify_antisymmetry_ij(1e-10));
    }

    #[test]
    fn test_weyl_antisymmetry_kl() {
        let mut w = WeylTensor::zeros(3);
        w.set(0, 0, 1, 2, 3.0);
        w.set(0, 0, 2, 1, -3.0);
        assert!(w.verify_antisymmetry_kl(1e-10));
    }

    #[test]
    fn test_weyl_pair_symmetry() {
        let mut w = WeylTensor::zeros(3);
        w.set(0, 1, 2, 0, 7.0);
        w.set(2, 0, 0, 1, 7.0);
        assert!(w.verify_pair_symmetry(1e-10));
    }

    #[test]
    fn test_cotton_zeros() {
        let c = CottonTensor::from_ricci_gradient(
            3,
            &DMatrix::zeros(9, 3),
            &[0.0; 3],
            &DMatrix::identity(3, 3),
        );
        assert!(c.is_zero(1e-10));
    }

    #[test]
    fn test_conformally_flat_weyl() {
        let w = WeylTensor::zeros(4);
        assert!(is_conformally_flat_weyl(&w));
    }

    #[test]
    fn test_conformally_flat_cotton() {
        let c = CottonTensor::from_ricci_gradient(
            3,
            &DMatrix::zeros(9, 3),
            &[0.0; 3],
            &DMatrix::identity(3, 3),
        );
        assert!(is_conformally_flat_cotton(&c));
    }

    #[test]
    fn test_independent_components_5() {
        // 5*5*(25-1)/12 = 50
        assert_eq!(WeylTensor::independent_components(5), 50);
    }

    #[test]
    fn test_weyl_set_nonzero() {
        let mut w = WeylTensor::zeros(4);
        w.set(0, 1, 2, 3, 1.0);
        assert!(!w.is_zero(0.1));
        assert_relative_eq!(w.get(0, 1, 2, 3), 1.0);
    }
}
