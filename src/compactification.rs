//! Conformal compactification.
//!
//! Add a point at infinity to make a non-compact manifold into a compact one,
//! preserving the conformal structure. The prototypical example: Riemann sphere = ℂ ∪ {∞}.

use nalgebra::{DMatrix, DVector};

/// Stereographic projection from S^n to R^n.
#[derive(Debug, Clone)]
pub struct StereographicProjection {
    /// Dimension n of the sphere S^n.
    pub dimension: usize,
    /// Projection pole (default: north pole = (0,...,0,1)).
    pub north_pole: bool,
}

impl StereographicProjection {
    /// Create a stereographic projection from the north pole of S^n.
    pub fn north(dimension: usize) -> Self {
        Self { dimension, north_pole: true }
    }

    /// Create a stereographic projection from the south pole of S^n.
    pub fn south(dimension: usize) -> Self {
        Self { dimension, north_pole: false }
    }

    /// Project a point on S^n ⊂ R^{n+1} to R^n.
    /// North pole projection: (x₁,...,xₙ, xₙ₊₁) ↦ (x₁,...,xₙ)/(1 - xₙ₊₁)
    /// South pole projection: (x₁,...,xₙ, xₙ₊₁) ↦ (x₁,...,xₙ)/(1 + xₙ₊₁)
    pub fn project(&self, point: &DVector<f64>) -> Option<DVector<f64>> {
        let n = self.dimension;
        if point.nrows() != n + 1 {
            return None;
        }
        let last = point[n];
        let denom = if self.north_pole { 1.0 - last } else { 1.0 + last };
        if denom.abs() < 1e-14 {
            return None; // at the pole, maps to infinity
        }
        let mut result = DVector::zeros(n);
        for i in 0..n {
            result[i] = point[i] / denom;
        }
        Some(result)
    }

    /// Inverse projection: R^n → S^n.
    /// North pole: (u₁,...,uₙ) ↦ (2u₁,...,2uₙ, |u|²-1) / (|u|²+1)
    pub fn inverse(&self, u: &DVector<f64>) -> DVector<f64> {
        let n = self.dimension;
        let norm_sq: f64 = (0..n).map(|i| u[i] * u[i]).sum();
        let denom = norm_sq + 1.0;
        let mut result = DVector::zeros(n + 1);
        for i in 0..n {
            result[i] = 2.0 * u[i] / denom;
        }
        if self.north_pole {
            result[n] = (norm_sq - 1.0) / denom;
        } else {
            result[n] = (1.0 - norm_sq) / denom;
        }
        result
    }
}

/// Conformal compactification of R^n to S^n.
#[derive(Debug, Clone)]
pub struct ConformalCompactification {
    pub dimension: usize,
}

impl ConformalCompactification {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Add the point at infinity to compactify.
    pub fn compactify(&self, points: &[DVector<f64>]) -> Vec<Option<DVector<f64>>> {
        let proj = StereographicProjection::north(self.dimension);
        points.iter().map(|p| proj.inverse(p)).map(Some).collect()
    }
}

/// The one-point compactification of R^n: R^n ∪ {∞} ≅ S^n.
pub fn one_point_compactify(point: &DVector<f64>, dimension: usize) -> Option<DVector<f64>> {
    let proj = StereographicProjection::north(dimension);
    Some(proj.inverse(point))
}

/// The point at infinity on S^n.
pub fn infinity_point(dimension: usize) -> DVector<f64> {
    let mut p = DVector::zeros(dimension + 1);
    p[dimension] = 1.0;
    p
}

/// Möbius action on S^n via stereographic projection.
/// Apply a conformal map to R^n, then compactify back to S^n.
pub fn mobius_action_on_sphere(
    point_on_sphere: &DVector<f64>,
    map: impl Fn(&DVector<f64>) -> Option<DVector<f64>>,
    dimension: usize,
) -> Option<DVector<f64>> {
    let proj = StereographicProjection::north(dimension);
    let r_n = proj.project(point_on_sphere)?;
    let mapped = map(&r_n)?;
    Some(proj.inverse(&mapped))
}

/// Inversion on R^n: x ↦ x/|x|² (conformal, adds infinity).
pub fn inversion_rn(x: &DVector<f64>) -> Option<DVector<f64>> {
    let norm_sq: f64 = x.iter().map(|v| v * v).sum();
    if norm_sq < 1e-14 {
        None
    } else {
        Some(x.scale(1.0 / norm_sq))
    }
}

/// The round metric on S^n in stereographic coordinates.
/// g = 4/(1 + |u|²)² δ_{ij}
pub fn round_metric_stereographic(u: &DVector<f64>) -> DMatrix<f64> {
    let n = u.nrows();
    let norm_sq: f64 = u.iter().map(|v| v * v).sum();
    let factor = 4.0 / (1.0 + norm_sq).powi(2);
    DMatrix::from_diagonal_element(n, n, factor)
}

/// Conformal factor for the round metric: Ω² = 4/(1 + |u|²)².
pub fn round_metric_conformal_factor(u: &DVector<f64>) -> f64 {
    let norm_sq: f64 = u.iter().map(|v| v * v).sum();
    4.0 / (1.0 + norm_sq).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stereographic_north_origin() {
        let proj = StereographicProjection::north(2);
        let north = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        assert!(proj.project(&north).is_none()); // north pole → infinity
    }

    #[test]
    fn test_stereographic_south_origin() {
        let proj = StereographicProjection::south(2);
        let south = DVector::from_vec(vec![0.0, 0.0, -1.0]);
        assert!(proj.project(&south).is_none()); // south pole → infinity
    }

    #[test]
    fn test_stereographic_north_equator() {
        let proj = StereographicProjection::north(2);
        let equator = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let projected = proj.project(&equator).unwrap();
        assert_relative_eq!(projected[0], 1.0);
        assert_relative_eq!(projected[1], 0.0);
    }

    #[test]
    fn test_stereographic_roundtrip() {
        let proj = StereographicProjection::north(2);
        let u = DVector::from_vec(vec![3.0, 4.0]);
        let on_sphere = proj.inverse(&u);
        let back = proj.project(&on_sphere).unwrap();
        assert_relative_eq!((back - u).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stereographic_south_roundtrip() {
        let proj = StereographicProjection::south(2);
        let u = DVector::from_vec(vec![1.0, 2.0]);
        let on_sphere = proj.inverse(&u);
        let back = proj.project(&on_sphere).unwrap();
        assert_relative_eq!((back - u).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_on_sphere() {
        let proj = StereographicProjection::north(2);
        let u = DVector::from_vec(vec![0.0, 0.0]);
        let p = proj.inverse(&u);
        // origin maps to south pole (0,0,-1)
        assert_relative_eq!(p[0], 0.0);
        assert_relative_eq!(p[1], 0.0);
        assert_relative_eq!(p[2], -1.0);
    }

    #[test]
    fn test_infinity_point() {
        let inf = infinity_point(2);
        assert_relative_eq!(inf[0], 0.0);
        assert_relative_eq!(inf[1], 0.0);
        assert_relative_eq!(inf[2], 1.0);
    }

    #[test]
    fn test_inversion_rn() {
        let x = DVector::from_vec(vec![3.0, 4.0]); // |x|² = 25
        let inv = inversion_rn(&x).unwrap();
        assert_relative_eq!(inv[0], 3.0 / 25.0);
        assert_relative_eq!(inv[1], 4.0 / 25.0);
    }

    #[test]
    fn test_inversion_rn_origin() {
        let x = DVector::zeros(3);
        assert!(inversion_rn(&x).is_none());
    }

    #[test]
    fn test_inversion_involutive() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let inv1 = inversion_rn(&x).unwrap();
        let inv2 = inversion_rn(&inv1).unwrap();
        assert_relative_eq!((inv2 - x).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_round_metric_origin() {
        let u = DVector::zeros(3);
        let g = round_metric_stereographic(&u);
        assert_relative_eq!(g[(0, 0)], 4.0);
    }

    #[test]
    fn test_round_metric_factor() {
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let factor = round_metric_conformal_factor(&u);
        assert_relative_eq!(factor, 4.0 / 4.0); // 4/(1+1)² = 1
    }

    #[test]
    fn test_mobius_action_inversion() {
        let proj = StereographicProjection::north(2);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let on_sphere = proj.inverse(&u);
        let result = mobius_action_on_sphere(&on_sphere, |x| inversion_rn(x), 2);
        assert!(result.is_some());
    }

    #[test]
    fn test_one_point_compactify() {
        let p = DVector::from_vec(vec![0.0, 0.0]);
        let on_sphere = one_point_compactify(&p, 2).unwrap();
        // origin maps to south pole
        assert_relative_eq!(on_sphere[2], -1.0);
    }
}
