//! Conformal maps — transformations that preserve angles.
//!
//! A smooth map f: U → V between Riemannian manifolds is conformal if it
//! preserves angles between curves. Equivalently, f* g = e^{2σ} g for some
//! smooth function σ (the conformal factor).

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// A conformal map in 2D defined by its action on the complex plane.
#[derive(Debug, Clone)]
pub struct ConformalMap {
    /// Jacobian matrix at a point (2×2 real matrix representing the complex derivative).
    jacobian: DMatrix<f64>,
    /// Point where the Jacobian is evaluated.
    #[allow(dead_code)]
    point: DVector<f64>,
    /// Conformal factor |f'(z)| at the point.
    conformal_factor: f64,
}

impl ConformalMap {
    /// Create a conformal map from a complex derivative at a point.
    pub fn from_complex_derivative(z: Complex64, f_prime: Complex64) -> Self {
        let conformal_factor = f_prime.norm();
        // Complex derivative f'(z) = a + ib corresponds to 2x2 matrix [[a, -b], [b, a]]
        let jacobian = DMatrix::from_row_slice(2, 2, &[
            f_prime.re, -f_prime.im,
            f_prime.im, f_prime.re,
        ]);
        Self {
            jacobian,
            point: DVector::from_vec(vec![z.re, z.im]),
            conformal_factor,
        }
    }

    /// The conformal factor (scale factor of the map).
    pub fn conformal_factor(&self) -> f64 {
        self.conformal_factor
    }

    /// The Jacobian matrix.
    pub fn jacobian(&self) -> &DMatrix<f64> {
        &self.jacobian
    }

    /// Check if the map is conformal (Jacobian is a scaled rotation).
    pub fn is_conformal(&self) -> bool {
        // A 2x2 matrix is conformal iff [[a,b],[c,d]] with a=d and b=-c
        let a = self.jacobian[(0, 0)];
        let b = self.jacobian[(0, 1)];
        let c = self.jacobian[(1, 0)];
        let d = self.jacobian[(1, 1)];
        (a - d).abs() < 1e-10 && (b + c).abs() < 1e-10
    }

    /// Evaluate the pullback metric: f* g = e^{2σ} g.
    /// Returns the conformal factor squared (metric scaling).
    pub fn pullback_metric_scale(&self) -> f64 {
        self.conformal_factor * self.conformal_factor
    }
}

/// The exponential map is conformal: w = e^z.
pub fn exponential_map(z: Complex64) -> Complex64 {
    z.exp()
}

/// The logarithm map (inverse of exponential, principal branch).
pub fn logarithm_map(w: Complex64) -> Complex64 {
    w.ln()
}

/// Power map w = z^α (conformal except at branch points).
pub fn power_map(z: Complex64, alpha: f64) -> Complex64 {
    z.powf(alpha)
}

/// Check if a function defined by its complex derivative is conformal at a point.
pub fn is_conformal_at(f_prime: Complex64) -> bool {
    // Conformal iff f'(z) ≠ 0
    f_prime.norm() > 1e-15
}

/// Compose two conformal maps (complex function composition).
pub fn compose(f: impl Fn(Complex64) -> Complex64, g: impl Fn(Complex64) -> Complex64, z: Complex64) -> Complex64 {
    f(g(z))
}

/// Inversion map w = 1/z (conformal everywhere except z=0).
pub fn conformal_inversion(z: Complex64) -> Option<Complex64> {
    if z.norm() < 1e-15 {
        None
    } else {
        Some(Complex64::new(1.0, 0.0) / z)
    }
}

/// Compute the Schwarzian derivative {f, z} = f'''/f' - (3/2)(f''/f')^2.
pub fn schwarzian_derivative(
    z: Complex64,
    _f: impl Fn(Complex64) -> Complex64,
    f_prime: impl Fn(Complex64) -> Complex64,
    f_double_prime: impl Fn(Complex64) -> Complex64,
    f_triple_prime: impl Fn(Complex64) -> Complex64,
) -> Complex64 {
    let fp = f_prime(z);
    let fpp = f_double_prime(z);
    let fppp = f_triple_prime(z);
    // {f, z} = f'''/f' - (3/2)(f''/f')^2
    fppp / fp - Complex64::new(1.5, 0.0) * (fpp / fp) * (fpp / fp)
}

/// Conformal maps on the upper half-plane.
pub mod half_plane {
    use num_complex::Complex64;

    /// The Cayley transform maps the upper half-plane to the unit disk.
    /// w = (z - i) / (z + i)
    pub fn cayley_to_disk(z: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        (z - i) / (z + i)
    }

    /// Inverse Cayley transform maps the unit disk to the upper half-plane.
    /// z = i(1 + w) / (1 - w)
    pub fn cayley_from_disk(w: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        i * (Complex64::new(1.0, 0.0) + w) / (Complex64::new(1.0, 0.0) - w)
    }

    /// Check if a point is in the upper half-plane (Im(z) > 0).
    pub fn is_in_upper_half_plane(z: Complex64) -> bool {
        z.im > 0.0
    }

    /// Check if a point is in the unit disk (|z| < 1).
    pub fn is_in_unit_disk(z: Complex64) -> bool {
        z.norm() < 1.0
    }
}

/// Circle-preserving maps (send circles/lines to circles/lines).
pub mod circle_maps {
    use num_complex::Complex64;

    /// A generalized circle (circle or line) in the complex plane.
    #[derive(Debug, Clone)]
    pub struct GeneralizedCircle {
        /// Center (None for lines).
        pub center: Option<Complex64>,
        /// Radius (None for lines).
        pub radius: Option<f64>,
        /// For a line ax + by = c: coefficients [a, b, c].
        pub line_coeffs: Option<(f64, f64, f64)>,
    }

    impl GeneralizedCircle {
        /// Create a circle from center and radius.
        pub fn circle(center: Complex64, radius: f64) -> Self {
            Self {
                center: Some(center),
                radius: Some(radius),
                line_coeffs: None,
            }
        }

        /// Create a line from ax + by = c.
        pub fn line(a: f64, b: f64, c: f64) -> Self {
            Self {
                center: None,
                radius: None,
                line_coeffs: Some((a, b, c)),
            }
        }

        /// Check if a point lies on the circle/line within tolerance.
        pub fn contains(&self, z: Complex64, tol: f64) -> bool {
            if let (Some(c), Some(r)) = (&self.center, self.radius) {
                (z - c).norm() - r < tol
            } else if let Some((a, b, c_val)) = self.line_coeffs {
                (a * z.re + b * z.im - c_val).abs() < tol
            } else {
                false
            }
        }

        /// Is this a line (not a circle)?
        pub fn is_line(&self) -> bool {
            self.line_coeffs.is_some()
        }

        /// Is this a circle?
        pub fn is_circle(&self) -> bool {
            self.center.is_some() && self.radius.is_some()
        }
    }

    /// Three points determine a generalized circle.
    pub fn circle_through_three_points(
        z1: Complex64,
        z2: Complex64,
        z3: Complex64,
    ) -> Option<GeneralizedCircle> {
        // Check collinearity via cross product
        let cross = (z2.re - z1.re) * (z3.im - z1.im) - (z2.im - z1.im) * (z3.re - z1.re);
        if cross.abs() < 1e-10 {
            // Collinear: they lie on a line
            let a = z2.im - z1.im;
            let b = z1.re - z2.re;
            let c = a * z1.re + b * z1.im;
            Some(GeneralizedCircle::line(a, b, c))
        } else {
            // Find circumcircle
            let ax = z1.re;
            let ay = z1.im;
            let bx = z2.re;
            let by = z2.im;
            let cx = z3.re;
            let cy = z3.im;
            let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
            let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
                + (cx * cx + cy * cy) * (ay - by))
                / d;
            let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
                + (cx * cx + cy * cy) * (bx - ax))
                / d;
            let center = Complex64::new(ux, uy);
            let radius = (z1 - center).norm();
            Some(GeneralizedCircle::circle(center, radius))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_conformal_map_identity() {
        let f_prime = Complex64::new(1.0, 0.0);
        let z = Complex64::new(2.0, 3.0);
        let cm = ConformalMap::from_complex_derivative(z, f_prime);
        assert!(cm.is_conformal());
        assert_relative_eq!(cm.conformal_factor(), 1.0);
    }

    #[test]
    fn test_conformal_map_rotation() {
        // f(z) = iz, f'(z) = i
        let f_prime = Complex64::new(0.0, 1.0);
        let z = Complex64::new(1.0, 0.0);
        let cm = ConformalMap::from_complex_derivative(z, f_prime);
        assert!(cm.is_conformal());
        assert_relative_eq!(cm.conformal_factor(), 1.0);
    }

    #[test]
    fn test_conformal_map_scaling() {
        // f(z) = 3z, f'(z) = 3
        let f_prime = Complex64::new(3.0, 0.0);
        let z = Complex64::new(1.0, 0.0);
        let cm = ConformalMap::from_complex_derivative(z, f_prime);
        assert!(cm.is_conformal());
        assert_relative_eq!(cm.conformal_factor(), 3.0);
    }

    #[test]
    fn test_exponential_map() {
        let z = Complex64::new(0.0, std::f64::consts::FRAC_PI_2);
        let w = exponential_map(z);
        assert_relative_eq!(w.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(w.im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_logarithm_roundtrip() {
        let z = Complex64::new(2.0, 1.0);
        assert_relative_eq!((logarithm_map(exponential_map(z)) - z).re, 0.0, epsilon = 1e-10);
        assert_relative_eq!((logarithm_map(exponential_map(z)) - z).im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_power_map() {
        let z = Complex64::new(4.0, 0.0);
        let w = power_map(z, 0.5);
        assert_relative_eq!(w.re, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inversion() {
        let z = Complex64::new(2.0, 0.0);
        let w = conformal_inversion(z).unwrap();
        assert_relative_eq!(w.re, 0.5);
    }

    #[test]
    fn test_inversion_zero() {
        assert!(conformal_inversion(Complex64::new(0.0, 0.0)).is_none());
    }

    #[test]
    fn test_is_conformal_at_nonzero() {
        assert!(is_conformal_at(Complex64::new(1.0, 1.0)));
    }

    #[test]
    fn test_is_conformal_at_zero() {
        assert!(!is_conformal_at(Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_compose() {
        let f = |z: Complex64| z * Complex64::new(2.0, 0.0);
        let g = |z: Complex64| z + Complex64::new(1.0, 0.0);
        let z = Complex64::new(3.0, 0.0);
        let result = compose(f, g, z);
        assert_relative_eq!(result.re, 8.0);
    }

    #[test]
    fn test_cayley_roundtrip() {
        use half_plane::*;
        let z = Complex64::new(1.0, 2.0); // upper half-plane
        let w = cayley_to_disk(z);
        assert!(is_in_unit_disk(w));
        let z_back = cayley_from_disk(w);
        assert_relative_eq!((z_back - z).re, 0.0, epsilon = 1e-10);
        assert_relative_eq!((z_back - z).im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_upper_half_plane() {
        assert!(half_plane::is_in_upper_half_plane(Complex64::new(0.0, 1.0)));
        assert!(!half_plane::is_in_upper_half_plane(Complex64::new(0.0, -1.0)));
    }

    #[test]
    fn test_generalized_circle() {
        use circle_maps::*;
        let c = GeneralizedCircle::circle(Complex64::new(0.0, 0.0), 1.0);
        assert!(c.is_circle());
        assert!(!c.is_line());
        assert!(c.contains(Complex64::new(1.0, 0.0), 1e-10));
    }

    #[test]
    fn test_line_circle() {
        use circle_maps::*;
        let l = GeneralizedCircle::line(1.0, 0.0, 3.0); // x = 3
        assert!(l.is_line());
        assert!(l.contains(Complex64::new(3.0, 5.0), 1e-10));
    }

    #[test]
    fn test_circle_through_three_points_circle() {
        use circle_maps::*;
        let z1 = Complex64::new(1.0, 0.0);
        let z2 = Complex64::new(0.0, 1.0);
        let z3 = Complex64::new(-1.0, 0.0);
        let c = circle_through_three_points(z1, z2, z3).unwrap();
        assert!(c.is_circle());
    }

    #[test]
    fn test_circle_through_three_collinear() {
        use circle_maps::*;
        let z1 = Complex64::new(0.0, 0.0);
        let z2 = Complex64::new(1.0, 0.0);
        let z3 = Complex64::new(2.0, 0.0);
        let c = circle_through_three_points(z1, z2, z3).unwrap();
        assert!(c.is_line());
    }

    #[test]
    fn test_pullback_metric_scale() {
        let cm = ConformalMap::from_complex_derivative(
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        );
        assert_relative_eq!(cm.pullback_metric_scale(), 4.0);
    }
}
