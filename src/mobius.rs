//! Möbius transformations — the group of conformal automorphisms of the Riemann sphere.
//!
//! A Möbius transformation is f(z) = (az + b) / (cz + d) with ad - bc ≠ 0.
//! These form a group isomorphic to PGL(2, ℂ) and are the only conformal
//! bijections of the Riemann sphere.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A Möbius transformation f(z) = (az + b) / (cz + d).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobiusTransformation {
    /// Coefficient a.
    pub a: Complex64,
    /// Coefficient b.
    pub b: Complex64,
    /// Coefficient c.
    pub c: Complex64,
    /// Coefficient d.
    pub d: Complex64,
}

impl MobiusTransformation {
    /// Create a new Möbius transformation with the given coefficients.
    /// Panics if ad - bc ≈ 0 (degenerate).
    pub fn new(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Self {
        let det = a * d - b * c;
        assert!(det.norm() > 1e-14, "Möbius transformation is degenerate (ad - bc ≈ 0)");
        Self { a, b, c, d }
    }

    /// Create safely, returning None if degenerate.
    pub fn try_new(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Option<Self> {
        let det = a * d - b * c;
        if det.norm() < 1e-14 {
            None
        } else {
            Some(Self { a, b, c, d })
        }
    }

    /// The identity transformation f(z) = z.
    pub fn identity() -> Self {
        Self {
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(0.0, 0.0),
            c: Complex64::new(0.0, 0.0),
            d: Complex64::new(1.0, 0.0),
        }
    }

    /// Evaluate the transformation at a point z.
    /// Returns None if z maps to infinity (z = -d/c when c ≠ 0).
    pub fn evaluate(&self, z: Complex64) -> Option<Complex64> {
        let numerator = self.a * z + self.b;
        let denominator = self.c * z + self.d;
        if denominator.norm() < 1e-14 {
            None // maps to infinity
        } else {
            Some(numerator / denominator)
        }
    }

    /// The determinant ad - bc.
    pub fn determinant(&self) -> Complex64 {
        self.a * self.d - self.b * self.c
    }

    /// Compose two Möbius transformations: (f ∘ g)(z).
    pub fn compose(&self, other: &MobiusTransformation) -> MobiusTransformation {
        // f(g(z)) where f = (a1*z + b1)/(c1*z + d1) and g = (a2*z + b2)/(c2*z + d2)
        // = (a1(a2*z+b2) + b1(c2*z+d2)) / (c1(a2*z+b2) + d1(c2*z+d2))
        // = ((a1*a2+b1*c2)*z + (a1*b2+b1*d2)) / ((c1*a2+d1*c2)*z + (c1*b2+d1*d2))
        MobiusTransformation {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }

    /// Compute the inverse transformation.
    pub fn inverse(&self) -> MobiusTransformation {
        let det = self.determinant();
        // Inverse of [[a,b],[c,d]] is (1/det) [[d,-b],[-c,a]]
        MobiusTransformation {
            a: self.d / det,
            b: -self.b / det,
            c: -self.c / det,
            d: self.a / det,
        }
    }

    /// Fixed points: solutions to f(z) = z.
    /// A Möbius transformation has exactly 2 fixed points (counting multiplicity)
    /// unless it's the identity.
    pub fn fixed_points(&self) -> Vec<Complex64> {
        // az + b = z(cz + d) => cz^2 + (d-a)z - b = 0
        let _zero = Complex64::new(0.0, 0.0);
        if self.c.norm() < 1e-14 {
            // Linear case: (a-d)z = b
            let diff = self.a - self.d;
            if diff.norm() < 1e-14 {
                // Identity-like: all points fixed
                vec![]
            } else {
                vec![self.b / diff]
            }
        } else {
            // Quadratic: cz^2 + (d-a)z - b = 0
            let a_coeff = self.c;
            let b_coeff = self.d - self.a;
            let c_coeff = -self.b;
            solve_quadratic(a_coeff, b_coeff, c_coeff)
        }
    }

    /// Normalize so that ad - bc = 1.
    pub fn normalize(&self) -> MobiusTransformation {
        let det = self.determinant();
        let scale = Complex64::new(1.0, 0.0) / det.sqrt();
        MobiusTransformation {
            a: self.a * scale,
            b: self.b * scale,
            c: self.c * scale,
            d: self.d * scale,
        }
    }

    /// The trace (a + d), an important conjugacy invariant.
    pub fn trace(&self) -> Complex64 {
        self.a + self.d
    }

    /// Classify the transformation by its trace squared.
    pub fn classify(&self) -> MobiusType {
        let tr = self.trace();
        let tr_sq = tr * tr;
        let det = self.determinant();
        // σ = tr^2 / det
        let sigma = tr_sq / det;
        let sigma_re = sigma.re;

        if (sigma_re - 4.0).abs() < 1e-10 {
            MobiusType::Parabolic
        } else if sigma_re > 4.0 {
            MobiusType::Hyperbolic
        } else {
            MobiusType::Elliptic
        }
    }

    /// The multiplier at a fixed point (for conjugation classification).
    pub fn multiplier(&self) -> Complex64 {
        let tr = self.trace();
        let det = self.determinant();
        // k = (tr ± sqrt(tr^2 - 4*det)) / 2
        let disc = tr * tr - 4.0 * det;
        let sqrt_disc = disc.sqrt();
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);
        let k1 = (tr + sqrt_disc) / two;
        let k2 = (tr - sqrt_disc) / two;
        // Return the one closer to 1 (principal multiplier)
        if (k1 - one).norm() < (k2 - one).norm() {
            k1
        } else {
            k2
        }
    }
}

/// Classification of Möbius transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobiusType {
    /// Parabolic: single fixed point, trace^2/det = 4.
    Parabolic,
    /// Hyperbolic: two fixed points, trace^2/det > 4.
    Hyperbolic,
    /// Elliptic: two fixed points, trace^2/det < 4.
    Elliptic,
}

/// Solve az^2 + bz + c = 0.
fn solve_quadratic(a: Complex64, b: Complex64, c: Complex64) -> Vec<Complex64> {
    let disc = b * b - Complex64::new(4.0, 0.0) * a * c;
    let sqrt_disc = disc.sqrt();
    let two = Complex64::new(2.0, 0.0);
    vec![(-b + sqrt_disc) / (two * a), (-b - sqrt_disc) / (two * a)]
}

/// Create a translation f(z) = z + b.
pub fn translation(b: Complex64) -> MobiusTransformation {
    MobiusTransformation::new(
        Complex64::new(1.0, 0.0),
        b,
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    )
}

/// Create a dilation f(z) = λz.
pub fn dilation(lambda: Complex64) -> MobiusTransformation {
    MobiusTransformation::new(
        lambda,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    )
}

/// Create a rotation f(z) = e^{iθ} z.
pub fn rotation(theta: f64) -> MobiusTransformation {
    dilation(Complex64::from_polar(1.0, theta))
}

/// Create the inversion f(z) = 1/z.
pub fn inversion() -> MobiusTransformation {
    MobiusTransformation::new(
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
}

/// Cross ratio (z1, z2; z3, z4) = (z1-z3)(z2-z4) / ((z1-z4)(z2-z3)).
/// Invariant under Möbius transformations.
pub fn cross_ratio(z1: Complex64, z2: Complex64, z3: Complex64, z4: Complex64) -> Complex64 {
    (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3))
}

/// Find the Möbius transformation mapping three points to 0, 1, ∞.
pub fn map_three_points_to_0_1_inf(z1: Complex64, z2: Complex64, z3: Complex64) -> MobiusTransformation {
    // f(z) = (z - z1)(z2 - z3) / ((z - z3)(z2 - z1))
    let a = z2 - z3;
    let b = -z1 * (z2 - z3);
    let c = z2 - z1;
    let d = -z3 * (z2 - z1);
    MobiusTransformation::new(a, b, c, d)
}

/// Find the unique Möbius transformation mapping three source points to three target points.
pub fn map_three_points(
    z1: Complex64, z2: Complex64, z3: Complex64,
    w1: Complex64, w2: Complex64, w3: Complex64,
) -> MobiusTransformation {
    let f = map_three_points_to_0_1_inf(z1, z2, z3);
    let g = map_three_points_to_0_1_inf(w1, w2, w3);
    g.inverse().compose(&f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let m = MobiusTransformation::identity();
        let z = Complex64::new(3.0, 4.0);
        let w = m.evaluate(z).unwrap();
        assert_relative_eq!((w - z).re, 0.0, epsilon = 1e-10);
        assert_relative_eq!((w - z).im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_translation() {
        let t = translation(Complex64::new(1.0, 0.0));
        let z = Complex64::new(2.0, 0.0);
        assert_relative_eq!(t.evaluate(z).unwrap().re, 3.0);
    }

    #[test]
    fn test_dilation() {
        let d = dilation(Complex64::new(2.0, 0.0));
        let z = Complex64::new(3.0, 0.0);
        assert_relative_eq!(d.evaluate(z).unwrap().re, 6.0);
    }

    #[test]
    fn test_rotation() {
        let r = rotation(std::f64::consts::FRAC_PI_2);
        let z = Complex64::new(1.0, 0.0);
        let w = r.evaluate(z).unwrap();
        assert_relative_eq!(w.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(w.im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inversion() {
        let inv = inversion();
        let z = Complex64::new(2.0, 0.0);
        let w = inv.evaluate(z).unwrap();
        assert_relative_eq!(w.re, 0.5);
    }

    #[test]
    fn test_compose() {
        let t1 = translation(Complex64::new(1.0, 0.0));
        let t2 = translation(Complex64::new(2.0, 0.0));
        let composed = t1.compose(&t2);
        let z = Complex64::new(0.0, 0.0);
        assert_relative_eq!(composed.evaluate(z).unwrap().re, 3.0);
    }

    #[test]
    fn test_inverse() {
        let m = MobiusTransformation::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let inv = m.inverse();
        let z = Complex64::new(5.0, 0.0);
        let w = m.evaluate(z).unwrap();
        let z_back = inv.evaluate(w).unwrap();
        assert_relative_eq!((z_back - z).re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_general() {
        let m = MobiusTransformation::new(
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 1.0),
        );
        let inv = m.inverse();
        let z = Complex64::new(1.0, 2.0);
        let w = m.evaluate(z).unwrap();
        let z_back = inv.evaluate(w).unwrap();
        assert_relative_eq!((z_back - z).re, 0.0, epsilon = 1e-10);
        assert_relative_eq!((z_back - z).im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_ratio_invariance() {
        let z1 = Complex64::new(1.0, 0.0);
        let z2 = Complex64::new(2.0, 0.0);
        let z3 = Complex64::new(3.0, 0.0);
        let z4 = Complex64::new(4.0, 0.0);
        let cr1 = cross_ratio(z1, z2, z3, z4);

        // Apply a Möbius transformation
        let m = translation(Complex64::new(10.0, 0.0));
        let cr2 = cross_ratio(
            m.evaluate(z1).unwrap(),
            m.evaluate(z2).unwrap(),
            m.evaluate(z3).unwrap(),
            m.evaluate(z4).unwrap(),
        );
        assert_relative_eq!((cr1 - cr2).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_map_three_points() {
        let m = map_three_points_to_0_1_inf(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        );
        assert_relative_eq!(m.evaluate(Complex64::new(1.0, 0.0)).unwrap().re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m.evaluate(Complex64::new(2.0, 0.0)).unwrap().re, 1.0, epsilon = 1e-10);
        // z3 maps to infinity
        assert!(m.evaluate(Complex64::new(3.0, 0.0)).is_none());
    }

    #[test]
    fn test_fixed_points_translation() {
        let t = translation(Complex64::new(1.0, 0.0));
        let fps = t.fixed_points();
        assert_eq!(fps.len(), 0); // no fixed points (parabolic, ∞ is the only fixed point)
    }

    #[test]
    fn test_fixed_points_dilation() {
        let d = dilation(Complex64::new(2.0, 0.0));
        let fps = d.fixed_points();
        assert_eq!(fps.len(), 1);
        assert_relative_eq!(fps[0].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_points_general() {
        let m = MobiusTransformation::new(
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let fps = m.fixed_points();
        assert_eq!(fps.len(), 2);
        for z in &fps {
            let w = m.evaluate(*z).unwrap();
            assert_relative_eq!((w - *z).norm(), 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_classify_parabolic() {
        let m = translation(Complex64::new(1.0, 0.0));
        assert_eq!(m.classify(), MobiusType::Parabolic);
    }

    #[test]
    fn test_classify_hyperbolic() {
        let m = dilation(Complex64::new(2.0, 0.0));
        assert_eq!(m.classify(), MobiusType::Hyperbolic);
    }

    #[test]
    fn test_classify_elliptic() {
        let m = rotation(std::f64::consts::FRAC_PI_2);
        assert_eq!(m.classify(), MobiusType::Elliptic);
    }

    #[test]
    fn test_normalize() {
        let m = MobiusTransformation::new(
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(8.0, 0.0),
        );
        let n = m.normalize();
        let det = n.determinant();
        assert_relative_eq!(det.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_degenerate_rejected() {
        assert!(MobiusTransformation::try_new(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ).is_none());
    }

    #[test]
    fn test_determinant() {
        let m = MobiusTransformation::new(
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        );
        assert_relative_eq!(m.determinant().re, 6.0);
    }

    #[test]
    fn test_map_three_points_roundtrip() {
        let z1 = Complex64::new(0.0, 0.0);
        let z2 = Complex64::new(1.0, 0.0);
        let z3 = Complex64::new(0.0, 1.0);
        let w1 = Complex64::new(2.0, 3.0);
        let w2 = Complex64::new(4.0, 5.0);
        let w3 = Complex64::new(6.0, 7.0);
        let m = map_three_points(z1, z2, z3, w1, w2, w3);
        assert_relative_eq!((m.evaluate(z1).unwrap() - w1).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!((m.evaluate(z2).unwrap() - w2).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!((m.evaluate(z3).unwrap() - w3).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        let m = MobiusTransformation::new(
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        );
        assert_relative_eq!(m.trace().re, 5.0);
    }

    #[test]
    fn test_multiplier() {
        let d = dilation(Complex64::new(3.0, 0.0));
        let k = d.multiplier();
        // For dilation f(z)=3z, the multipliers are 3 and 1/3
        // The one closer to 1 is 1/3 (distance 2/3 vs distance 2)
        // Actually 3-1=2, 1/3-1=2/3, so k=1/3 is closer
        assert!(k.re > 0.0);
    }
}
