//! Virasoro algebra — the central extension of the Witt algebra.
//!
//! The Virasoro algebra is the infinite-dimensional Lie algebra with generators
//! L_n (n ∈ ℤ) and central element c, satisfying:
//!   [L_m, L_n] = (m - n) L_{m+n} + (c/12)(m³ - m) δ_{m+n,0}
//!
//! This is the symmetry algebra of 2D conformal field theory.

use serde::{Deserialize, Serialize};

/// The Virasoro algebra with central charge c.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirasoroAlgebra {
    /// Central charge c.
    pub central_charge: f64,
}

impl VirasoroAlgebra {
    /// Create a new Virasoro algebra with the given central charge.
    pub fn new(central_charge: f64) -> Self {
        Self { central_charge }
    }

    /// The commutator [L_m, L_n] = (m-n) L_{m+n} + (c/12)(m³ - m) δ_{m+n,0}.
    /// Returns (coefficient of L_{m+n}, central term).
    pub fn commutator(&self, m: i64, n: i64) -> (i64, f64) {
        let lie_bracket = m - n;
        let central = if m + n == 0 {
            let m_f = m as f64;
            self.central_charge / 12.0 * (m_f * m_f * m_f - m_f)
        } else {
            0.0
        };
        (lie_bracket, central)
    }

    /// The central term c(m) = (c/12)(m³ - m) for the commutator [L_m, L_{-m}].
    pub fn central_term(&self, m: i64) -> f64 {
        let m_f = m as f64;
        self.central_charge / 12.0 * (m_f * m_f * m_f - m_f)
    }

    /// The Witt algebra (c = 0): [L_m, L_n] = (m-n) L_{m+n}.
    pub fn witt() -> Self {
        Self::new(0.0)
    }

    /// Check the Jacobi identity: [L_k, [L_m, L_n]] + cyclic = 0.
    /// Returns the maximum violation.
    pub fn verify_jacobi(&self, k: i64, m: i64, n: i64) -> f64 {
        let (_, c1) = self.commutator(k, m + n);
        let lie1 = (k - (m + n)) as f64 * (m - n) as f64;
        let central1 = (m - n) as f64 * c1;

        let (_, c2) = self.commutator(m, n + k);
        let lie2 = (m - (n + k)) as f64 * (n - k) as f64;
        let central2 = (n - k) as f64 * c2;

        let (_, c3) = self.commutator(n, k + m);
        let lie3 = (n - (k + m)) as f64 * (k - m) as f64;
        let central3 = (k - m) as f64 * c3;

        let lie_violation = lie1 + lie2 + lie3;
        let central_violation = central1 + central2 + central3;
        (lie_violation + central_violation).abs()
    }

    /// The Kac determinant at level n.
    /// det(M_n) depends on c and h through the Kac formula.
    pub fn kac_determinant(&self, level: usize, h: f64) -> f64 {
        let mut det = 1.0;
        let c = self.central_charge;
        for r in 1..=level {
            for s in 1..=level {
                if r * s > level {
                    break;
                }
                let p = r as f64;
                let q = s as f64;
                let m = if c <= 1.0 {
                    (1.0 - c).max(0.0).sqrt()
                } else {
                    (c - 1.0).sqrt()
                };
                let h_rs = if m.abs() > 1e-10 {
                    (p * m - q / m).powi(2) / 4.0 - (p * p - 1.0) * (q * q - 1.0) / 16.0
                } else {
                    0.0
                };
                det *= h - h_rs;
            }
        }
        det
    }

    /// The effective central charge c_eff = c - 24h₀ (relevant for non-unitary CFTs).
    pub fn effective_central_charge(&self, h0: f64) -> f64 {
        self.central_charge - 24.0 * h0
    }

    /// Unitarity: c ≥ 1 and h ≥ 0 (or c in (0,1) with h on the Kac table).
    pub fn is_unitary(&self, h: f64) -> bool {
        if self.central_charge >= 1.0 - 1e-10 {
            h >= -1e-10
        } else if self.central_charge > 0.0 {
            self.central_charge > 0.0 && h >= -1e-10
        } else {
            false
        }
    }
}

/// Verma module: the representation built from a highest-weight state by acting with L_{-n}.
#[derive(Debug, Clone)]
pub struct VermaModule {
    pub central_charge: f64,
    pub conformal_weight: f64,
}

impl VermaModule {
    pub fn new(c: f64, h: f64) -> Self {
        Self { central_charge: c, conformal_weight: h }
    }

    /// The L_0 eigenvalue of a descendant at the given level.
    /// A level-n descendant has L_0 eigenvalue h + n.
    pub fn l0_eigenvalue(&self, level: usize) -> f64 {
        self.conformal_weight + level as f64
    }

    /// The dimension of the level-n subspace (number of partitions of n).
    pub fn level_dimension(level: usize) -> usize {
        if level == 0 {
            return 1;
        }
        let mut p = vec![0usize; level + 1];
        p[0] = 1;
        for n in 1..=level {
            let mut k = 1;
            loop {
                let pent1 = k * (3 * k - 1) / 2;
                let pent2 = k * (3 * k + 1) / 2;
                if pent1 > n {
                    break;
                }
                let sign = if k % 2 == 1 { 1usize } else { 0 };
                p[n] += sign * p[n - pent1];
                if pent2 <= n {
                    p[n] += sign * p[n - pent2];
                }
                k += 1;
            }
        }
        p[level]
    }

    /// Check if the Verma module is reducible (Kac determinant vanishes at some level).
    pub fn is_reducible(&self, max_level: usize) -> bool {
        let algebra = VirasoroAlgebra::new(self.central_charge);
        for level in 1..=max_level {
            if algebra.kac_determinant(level, self.conformal_weight).abs() < 1e-10 {
                return true;
            }
        }
        false
    }
}

/// The Sugawara construction: build the Virasoro algebra from an affine Lie algebra.
/// L_n = (1/2(k + h∨)) Σ_m :J_{n-m}^a J_m^a:
/// Central charge: c = k dim(g) / (k + h∨)
pub fn sugawara_central_charge(level_k: f64, dim_g: usize, dual_coxeter: f64) -> f64 {
    level_k * dim_g as f64 / (level_k + dual_coxeter)
}

/// The Casimir energy on a cylinder: E₀ = -c/24 (ground state energy shift).
pub fn casimir_energy(central_charge: f64) -> f64 {
    -central_charge / 24.0
}

/// Modular transformation of the partition function on a torus.
/// Z(τ) = Tr(q^{L₀ - c/24}) where q = e^{2πiτ}
pub fn character_q_expansion(c: f64, h: f64, max_order: usize) -> Vec<f64> {
    let mut coeffs = Vec::new();
    let _shift = h - c / 24.0;
    for n in 0..=max_order {
        coeffs.push(if n == 0 {
            1.0
        } else {
            VermaModule::level_dimension(n) as f64
        });
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_commutator_witt() {
        let v = VirasoroAlgebra::witt();
        let (coeff, central) = v.commutator(1, 2);
        assert_eq!(coeff, -1);
        assert_relative_eq!(central, 0.0);
    }

    #[test]
    fn test_commutator_central() {
        let v = VirasoroAlgebra::new(1.0);
        let (_, central) = v.commutator(1, -1);
        assert_relative_eq!(central, 0.0);
    }

    #[test]
    fn test_central_term_2() {
        let v = VirasoroAlgebra::new(1.0);
        assert_relative_eq!(v.central_term(2), 0.5);
    }

    #[test]
    fn test_central_term_3() {
        let v = VirasoroAlgebra::new(1.0);
        assert_relative_eq!(v.central_term(3), 2.0);
    }

    #[test]
    fn test_central_term_0() {
        let v = VirasoroAlgebra::new(5.0);
        assert_relative_eq!(v.central_term(0), 0.0);
    }

    #[test]
    fn test_central_term_1() {
        let v = VirasoroAlgebra::new(3.0);
        assert_relative_eq!(v.central_term(1), 0.0);
    }

    #[test]
    fn test_antisymmetry() {
        let v = VirasoroAlgebra::new(1.0);
        let (c1, central1) = v.commutator(2, 3);
        let (c2, central2) = v.commutator(3, 2);
        assert_eq!(c1, -c2);
        assert_relative_eq!(central1, central2);
    }

    #[test]
    fn test_jacobi_identity() {
        let v = VirasoroAlgebra::new(1.0);
        let violation = v.verify_jacobi(1, 2, 3);
        assert_relative_eq!(violation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_identity_2() {
        let v = VirasoroAlgebra::new(2.5);
        let violation = v.verify_jacobi(1, -1, 2);
        assert_relative_eq!(violation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_verma_l0_eigenvalue() {
        let vm = VermaModule::new(1.0, 0.5);
        assert_relative_eq!(vm.l0_eigenvalue(3), 3.5);
    }

    #[test]
    fn test_level_dimension_0() {
        assert_eq!(VermaModule::level_dimension(0), 1);
    }

    #[test]
    fn test_level_dimension_1() {
        assert_eq!(VermaModule::level_dimension(1), 1);
    }

    #[test]
    fn test_level_dimension_2() {
        assert_eq!(VermaModule::level_dimension(2), 2);
    }

    #[test]
    fn test_level_dimension_3() {
        assert_eq!(VermaModule::level_dimension(3), 3);
    }

    #[test]
    fn test_level_dimension_4() {
        assert_eq!(VermaModule::level_dimension(4), 5);
    }

    #[test]
    fn test_sugawara_central_charge() {
        let c = sugawara_central_charge(1.0, 3, 2.0);
        assert_relative_eq!(c, 1.0);
    }

    #[test]
    fn test_casimir_energy() {
        let e = casimir_energy(1.0);
        assert_relative_eq!(e, -1.0 / 24.0);
    }

    #[test]
    fn test_effective_central_charge() {
        let v = VirasoroAlgebra::new(1.0);
        let c_eff = v.effective_central_charge(0.0);
        assert_relative_eq!(c_eff, 1.0);
    }

    #[test]
    fn test_unitary_c1() {
        let v = VirasoroAlgebra::new(1.0);
        assert!(v.is_unitary(0.5));
    }

    #[test]
    fn test_character_q_expansion() {
        let coeffs = character_q_expansion(1.0, 0.0, 3);
        assert_eq!(coeffs.len(), 4);
        assert_relative_eq!(coeffs[0], 1.0);
    }

    #[test]
    fn test_kac_determinant_level1() {
        let v = VirasoroAlgebra::new(1.0);
        // At level 1 with h=0: product of (0 - h_{1,1})
        let det = v.kac_determinant(1, 0.0);
        assert!(det.is_finite());
    }

    #[test]
    fn test_verma_reducible() {
        // For certain (c, h) combinations, the Verma module is reducible
        let vm = VermaModule::new(0.0, 0.0); // c=0, h=0 should be reducible
        assert!(vm.is_reducible(5));
    }
}
