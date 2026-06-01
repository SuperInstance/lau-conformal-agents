//! Agent learning through conformal geometry.
//!
//! Applications of conformal geometry to learning agents:
//! conformal prediction, belief space rescaling, invariant feature extraction.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Conformal prediction set for uncertainty quantification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionSet<T: Clone> {
    /// The prediction set.
    pub predictions: Vec<T>,
    /// Confidence level (1 - α).
    pub confidence: f64,
    /// Nonconformity scores used.
    pub scores: Vec<f64>,
    /// The quantile threshold.
    pub threshold: f64,
}

/// Nonconformity scores for conformal prediction.
pub mod nonconformity {
    /// Simple absolute residual score.
    pub fn absolute_residual(prediction: f64, observation: f64) -> f64 {
        (prediction - observation).abs()
    }

    /// Normalized residual score.
    pub fn normalized_residual(prediction: f64, observation: f64, scale: f64) -> f64 {
        if scale.abs() < 1e-15 {
            0.0
        } else {
            (prediction - observation).abs() / scale
        }
    }

    /// Rank-based score for classification.
    pub fn rank_score(probabilities: &[f64], true_class: usize) -> f64 {
        if true_class >= probabilities.len() {
            return f64::MAX;
        }
        // Score = 1 - p(true_class) + sum of p(other_class) where p(other) > p(true)
        let true_prob = probabilities[true_class];
        probabilities.iter().filter(|&&p| p > true_prob).map(|&p| p - true_prob).sum::<f64>()
            + (1.0 - true_prob)
    }
}

/// Compute a conformal prediction interval.
pub fn conformal_prediction_interval(
    calibration_scores: &[f64],
    alpha: f64,
    prediction: f64,
) -> (f64, f64) {
    let n = calibration_scores.len();
    if n == 0 {
        return (prediction, prediction);
    }
    let mut sorted_scores = calibration_scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Quantile: ceil((n+1)(1-α))/n
    let quantile_idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
    let idx = quantile_idx.saturating_sub(1).min(n - 1);
    let threshold = sorted_scores[idx];
    (prediction - threshold, prediction + threshold)
}

/// Conformal rescaling of a belief vector.
/// Rescales beliefs while preserving the angular structure.
pub fn conformal_rescale_belief(belief: &DVector<f64>, factor: f64) -> DVector<f64> {
    belief.scale(factor)
}

/// Compute the angle between two belief vectors (conformal invariant).
pub fn belief_angle(b1: &DVector<f64>, b2: &DVector<f64>) -> f64 {
    let dot = b1.dot(b2);
    let norm1 = b1.norm();
    let norm2 = b2.norm();
    if norm1 < 1e-15 || norm2 < 1e-15 {
        return 0.0;
    }
    let cos_theta = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Conformal invariant features: extract features that are invariant under conformal rescaling.
pub fn conformal_invariant_features(beliefs: &[DVector<f64>]) -> Vec<f64> {
    if beliefs.is_empty() {
        return vec![];
    }
    let mut angles = Vec::new();
    for i in 0..beliefs.len() {
        for j in (i + 1)..beliefs.len() {
            angles.push(belief_angle(&beliefs[i], &beliefs[j]));
        }
    }
    angles
}

/// Adaptive conformal inference: update the miscoverage level online.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConformal {
    /// Current α level.
    pub alpha: f64,
    /// Target miscoverage level.
    pub target_alpha: f64,
    /// Learning rate.
    pub gamma: f64,
}

impl AdaptiveConformal {
    pub fn new(target_alpha: f64, gamma: f64) -> Self {
        Self { alpha: target_alpha, target_alpha, gamma }
    }

    /// Update α after observing whether the true value was in the prediction set.
    pub fn update(&mut self, covered: bool) {
        let err_t = if covered { 0.0 } else { 1.0 };
        self.alpha += self.gamma * (err_t - self.target_alpha);
        self.alpha = self.alpha.clamp(0.001, 0.999);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_absolute_residual() {
        let score = nonconformity::absolute_residual(3.0, 5.0);
        assert_relative_eq!(score, 2.0);
    }

    #[test]
    fn test_normalized_residual() {
        let score = nonconformity::normalized_residual(3.0, 5.0, 2.0);
        assert_relative_eq!(score, 1.0);
    }

    #[test]
    fn test_rank_score() {
        let probs = vec![0.1, 0.6, 0.3];
        let score = nonconformity::rank_score(&probs, 0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_conformal_prediction_interval() {
        let scores = vec![0.5, 1.0, 0.3, 0.8, 0.2];
        let (lo, hi) = conformal_prediction_interval(&scores, 0.1, 10.0);
        assert!(lo <= 10.0);
        assert!(hi >= 10.0);
    }

    #[test]
    fn test_conformal_prediction_interval_empty() {
        let (lo, hi) = conformal_prediction_interval(&[], 0.1, 5.0);
        assert_relative_eq!(lo, 5.0);
        assert_relative_eq!(hi, 5.0);
    }

    #[test]
    fn test_conformal_rescale_belief() {
        let belief = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let rescaled = conformal_rescale_belief(&belief, 2.0);
        assert_relative_eq!(rescaled[0], 2.0);
        assert_relative_eq!(rescaled[1], 4.0);
    }

    #[test]
    fn test_belief_angle_parallel() {
        let b1 = DVector::from_vec(vec![1.0, 0.0]);
        let b2 = DVector::from_vec(vec![2.0, 0.0]);
        assert_relative_eq!(belief_angle(&b1, &b2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_angle_orthogonal() {
        let b1 = DVector::from_vec(vec![1.0, 0.0]);
        let b2 = DVector::from_vec(vec![0.0, 1.0]);
        assert_relative_eq!(belief_angle(&b1, &b2), std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_angle_opposite() {
        let b1 = DVector::from_vec(vec![1.0, 0.0]);
        let b2 = DVector::from_vec(vec![-1.0, 0.0]);
        assert_relative_eq!(belief_angle(&b1, &b2), std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_conformal_invariant_features() {
        let beliefs = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let features = conformal_invariant_features(&beliefs);
        assert_eq!(features.len(), 1);
        assert_relative_eq!(features[0], std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_conformal_creation() {
        let ac = AdaptiveConformal::new(0.1, 0.01);
        assert_relative_eq!(ac.alpha, 0.1);
    }

    #[test]
    fn test_adaptive_conformal_update_covered() {
        let mut ac = AdaptiveConformal::new(0.1, 0.1);
        ac.update(true);
        assert!(ac.alpha < 0.1); // α decreases when covered
    }

    #[test]
    fn test_adaptive_conformal_update_not_covered() {
        let mut ac = AdaptiveConformal::new(0.1, 0.1);
        ac.update(false);
        assert!(ac.alpha > 0.1); // α increases when not covered
    }

    #[test]
    fn test_normalized_residual_zero_scale() {
        let score = nonconformity::normalized_residual(1.0, 2.0, 0.0);
        assert_relative_eq!(score, 0.0);
    }
}
