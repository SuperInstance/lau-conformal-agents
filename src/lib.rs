//! # lau-conformal-agents
//!
//! Conformal geometry for agents — transformations that preserve angles but not distances.
//!
//! Conformal maps preserve local shape but allow global deformation. For agents,
//! conformal maps of belief space change the scale but preserve the structure.

pub mod conformal_map;
pub mod mobius;
pub mod liouville;
pub mod conformal_laplacian;
pub mod weyl;
pub mod conformal_weight;
pub mod compactification;
pub mod cft;
pub mod virasoro;
pub mod agent_learning;

pub use conformal_map::*;
pub use mobius::*;
pub use liouville::*;
pub use conformal_laplacian::*;
pub use weyl::*;
pub use conformal_weight::*;
pub use compactification::*;
pub use cft::*;
pub use virasoro::*;
pub use agent_learning::*;
