pub mod prisoners_dilemma;
pub mod quorum;

use crate::models::{ConsensusInput, ConsensusResult};

/// Trait for consensus strategies.
pub trait ConsensusStrategy {
    fn name(&self) -> &str;
    fn evaluate(
        &self,
        input: &ConsensusInput,
        similarity_threshold: f64,
    ) -> ConsensusResult;
}

/// Available strategy names.
pub const STRATEGY_NAMES: &[&str] = &["quorum", "prisoners_dilemma"];

/// Create a strategy by name.
pub fn get_strategy(name: &str) -> Result<Box<dyn ConsensusStrategy>, String> {
    match name {
        "quorum" => Ok(Box::new(quorum::QuorumStrategy)),
        "prisoners_dilemma" => Ok(Box::new(prisoners_dilemma::PrisonersDilemmaStrategy)),
        _ => Err(format!(
            "Unknown strategy: {name}. Available: {}",
            STRATEGY_NAMES.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_strategy_quorum() {
        let s = get_strategy("quorum").unwrap();
        assert_eq!(s.name(), "quorum");
    }

    #[test]
    fn test_get_strategy_prisoners_dilemma() {
        let s = get_strategy("prisoners_dilemma").unwrap();
        assert_eq!(s.name(), "prisoners_dilemma");
    }

    #[test]
    fn test_get_strategy_unknown() {
        let err = get_strategy("unknown").err().unwrap();
        assert!(err.contains("Unknown strategy"));
    }
}
