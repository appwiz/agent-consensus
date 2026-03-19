use std::collections::HashSet;

use crate::models::{ConsensusInput, ConsensusResult};
use crate::strategies::get_strategy;

#[derive(Debug)]
pub struct ConsensusEngine {
    strategy_name: String,
    similarity_threshold: f64,
}

impl ConsensusEngine {
    pub fn new(strategy_name: &str, similarity_threshold: f64) -> Result<Self, String> {
        // Validate strategy exists
        get_strategy(strategy_name)?;
        Ok(Self {
            strategy_name: strategy_name.to_string(),
            similarity_threshold,
        })
    }

    pub fn run(&self, input: &ConsensusInput) -> Result<ConsensusResult, String> {
        self.validate(input)?;
        let strategy = get_strategy(&self.strategy_name)?;
        Ok(strategy.evaluate(input, self.similarity_threshold))
    }

    fn validate(&self, input: &ConsensusInput) -> Result<(), String> {
        if input.responses.len() < 2 {
            return Err("At least 2 agent responses required for consensus".to_string());
        }
        let mut ids = HashSet::new();
        for r in &input.responses {
            if !ids.insert(&r.agent_id) {
                return Err("Duplicate agent_id values found".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::AgentResponse;

    fn make_response(id: &str, content: &str, confidence: f64) -> AgentResponse {
        AgentResponse {
            agent_id: id.to_string(),
            content: content.to_string(),
            confidence,
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_engine_quorum() {
        let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
        let input = ConsensusInput {
            prompt: "test?".to_string(),
            responses: vec![
                make_response("a1", "yes", 0.9),
                make_response("a2", "yes", 0.8),
            ],
        };
        let result = engine.run(&input).unwrap();
        assert_eq!(result.strategy, "quorum");
    }

    #[test]
    fn test_engine_prisoners_dilemma() {
        let engine = ConsensusEngine::new("prisoners_dilemma", 0.5).unwrap();
        let input = ConsensusInput {
            prompt: "test?".to_string(),
            responses: vec![
                make_response("a1", "yes", 0.9),
                make_response("a2", "yes", 0.8),
            ],
        };
        let result = engine.run(&input).unwrap();
        assert_eq!(result.strategy, "prisoners_dilemma");
    }

    #[test]
    fn test_engine_invalid_strategy() {
        let err = ConsensusEngine::new("invalid", 0.6).err().unwrap();
        assert!(err.contains("Unknown strategy"));
    }

    #[test]
    fn test_engine_too_few_responses() {
        let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
        let input = ConsensusInput {
            prompt: "test?".to_string(),
            responses: vec![make_response("a1", "yes", 0.9)],
        };
        let err = engine.run(&input).unwrap_err();
        assert!(err.contains("At least 2"));
    }

    #[test]
    fn test_engine_duplicate_ids() {
        let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
        let input = ConsensusInput {
            prompt: "test?".to_string(),
            responses: vec![
                make_response("a1", "yes", 0.9),
                make_response("a1", "no", 0.8),
            ],
        };
        let err = engine.run(&input).unwrap_err();
        assert!(err.contains("Duplicate"));
    }
}
