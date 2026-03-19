use std::collections::{HashMap, HashSet};

use crate::models::{
    AgentResponse, ConsensusInput, ConsensusResult, ResponseCluster, WinningResponse,
};
use crate::similarity::{build_similarity_matrix, get_similarity};
use crate::strategies::ConsensusStrategy;

// Payoff values
const REWARD: f64 = 3.0; // both cooperate
const TEMPTATION: f64 = 5.0; // defector vs cooperator
const SUCKER: f64 = 0.0; // cooperator vs defector
const PUNISHMENT: f64 = 1.0; // both defect

const NUM_ROUNDS: usize = 3;
const PENALTY_FACTOR: f64 = 0.2;

pub struct PrisonersDilemmaStrategy;

impl ConsensusStrategy for PrisonersDilemmaStrategy {
    fn name(&self) -> &str {
        "prisoners_dilemma"
    }

    fn evaluate(
        &self,
        input: &ConsensusInput,
        similarity_threshold: f64,
    ) -> ConsensusResult {
        let responses = &input.responses;
        let n = responses.len();
        let matrix = build_similarity_matrix(responses);

        let mut effective_confidence: HashMap<String, f64> = responses
            .iter()
            .map(|r| (r.agent_id.clone(), r.confidence))
            .collect();

        let mut cumulative_payoffs: HashMap<String, f64> = responses
            .iter()
            .map(|r| (r.agent_id.clone(), 0.0))
            .collect();

        for _round in 0..NUM_ROUNDS {
            let cooperator_ids = classify_cooperator_ids(
                responses,
                &matrix,
                &effective_confidence,
                similarity_threshold,
            );

            // Compute pairwise payoffs
            let mut round_payoffs: HashMap<String, f64> = responses
                .iter()
                .map(|r| (r.agent_id.clone(), 0.0))
                .collect();

            for i in 0..responses.len() {
                for j in (i + 1)..responses.len() {
                    let r1_coop = cooperator_ids.contains(&responses[i].agent_id);
                    let r2_coop = cooperator_ids.contains(&responses[j].agent_id);
                    let (p1, p2) = payoff(r1_coop, r2_coop);
                    *round_payoffs.get_mut(&responses[i].agent_id).unwrap() += p1;
                    *round_payoffs.get_mut(&responses[j].agent_id).unwrap() += p2;
                }
            }

            // Accumulate
            for (agent_id, pay) in &round_payoffs {
                *cumulative_payoffs.get_mut(agent_id).unwrap() += pay;
            }

            // Penalize defectors
            for r in responses {
                if !cooperator_ids.contains(&r.agent_id) {
                    let conf = effective_confidence.get_mut(&r.agent_id).unwrap();
                    *conf *= 1.0 - PENALTY_FACTOR;
                }
            }
        }

        // Final classification
        let cooperator_ids = classify_cooperator_ids(
            responses,
            &matrix,
            &effective_confidence,
            similarity_threshold,
        );
        let cooperators: Vec<&AgentResponse> = responses
            .iter()
            .filter(|r| cooperator_ids.contains(&r.agent_id))
            .collect();
        let defectors: Vec<&AgentResponse> = responses
            .iter()
            .filter(|r| !cooperator_ids.contains(&r.agent_id))
            .collect();

        let consensus_reached = cooperators.len() >= 2;

        let mut clusters = Vec::new();
        if let Some(c) = build_cluster(&cooperators, &matrix) {
            clusters.push(c);
        }
        if let Some(c) = build_cluster(&defectors, &matrix) {
            clusters.push(c);
        }

        let winning_response = if consensus_reached {
            let winner = cooperators
                .iter()
                .max_by(|a, b| {
                    cumulative_payoffs[&a.agent_id]
                        .partial_cmp(&cumulative_payoffs[&b.agent_id])
                        .unwrap()
                })
                .unwrap();
            let max_possible = TEMPTATION * (n as f64 - 1.0) * NUM_ROUNDS as f64;
            let confidence_score = if max_possible > 0.0 {
                cumulative_payoffs[&winner.agent_id] / max_possible
            } else {
                0.0
            };
            Some(WinningResponse {
                content: winner.content.clone(),
                supporting_agents: cooperators.iter().map(|r| r.agent_id.clone()).collect(),
                agreement_ratio: round3(cooperators.len() as f64 / n as f64),
                confidence_score: round3(confidence_score),
            })
        } else {
            None
        };

        let final_payoffs: serde_json::Value = cumulative_payoffs
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(round3(*v))))
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        ConsensusResult {
            strategy: "prisoners_dilemma".to_string(),
            prompt: input.prompt.clone(),
            consensus_reached,
            winning_response,
            clusters,
            metadata: serde_json::json!({
                "total_responses": n,
                "similarity_threshold": similarity_threshold,
                "num_rounds": NUM_ROUNDS,
                "final_payoffs": final_payoffs,
                "cooperators": cooperators.iter().map(|r| r.agent_id.clone()).collect::<Vec<_>>(),
                "defectors": defectors.iter().map(|r| r.agent_id.clone()).collect::<Vec<_>>(),
            }),
        }
    }
}

fn classify_cooperator_ids(
    responses: &[AgentResponse],
    matrix: &HashMap<(String, String), f64>,
    effective_confidence: &HashMap<String, f64>,
    threshold: f64,
) -> HashSet<String> {
    let mut cooperator_ids = HashSet::new();

    for r in responses {
        let others: Vec<&AgentResponse> = responses
            .iter()
            .filter(|o| o.agent_id != r.agent_id)
            .collect();

        if others.is_empty() {
            cooperator_ids.insert(r.agent_id.clone());
            continue;
        }

        let weighted_sum: f64 = others
            .iter()
            .map(|o| {
                get_similarity(matrix, &r.agent_id, &o.agent_id)
                    * effective_confidence[&o.agent_id]
            })
            .sum();
        let weight_total: f64 = others
            .iter()
            .map(|o| effective_confidence[&o.agent_id])
            .sum();

        let avg_sim = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        };

        if avg_sim >= threshold {
            cooperator_ids.insert(r.agent_id.clone());
        }
    }

    cooperator_ids
}

fn payoff(r1_coop: bool, r2_coop: bool) -> (f64, f64) {
    match (r1_coop, r2_coop) {
        (true, true) => (REWARD, REWARD),
        (true, false) => (SUCKER, TEMPTATION),
        (false, true) => (TEMPTATION, SUCKER),
        (false, false) => (PUNISHMENT, PUNISHMENT),
    }
}

fn build_cluster(
    agents: &[&AgentResponse],
    matrix: &HashMap<(String, String), f64>,
) -> Option<ResponseCluster> {
    if agents.is_empty() {
        return None;
    }
    let representative = agents
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
        .unwrap();
    let members: Vec<String> = agents.iter().map(|r| r.agent_id.clone()).collect();
    let mut sim_scores: HashMap<String, f64> = HashMap::new();
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let key = format!("{}:{}", agents[i].agent_id, agents[j].agent_id);
            sim_scores.insert(
                key,
                get_similarity(matrix, &agents[i].agent_id, &agents[j].agent_id),
            );
        }
    }
    Some(ResponseCluster {
        representative_content: representative.content.clone(),
        members,
        similarity_scores: sim_scores,
    })
}

fn round3(val: f64) -> f64 {
    (val * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(responses: Vec<(&str, &str, f64)>) -> ConsensusInput {
        ConsensusInput {
            prompt: "test?".to_string(),
            responses: responses
                .into_iter()
                .map(|(id, content, confidence)| AgentResponse {
                    agent_id: id.to_string(),
                    content: content.to_string(),
                    confidence,
                    metadata: Default::default(),
                })
                .collect(),
        }
    }

    #[test]
    fn test_cooperators_reach_consensus() {
        let input = make_input(vec![
            ("a1", "Paris is the capital.", 0.95),
            ("a2", "The capital is Paris.", 0.90),
            ("a3", "Paris, the capital.", 0.85),
        ]);
        let strategy = PrisonersDilemmaStrategy;
        let result = strategy.evaluate(&input, 0.5);
        assert!(result.consensus_reached);
        let winner = result.winning_response.unwrap();
        assert!(winner.supporting_agents.len() >= 2);
    }

    #[test]
    fn test_defector_detected() {
        let input = make_input(vec![
            ("a1", "The answer is 4.", 0.99),
            ("a2", "The answer is 4.", 0.95),
            ("a3", "Something completely different and unrelated.", 0.60),
        ]);
        let strategy = PrisonersDilemmaStrategy;
        let result = strategy.evaluate(&input, 0.6);
        assert!(result.consensus_reached);
        let defectors = result.metadata["defectors"].as_array().unwrap();
        assert!(!defectors.is_empty());
    }

    #[test]
    fn test_all_disagree() {
        let input = make_input(vec![
            ("a1", "apples", 0.8),
            ("a2", "oranges", 0.8),
        ]);
        let strategy = PrisonersDilemmaStrategy;
        let result = strategy.evaluate(&input, 0.8);
        // With only 2 agents both disagreeing, no consensus
        assert!(!result.consensus_reached);
    }

    #[test]
    fn test_payoff_values() {
        assert_eq!(payoff(true, true), (REWARD, REWARD));
        assert_eq!(payoff(true, false), (SUCKER, TEMPTATION));
        assert_eq!(payoff(false, true), (TEMPTATION, SUCKER));
        assert_eq!(payoff(false, false), (PUNISHMENT, PUNISHMENT));
    }

    #[test]
    fn test_metadata_fields() {
        let input = make_input(vec![
            ("a1", "yes", 0.9),
            ("a2", "yes", 0.8),
            ("a3", "no", 0.7),
        ]);
        let strategy = PrisonersDilemmaStrategy;
        let result = strategy.evaluate(&input, 0.5);
        assert_eq!(result.metadata["total_responses"], 3);
        assert_eq!(result.metadata["num_rounds"], NUM_ROUNDS);
        assert!(result.metadata["final_payoffs"].is_object());
        assert!(result.metadata["cooperators"].is_array());
        assert!(result.metadata["defectors"].is_array());
    }

    #[test]
    fn test_penalty_reduces_defector_influence() {
        // All similar content — everyone cooperates with low threshold
        let input = make_input(vec![
            ("a1", "hello world", 0.9),
            ("a2", "hello world", 0.9),
            ("a3", "hello world", 0.9),
        ]);
        let strategy = PrisonersDilemmaStrategy;
        let result = strategy.evaluate(&input, 0.1);
        assert!(result.consensus_reached);
        // All should be cooperators
        let cooperators = result.metadata["cooperators"].as_array().unwrap();
        assert_eq!(cooperators.len(), 3);
    }

    #[test]
    fn test_build_cluster_empty() {
        let matrix = HashMap::new();
        let agents: Vec<&AgentResponse> = vec![];
        assert!(build_cluster(&agents, &matrix).is_none());
    }

    #[test]
    fn test_classify_single_agent() {
        let r = AgentResponse {
            agent_id: "a1".to_string(),
            content: "hello".to_string(),
            confidence: 1.0,
            metadata: Default::default(),
        };
        let responses = vec![r];
        let matrix = build_similarity_matrix(&responses);
        let eff: HashMap<String, f64> = [("a1".to_string(), 1.0)].into();
        let ids = classify_cooperator_ids(&responses, &matrix, &eff, 0.5);
        assert!(ids.contains("a1"));
    }
}
