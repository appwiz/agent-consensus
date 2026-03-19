use std::collections::HashMap;

use crate::models::{
    AgentResponse, ConsensusInput, ConsensusResult, ResponseCluster, WinningResponse,
};
use crate::similarity::{build_similarity_matrix, get_similarity};
use crate::strategies::ConsensusStrategy;

pub struct QuorumStrategy;

impl ConsensusStrategy for QuorumStrategy {
    fn name(&self) -> &str {
        "quorum"
    }

    fn evaluate(
        &self,
        input: &ConsensusInput,
        similarity_threshold: f64,
    ) -> ConsensusResult {
        let responses = &input.responses;
        let n = responses.len();
        let quorum_size = n / 2 + 1;

        let matrix = build_similarity_matrix(responses);
        let mut clusters = cluster_responses(responses, &matrix, similarity_threshold);

        // Sort clusters by size descending, then by mean confidence descending
        clusters.sort_by(|a, b| {
            let size_cmp = b.len().cmp(&a.len());
            if size_cmp != std::cmp::Ordering::Equal {
                return size_cmp;
            }
            mean_confidence(b)
                .partial_cmp(&mean_confidence(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let largest = &clusters[0];
        let consensus_reached = largest.len() >= quorum_size;

        // Build cluster output objects
        let response_clusters: Vec<ResponseCluster> = clusters
            .iter()
            .map(|cluster| build_response_cluster(cluster, &matrix))
            .collect();

        let winning_response = if consensus_reached {
            let representative = largest
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();
            Some(WinningResponse {
                content: representative.content.clone(),
                supporting_agents: largest.iter().map(|r| r.agent_id.clone()).collect(),
                agreement_ratio: round3(largest.len() as f64 / n as f64),
                confidence_score: round3(mean_confidence(largest)),
            })
        } else {
            None
        };

        ConsensusResult {
            strategy: "quorum".to_string(),
            prompt: input.prompt.clone(),
            consensus_reached,
            winning_response,
            clusters: response_clusters,
            metadata: serde_json::json!({
                "total_responses": n,
                "similarity_threshold": similarity_threshold,
                "quorum_size": quorum_size,
            }),
        }
    }
}

/// Single-linkage clustering: merge clusters if any pair exceeds threshold.
fn cluster_responses(
    responses: &[AgentResponse],
    matrix: &HashMap<(String, String), f64>,
    threshold: f64,
) -> Vec<Vec<AgentResponse>> {
    let mut clusters: Vec<Vec<AgentResponse>> =
        responses.iter().map(|r| vec![r.clone()]).collect();

    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                if clusters_linked(&clusters[i], &clusters[j], matrix, threshold) {
                    let merged: Vec<AgentResponse> = clusters[j].clone();
                    clusters[i].extend(merged);
                    clusters.remove(j);
                    changed = true;
                    break 'outer;
                }
            }
        }
    }

    clusters
}

/// Check if any member of c1 is similar enough to any member of c2.
fn clusters_linked(
    c1: &[AgentResponse],
    c2: &[AgentResponse],
    matrix: &HashMap<(String, String), f64>,
    threshold: f64,
) -> bool {
    for r1 in c1 {
        for r2 in c2 {
            if get_similarity(matrix, &r1.agent_id, &r2.agent_id) >= threshold {
                return true;
            }
        }
    }
    false
}

fn build_response_cluster(
    cluster: &[AgentResponse],
    matrix: &HashMap<(String, String), f64>,
) -> ResponseCluster {
    let representative = cluster
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
        .unwrap();
    let members: Vec<String> = cluster.iter().map(|r| r.agent_id.clone()).collect();
    let mut sim_scores: HashMap<String, f64> = HashMap::new();
    for i in 0..cluster.len() {
        for j in (i + 1)..cluster.len() {
            let key = format!("{}:{}", cluster[i].agent_id, cluster[j].agent_id);
            sim_scores.insert(
                key,
                get_similarity(matrix, &cluster[i].agent_id, &cluster[j].agent_id),
            );
        }
    }
    ResponseCluster {
        representative_content: representative.content.clone(),
        members,
        similarity_scores: sim_scores,
    }
}

fn mean_confidence(cluster: &[AgentResponse]) -> f64 {
    let sum: f64 = cluster.iter().map(|r| r.confidence).sum();
    sum / cluster.len() as f64
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
    fn test_unanimous_consensus() {
        let input = make_input(vec![
            ("a1", "Paris is the capital of France.", 0.95),
            ("a2", "The capital of France is Paris.", 0.90),
            ("a3", "Paris, capital of France.", 0.85),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.6);
        assert!(result.consensus_reached);
        let winner = result.winning_response.unwrap();
        assert_eq!(winner.supporting_agents.len(), 3);
        assert!((winner.agreement_ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_majority_consensus() {
        let input = make_input(vec![
            ("a1", "The answer is 4.", 0.99),
            ("a2", "2 + 2 equals 4.", 0.95),
            ("a3", "The result is 5.", 0.60),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.3);
        assert!(result.consensus_reached);
        let winner = result.winning_response.unwrap();
        assert!(winner.supporting_agents.len() >= 2);
    }

    #[test]
    fn test_no_consensus() {
        let input = make_input(vec![
            ("a1", "Python is the best language.", 0.80),
            ("a2", "Rust is superior to all others.", 0.85),
            ("a3", "JavaScript dominates the web.", 0.75),
            ("a4", "Go is the most practical choice.", 0.70),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.8);
        assert!(!result.consensus_reached);
        assert!(result.winning_response.is_none());
    }

    #[test]
    fn test_two_responses_agree() {
        let input = make_input(vec![
            ("a1", "yes", 0.9),
            ("a2", "yes", 0.8),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.6);
        assert!(result.consensus_reached);
        let winner = result.winning_response.unwrap();
        assert_eq!(winner.supporting_agents.len(), 2);
    }

    #[test]
    fn test_clusters_output() {
        let input = make_input(vec![
            ("a1", "hello world", 0.9),
            ("a2", "hello world", 0.8),
            ("a3", "goodbye moon", 0.7),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.6);
        assert!(!result.clusters.is_empty());
    }

    #[test]
    fn test_metadata() {
        let input = make_input(vec![
            ("a1", "yes", 0.9),
            ("a2", "no", 0.8),
        ]);
        let strategy = QuorumStrategy;
        let result = strategy.evaluate(&input, 0.6);
        assert_eq!(result.metadata["total_responses"], 2);
        assert_eq!(result.metadata["quorum_size"], 2);
        assert_eq!(result.metadata["similarity_threshold"], 0.6);
    }
}
