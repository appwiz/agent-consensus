use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentResponse {
    pub agent_id: String,
    pub content: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_confidence() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusInput {
    pub prompt: String,
    pub responses: Vec<AgentResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCluster {
    pub representative_content: String,
    pub members: Vec<String>,
    pub similarity_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub strategy: String,
    pub prompt: String,
    pub consensus_reached: bool,
    pub winning_response: Option<WinningResponse>,
    pub clusters: Vec<ResponseCluster>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinningResponse {
    pub content: String,
    pub supporting_agents: Vec<String>,
    pub agreement_ratio: f64,
    pub confidence_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_response_deserialize_defaults() {
        let json = r#"{"agent_id": "a1", "content": "hello"}"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.agent_id, "a1");
        assert_eq!(resp.content, "hello");
        assert_eq!(resp.confidence, 1.0);
        assert!(resp.metadata.is_empty());
    }

    #[test]
    fn test_agent_response_deserialize_full() {
        let json = r#"{"agent_id": "a1", "content": "hello", "confidence": 0.9, "metadata": {"key": "val"}}"#;
        let resp: AgentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.confidence, 0.9);
        assert_eq!(resp.metadata.get("key").unwrap(), "val");
    }

    #[test]
    fn test_consensus_input_roundtrip() {
        let input = ConsensusInput {
            prompt: "test?".to_string(),
            responses: vec![
                AgentResponse {
                    agent_id: "a1".to_string(),
                    content: "yes".to_string(),
                    confidence: 0.9,
                    metadata: HashMap::new(),
                },
                AgentResponse {
                    agent_id: "a2".to_string(),
                    content: "no".to_string(),
                    confidence: 0.8,
                    metadata: HashMap::new(),
                },
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: ConsensusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.prompt, "test?");
        assert_eq!(deserialized.responses.len(), 2);
    }

    #[test]
    fn test_consensus_result_serialize() {
        let result = ConsensusResult {
            strategy: "quorum".to_string(),
            prompt: "test?".to_string(),
            consensus_reached: true,
            winning_response: Some(WinningResponse {
                content: "yes".to_string(),
                supporting_agents: vec!["a1".to_string()],
                agreement_ratio: 1.0,
                confidence_score: 0.9,
            }),
            clusters: vec![],
            metadata: serde_json::json!({"total_responses": 1}),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"consensus_reached\":true"));
    }

    #[test]
    fn test_response_cluster_serialize() {
        let cluster = ResponseCluster {
            representative_content: "Paris".to_string(),
            members: vec!["a1".to_string(), "a2".to_string()],
            similarity_scores: HashMap::from([("a1:a2".to_string(), 0.85)]),
        };
        let json = serde_json::to_string(&cluster).unwrap();
        assert!(json.contains("\"representative_content\":\"Paris\""));
    }
}
