use crate::models::AgentResponse;
use std::collections::HashMap;

/// Compute similarity ratio between two strings (case-insensitive, trimmed).
/// Uses longest common subsequence ratio similar to Python's SequenceMatcher.
pub fn text_similarity(a: &str, b: &str) -> f64 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    let a = a.trim();
    let b = b.trim();

    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let total = a_chars.len() + b_chars.len();

    let matching = count_matching_chars(&a_chars, &b_chars);
    (2.0 * matching as f64) / total as f64
}

/// Count matching characters using LCS (longest common subsequence).
fn count_matching_chars(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();
    // DP table for LCS length
    let mut dp = vec![vec![0u32; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n] as usize
}

/// Build pairwise similarity matrix for all response pairs.
/// Keys are (agent_id_1, agent_id_2) with agent_id_1 < agent_id_2.
pub fn build_similarity_matrix(
    responses: &[AgentResponse],
) -> HashMap<(String, String), f64> {
    let mut matrix = HashMap::new();
    for i in 0..responses.len() {
        for j in (i + 1)..responses.len() {
            let score = text_similarity(&responses[i].content, &responses[j].content);
            let key = if responses[i].agent_id < responses[j].agent_id {
                (responses[i].agent_id.clone(), responses[j].agent_id.clone())
            } else {
                (responses[j].agent_id.clone(), responses[i].agent_id.clone())
            };
            matrix.insert(key, score);
        }
    }
    matrix
}

/// Look up similarity between two agents, handling key order.
pub fn get_similarity(
    matrix: &HashMap<(String, String), f64>,
    id1: &str,
    id2: &str,
) -> f64 {
    if id1 == id2 {
        return 1.0;
    }
    let key = if id1 < id2 {
        (id1.to_string(), id2.to_string())
    } else {
        (id2.to_string(), id1.to_string())
    };
    matrix.get(&key).copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_strings() {
        assert_eq!(text_similarity("hello", "hello"), 1.0);
    }

    #[test]
    fn test_empty_strings() {
        assert_eq!(text_similarity("", ""), 1.0);
    }

    #[test]
    fn test_one_empty() {
        assert_eq!(text_similarity("hello", ""), 0.0);
        assert_eq!(text_similarity("", "hello"), 0.0);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(text_similarity("Hello", "hello"), 1.0);
    }

    #[test]
    fn test_whitespace_trimmed() {
        assert_eq!(text_similarity("  hello  ", "hello"), 1.0);
    }

    #[test]
    fn test_completely_different() {
        let score = text_similarity("abc", "xyz");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_partial_similarity() {
        let score = text_similarity("The capital of France is Paris.", "Paris is the capital of France.");
        assert!(score > 0.5, "Expected > 0.5, got {score}");
    }

    #[test]
    fn test_high_similarity() {
        let score = text_similarity("Yes, water is wet.", "Yes, water is indeed wet.");
        assert!(score > 0.7, "Expected > 0.7, got {score}");
    }

    #[test]
    fn test_build_similarity_matrix() {
        let responses = vec![
            AgentResponse {
                agent_id: "a1".to_string(),
                content: "hello world".to_string(),
                confidence: 1.0,
                metadata: Default::default(),
            },
            AgentResponse {
                agent_id: "a2".to_string(),
                content: "hello world".to_string(),
                confidence: 1.0,
                metadata: Default::default(),
            },
            AgentResponse {
                agent_id: "a3".to_string(),
                content: "goodbye".to_string(),
                confidence: 1.0,
                metadata: Default::default(),
            },
        ];
        let matrix = build_similarity_matrix(&responses);
        assert_eq!(matrix.len(), 3);
        assert_eq!(
            *matrix.get(&("a1".to_string(), "a2".to_string())).unwrap(),
            1.0
        );
        assert!(
            *matrix.get(&("a1".to_string(), "a3".to_string())).unwrap() < 0.5
        );
    }

    #[test]
    fn test_get_similarity_ordering() {
        let mut matrix = HashMap::new();
        matrix.insert(("a1".to_string(), "a2".to_string()), 0.8);

        assert_eq!(get_similarity(&matrix, "a1", "a2"), 0.8);
        assert_eq!(get_similarity(&matrix, "a2", "a1"), 0.8);
        assert_eq!(get_similarity(&matrix, "a1", "a1"), 1.0);
        assert_eq!(get_similarity(&matrix, "a1", "a3"), 0.0);
    }
}
