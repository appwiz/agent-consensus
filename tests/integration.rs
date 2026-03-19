use std::collections::HashMap;

use agent_consensus::engine::ConsensusEngine;
use agent_consensus::models::{AgentResponse, ConsensusInput, ConsensusResult};

fn load_fixture(name: &str) -> ConsensusInput {
    let path = format!("tests/fixtures/{name}");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {path}: {e}"));
    serde_json::from_str(&raw).unwrap()
}

fn make_responses(n: usize, content_fn: impl Fn(usize) -> (String, f64)) -> Vec<AgentResponse> {
    (0..n)
        .map(|i| {
            let (content, confidence) = content_fn(i);
            AgentResponse {
                agent_id: format!("agent-{}", i + 1),
                content,
                confidence,
                metadata: HashMap::new(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Single agent: should be rejected (need >= 2 for consensus)
// ---------------------------------------------------------------------------

#[test]
fn single_agent_quorum_rejected() {
    let input = load_fixture("single_agent.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let err = engine.run(&input).unwrap_err();
    assert!(
        err.contains("At least 2"),
        "Expected validation error for single agent, got: {err}"
    );
}

#[test]
fn single_agent_prisoners_dilemma_rejected() {
    let input = load_fixture("single_agent.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.6).unwrap();
    let err = engine.run(&input).unwrap_err();
    assert!(
        err.contains("At least 2"),
        "Expected validation error for single agent, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 2 agents
// ---------------------------------------------------------------------------

#[test]
fn two_agents_agree_quorum() {
    let input = load_fixture("two_agents_agree.json");
    let engine = ConsensusEngine::new("quorum", 0.3).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let winner = result.winning_response.as_ref().unwrap();
    assert_eq!(winner.supporting_agents.len(), 2);
    assert_eq!(winner.agreement_ratio, 1.0);
    assert_eq!(result.metadata["total_responses"], 2);
    assert_eq!(result.metadata["quorum_size"], 2);
}

#[test]
fn two_agents_agree_prisoners_dilemma() {
    let input = load_fixture("two_agents_agree.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.3).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let winner = result.winning_response.as_ref().unwrap();
    assert_eq!(winner.supporting_agents.len(), 2);
}

#[test]
fn two_agents_disagree_quorum() {
    let input = load_fixture("two_agents_disagree.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // Two very different answers — no quorum (need both to agree)
    assert!(!result.consensus_reached);
    assert!(result.winning_response.is_none());
}

#[test]
fn two_agents_disagree_prisoners_dilemma() {
    let input = load_fixture("two_agents_disagree.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // Two dissimilar responses — both should be defectors, no consensus
    assert!(!result.consensus_reached);
}

// ---------------------------------------------------------------------------
// 5 agents
// ---------------------------------------------------------------------------

#[test]
fn five_agents_majority_quorum() {
    let input = load_fixture("five_agents_majority.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // 3 of 5 agree ("water is wet"), quorum = 3
    assert!(result.consensus_reached);
    let winner = result.winning_response.as_ref().unwrap();
    assert!(
        winner.supporting_agents.len() >= 3,
        "Expected at least 3 supporting agents, got {}",
        winner.supporting_agents.len()
    );
    assert!(winner.content.to_lowercase().contains("wet"));
    assert_eq!(result.metadata["quorum_size"], 3);
}

#[test]
fn five_agents_majority_prisoners_dilemma() {
    let input = load_fixture("five_agents_majority.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.5).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let cooperators = result.metadata["cooperators"].as_array().unwrap();
    assert!(
        cooperators.len() >= 2,
        "Expected at least 2 cooperators, got {}",
        cooperators.len()
    );
}

#[test]
fn five_agents_no_consensus_quorum() {
    let input = load_fixture("five_agents_no_consensus.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // All different answers — no cluster reaches quorum of 3
    assert!(!result.consensus_reached);
    assert!(result.winning_response.is_none());
    assert!(
        result.clusters.len() >= 3,
        "Expected multiple clusters for divergent answers"
    );
}

#[test]
fn five_agents_no_consensus_prisoners_dilemma() {
    let input = load_fixture("five_agents_no_consensus.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // All different — should produce defectors
    let defectors = result.metadata["defectors"].as_array().unwrap();
    assert!(
        !defectors.is_empty(),
        "Expected defectors with fully divergent answers"
    );
}

// ---------------------------------------------------------------------------
// 10 agents
// ---------------------------------------------------------------------------

#[test]
fn ten_agents_strong_consensus_quorum() {
    let input = load_fixture("ten_agents_strong_consensus.json");
    let engine = ConsensusEngine::new("quorum", 0.5).unwrap();
    let result = engine.run(&input).unwrap();

    // 8 of 10 agree on "100 degrees Celsius", quorum = 6
    assert!(result.consensus_reached);
    let winner = result.winning_response.as_ref().unwrap();
    assert!(
        winner.supporting_agents.len() >= 6,
        "Expected at least 6 supporting agents, got {}",
        winner.supporting_agents.len()
    );
    assert!(
        winner.confidence_score > 0.8,
        "Expected high confidence, got {}",
        winner.confidence_score
    );
    assert_eq!(result.metadata["total_responses"], 10);
    assert_eq!(result.metadata["quorum_size"], 6);
}

#[test]
fn ten_agents_strong_consensus_prisoners_dilemma() {
    let input = load_fixture("ten_agents_strong_consensus.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.5).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let cooperators = result.metadata["cooperators"].as_array().unwrap();
    assert!(
        cooperators.len() >= 6,
        "Expected at least 6 cooperators, got {}",
        cooperators.len()
    );
}

#[test]
fn ten_agents_split_quorum() {
    let input = load_fixture("ten_agents_split.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // 5 vs 5 split — neither side reaches quorum of 6
    assert!(!result.consensus_reached);
    assert!(result.winning_response.is_none());
    assert!(
        result.clusters.len() >= 2,
        "Expected at least 2 clusters for split opinions"
    );
}

#[test]
fn ten_agents_split_prisoners_dilemma() {
    let input = load_fixture("ten_agents_split.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    // With a 5/5 split, the strategy should still identify groups
    assert!(result.metadata["total_responses"] == 10);
    let cooperators = result.metadata["cooperators"].as_array().unwrap();
    let defectors = result.metadata["defectors"].as_array().unwrap();
    assert_eq!(cooperators.len() + defectors.len(), 10);
}

// ---------------------------------------------------------------------------
// Programmatic scale tests (generated inputs)
// ---------------------------------------------------------------------------

#[test]
fn two_agents_programmatic_identical() {
    let responses = make_responses(2, |_i| ("The answer is 42.".to_string(), 0.95));
    let input = ConsensusInput {
        prompt: "What is the meaning of life?".to_string(),
        responses,
    };
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let winner = result.winning_response.unwrap();
    assert_eq!(winner.agreement_ratio, 1.0);
}

#[test]
fn five_agents_programmatic_supermajority() {
    let responses = make_responses(5, |i| {
        if i < 4 {
            (format!("Earth orbits the Sun. Agent {i} agrees."), 0.9)
        } else {
            ("The Sun orbits the Earth.".to_string(), 0.5)
        }
    });
    let input = ConsensusInput {
        prompt: "What orbits what?".to_string(),
        responses,
    };
    let engine = ConsensusEngine::new("quorum", 0.5).unwrap();
    let result = engine.run(&input).unwrap();

    assert!(result.consensus_reached);
    let winner = result.winning_response.unwrap();
    assert!(winner.supporting_agents.len() >= 3);
}

#[test]
fn ten_agents_programmatic_all_agree() {
    let responses = make_responses(10, |i| {
        (
            format!("Water is H2O, agent {i} confirms."),
            0.95 - (i as f64 * 0.01),
        )
    });
    let input = ConsensusInput {
        prompt: "What is water?".to_string(),
        responses,
    };

    // Quorum
    let engine = ConsensusEngine::new("quorum", 0.5).unwrap();
    let result = engine.run(&input).unwrap();
    assert!(result.consensus_reached);
    let winner = result.winning_response.unwrap();
    assert_eq!(winner.supporting_agents.len(), 10);
    assert_eq!(winner.agreement_ratio, 1.0);
}

#[test]
fn ten_agents_programmatic_all_agree_prisoners_dilemma() {
    let responses = make_responses(10, |i| {
        (
            format!("Water is H2O, agent {i} confirms."),
            0.95 - (i as f64 * 0.01),
        )
    });
    let input = ConsensusInput {
        prompt: "What is water?".to_string(),
        responses,
    };

    let engine = ConsensusEngine::new("prisoners_dilemma", 0.5).unwrap();
    let result = engine.run(&input).unwrap();
    assert!(result.consensus_reached);
    let cooperators = result.metadata["cooperators"].as_array().unwrap();
    assert_eq!(cooperators.len(), 10);
}

#[test]
fn ten_agents_programmatic_all_different() {
    let responses = make_responses(10, |i| {
        let unique_answers = [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo",
            "Foxtrot", "Golf", "Hotel", "India", "Juliet",
        ];
        (unique_answers[i].to_string(), 0.8)
    });
    let input = ConsensusInput {
        prompt: "Name a word.".to_string(),
        responses,
    };

    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();
    assert!(!result.consensus_reached);
    assert!(result.winning_response.is_none());
}

// ---------------------------------------------------------------------------
// Output format validation
// ---------------------------------------------------------------------------

fn validate_result_shape(result: &ConsensusResult) {
    // strategy is always present
    assert!(!result.strategy.is_empty());
    // prompt is preserved
    assert!(!result.prompt.is_empty());
    // clusters are always present
    assert!(!result.clusters.is_empty());
    // if consensus reached, winning_response must exist and vice versa
    assert_eq!(result.consensus_reached, result.winning_response.is_some());

    if let Some(winner) = &result.winning_response {
        assert!(!winner.content.is_empty());
        assert!(!winner.supporting_agents.is_empty());
        assert!(winner.agreement_ratio > 0.0 && winner.agreement_ratio <= 1.0);
        assert!(winner.confidence_score >= 0.0 && winner.confidence_score <= 1.0);
    }

    for cluster in &result.clusters {
        assert!(!cluster.members.is_empty());
        assert!(!cluster.representative_content.is_empty());
    }
}

#[test]
fn output_format_quorum_consensus() {
    let input = load_fixture("five_agents_majority.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();
    validate_result_shape(&result);

    // Verify it serializes to valid JSON
    let json = serde_json::to_string(&result).unwrap();
    let _: serde_json::Value = serde_json::from_str(&json).unwrap();
}

#[test]
fn output_format_prisoners_dilemma_consensus() {
    let input = load_fixture("five_agents_majority.json");
    let engine = ConsensusEngine::new("prisoners_dilemma", 0.5).unwrap();
    let result = engine.run(&input).unwrap();
    validate_result_shape(&result);

    let json = serde_json::to_string(&result).unwrap();
    let _: serde_json::Value = serde_json::from_str(&json).unwrap();
}

#[test]
fn output_format_no_consensus() {
    let input = load_fixture("five_agents_no_consensus.json");
    let engine = ConsensusEngine::new("quorum", 0.6).unwrap();
    let result = engine.run(&input).unwrap();
    validate_result_shape(&result);
}

// ---------------------------------------------------------------------------
// CLI integration (binary execution)
// ---------------------------------------------------------------------------

#[test]
fn cli_file_input_quorum() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["-i", "examples/sample_input.json", "-s", "quorum", "--pretty"])
        .output()
        .expect("Failed to run binary");

    assert!(output.status.success(), "CLI exited with error");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed["strategy"], "quorum");
    assert_eq!(parsed["consensus_reached"], true);
}

#[test]
fn cli_file_input_prisoners_dilemma() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["-i", "examples/sample_input.json", "-s", "prisoners_dilemma"])
        .output()
        .expect("Failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed["strategy"], "prisoners_dilemma");
}

#[test]
fn cli_list_strategies() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["--list-strategies"])
        .output()
        .expect("Failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let strategies: Vec<String> = serde_json::from_str(&stdout).unwrap();
    assert!(strategies.contains(&"quorum".to_string()));
    assert!(strategies.contains(&"prisoners_dilemma".to_string()));
}

#[test]
fn cli_no_consensus_exit_code() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["-i", "tests/fixtures/five_agents_no_consensus.json", "-s", "quorum"])
        .output()
        .expect("Failed to run binary");

    // Exit code 2 = no consensus
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn cli_invalid_file_exit_code() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["-i", "nonexistent.json"])
        .output()
        .expect("Failed to run binary");

    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn cli_stdin_input() {
    use std::io::Write;

    let fixture = std::fs::read_to_string("examples/sample_input.json").unwrap();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_agent-consensus"))
        .args(["-s", "quorum", "--pretty"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn binary");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(fixture.as_bytes())
        .unwrap();
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "CLI failed with code {:?}, stderr: {stderr}",
        output.status.code()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed["consensus_reached"], true);
}
