# Agent Consensus System - Implementation Plan

## Context

The spec requires a system that takes a JSON document containing multiple agent responses to a prompt and determines which responses are "true" via consensus algorithms inspired by Prisoner's Dilemma and Raft/Paxos. Implemented in Rust with serde for JSON and clap for CLI.

## Project Structure

```
src/
├── lib.rs                          # Library root
├── main.rs                         # CLI binary (clap)
├── models.rs                       # Serde types: AgentResponse, ConsensusInput, ConsensusResult
├── similarity.rs                   # LCS-based text similarity and pairwise matrix
├── engine.rs                       # Orchestrates strategy execution with validation
└── strategies/
    ├── mod.rs                      # ConsensusStrategy trait and registry
    ├── quorum.rs                   # Raft/Paxos quorum strategy
    └── prisoners_dilemma.rs        # Prisoner's Dilemma strategy
tests/
├── integration.rs                  # Integration tests (1, 2, 5, 10 agents + CLI)
└── fixtures/                       # JSON fixtures for integration tests
    ├── single_agent.json
    ├── two_agents_agree.json
    ├── two_agents_disagree.json
    ├── five_agents_majority.json
    ├── five_agents_no_consensus.json
    ├── ten_agents_strong_consensus.json
    └── ten_agents_split.json
examples/
└── sample_input.json
Cargo.toml
```

## Input/Output Format

**Input:**
```json
{
  "prompt": "What is the capital of France?",
  "responses": [
    {"agent_id": "agent-1", "content": "Paris", "confidence": 0.95, "metadata": {}},
    {"agent_id": "agent-2", "content": "Paris", "confidence": 0.90, "metadata": {}}
  ]
}
```

**Output:**
```json
{
  "strategy": "quorum",
  "prompt": "...",
  "consensus_reached": true,
  "winning_response": {
    "content": "...",
    "supporting_agents": ["agent-1", "agent-2"],
    "agreement_ratio": 0.667,
    "confidence_score": 0.85
  },
  "clusters": [...],
  "metadata": {"total_responses": 3, "similarity_threshold": 0.6, "quorum_size": 2}
}
```

## Consensus Strategies

### Quorum (Raft/Paxos-inspired)
1. Build pairwise similarity matrix using LCS ratio
2. Cluster responses via single-linkage clustering (threshold default 0.6)
3. Require strict majority quorum: `floor(N/2) + 1`
4. Winner = largest cluster meeting quorum; representative = highest confidence member
5. If no cluster meets quorum, `consensus_reached = false`

### Prisoner's Dilemma
1. Classify agents as cooperators (confidence-weighted avg similarity above threshold) or defectors
2. Assign payoffs: R=3 (cooperate/cooperate), T=5 (defect/cooperate), S=0 (cooperate/defect), P=1 (defect/defect)
3. Run 3 iterated rounds, penalizing defectors each round (reduce effective confidence by 20%)
4. Winner = highest cumulative payoff cooperator; consensus requires >= 2 cooperators

## CLI

```
cargo run -- -i input.json -s quorum -t 0.6 --pretty
cat input.json | cargo run --
cargo run -- --list-strategies
```

Exit codes: 0 = success, 1 = error, 2 = no consensus reached.

## Integration Tests

Tests cover the spec-required use cases:
- **Single agent**: Rejected with validation error (need >= 2 for consensus)
- **2 agents**: Agreement (consensus) and disagreement (no consensus) for both strategies
- **5 agents**: Majority consensus, full disagreement, supermajority programmatic
- **10 agents**: Strong consensus (8/10), even split (5/5), all-agree, all-different

Plus CLI integration tests: file input, stdin pipe, --list-strategies, exit codes.

## Verification

```bash
cargo test                    # 65 tests (37 unit + 28 integration)
cargo tarpaulin --skip-clean  # 95.39% coverage
cargo run -- -i examples/sample_input.json -s quorum --pretty
cargo run -- -i examples/sample_input.json -s prisoners_dilemma --pretty
```
