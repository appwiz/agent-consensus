# agent-consensus

A Rust CLI tool that builds consensus from multiple AI agent responses. Given a JSON document containing responses from several agents to the same prompt, it determines which responses are most likely true using consensus algorithms inspired by Raft/Paxos quorum voting and the Prisoner's Dilemma.

## How It Works

### Input

A JSON document with a prompt and an array of agent responses:

```json
{
  "prompt": "What is the capital of France?",
  "responses": [
    {"agent_id": "agent-1", "content": "Paris is the capital.", "confidence": 0.95},
    {"agent_id": "agent-2", "content": "The capital is Paris.", "confidence": 0.90},
    {"agent_id": "agent-3", "content": "The capital is Lyon.", "confidence": 0.70}
  ]
}
```

### Consensus Strategies

**Quorum (Raft/Paxos-inspired)**

1. Builds a pairwise text similarity matrix using longest common subsequence ratio
2. Clusters responses via single-linkage clustering (default threshold: 0.6)
3. Requires strict majority quorum: `floor(N/2) + 1` agreeing agents
4. Winner = largest cluster meeting quorum; representative = highest confidence member
5. If no cluster meets quorum, consensus fails

**Prisoner's Dilemma**

1. Classifies agents as cooperators (high weighted avg similarity to others) or defectors
2. Assigns payoffs: R=3 (cooperate/cooperate), T=5 (defect/cooperate), S=0 (cooperate/defect), P=1 (defect/defect)
3. Runs 3 iterated rounds; defectors lose 20% effective confidence each round
4. Winner = cooperator with highest cumulative payoff; consensus requires >= 2 cooperators

### Output

```json
{
  "strategy": "quorum",
  "prompt": "What is the capital of France?",
  "consensus_reached": true,
  "winning_response": {
    "content": "Paris is the capital.",
    "supporting_agents": ["agent-1", "agent-2"],
    "agreement_ratio": 0.667,
    "confidence_score": 0.925
  },
  "clusters": [...],
  "metadata": {"total_responses": 3, "similarity_threshold": 0.6, "quorum_size": 2}
}
```

## Usage

```bash
# Build
cargo build --release

# Run with a file
cargo run -- -i examples/sample_input.json -s quorum --pretty

# Run with stdin
cat examples/sample_input.json | cargo run -- --pretty

# Use prisoners_dilemma strategy
cargo run -- -i examples/sample_input.json -s prisoners_dilemma --pretty

# List available strategies
cargo run -- --list-strategies

# Custom similarity threshold
cargo run -- -i examples/sample_input.json -t 0.8
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input <FILE>` | Input JSON file path | stdin |
| `-s, --strategy <NAME>` | Consensus strategy | `quorum` |
| `-t, --threshold <FLOAT>` | Similarity threshold (0.0-1.0) | `0.6` |
| `-p, --pretty` | Pretty-print JSON output | off |
| `--list-strategies` | List strategies and exit | |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success, consensus reached |
| 1 | Error (bad input, unknown strategy) |
| 2 | Success, but no consensus reached |

## Testing

```bash
# Run all tests (37 unit + 28 integration)
cargo test

# Run with coverage (requires cargo-tarpaulin)
cargo tarpaulin --skip-clean
```

Integration tests cover the spec-required use cases:

| Scenario | Agents | Tests |
|----------|--------|-------|
| Single agent | 1 | Rejected by both strategies (need >= 2) |
| Two agents agree | 2 | Consensus reached via quorum and PD |
| Two agents disagree | 2 | No consensus for either strategy |
| Five agents majority | 5 | 3/5 agree, quorum met |
| Five agents all different | 5 | No cluster reaches quorum |
| Ten agents strong consensus | 10 | 8/10 agree, clear winner |
| Ten agents split | 10 | 5/5 split, no quorum |

Plus CLI tests for file input, stdin piping, `--list-strategies`, and exit codes (0, 1, 2).

## Project Structure

```
src/
├── lib.rs                          # Library root
├── main.rs                         # CLI binary entry point
├── models.rs                       # Data types: AgentResponse, ConsensusInput, ConsensusResult
├── similarity.rs                   # LCS-based text similarity and pairwise matrix
├── engine.rs                       # Orchestrates strategy execution with validation
└── strategies/
    ├── mod.rs                      # Strategy trait and registry
    ├── quorum.rs                   # Raft/Paxos quorum strategy
    └── prisoners_dilemma.rs        # Prisoner's Dilemma strategy
tests/
├── integration.rs                  # Integration tests
└── fixtures/                       # JSON fixtures for integration tests
examples/
└── sample_input.json               # Example input file
```
