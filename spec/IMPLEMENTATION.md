# implementation spec

Multiple agents respond to prompts. The response document is JSON and contains a collection of responses in the same document. We want a system that looks at the responses from multiple agents and determines which are true. Build consensus amongst the responses using a practice like Prisoner's Dilemma or consensus like Raft/Paxos.

* Generate a plan and store in PLAN.md.
* Implement using Rust.
* Write unit tests and achieve at least 80% coverage.
* Update the README.md with a description of the project and details of how it works.
* Write integration tests for several use cases (single agent, 2 agents, 5 agents, 10 agents) to validate the behavior
