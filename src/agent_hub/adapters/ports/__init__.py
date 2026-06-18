"""Ports — the interfaces the application (graphs/nodes) depends on.

Defined as typing.Protocol so any adapter (real Ollama/gRPC, or a fake in a
test) satisfies them structurally. This is what makes the graph logic
unit-testable without Ollama or know-hub running (Principle 4: every decision
observable AND verifiable).
"""
