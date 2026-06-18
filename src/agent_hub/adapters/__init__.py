"""Adapters — implementations of the ports the application layer depends on.

Shared by BOTH domains (the correct dependency direction): know-hub client,
Ollama factory, Postgres checkpointer, Kafka. Populated from M2 onward.
"""
