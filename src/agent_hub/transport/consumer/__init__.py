"""Ingestion entry point — the Kafka consumer that drives the crew (§5.4.3).
Separate code path from the HTTP/CRAG surface; they never share a thread."""
