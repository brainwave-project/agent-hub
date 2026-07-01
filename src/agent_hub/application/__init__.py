"""Graph definitions + node functions.

  crag/       — §6 single-agent Corrective RAG  (imports nothing from ingestion/)
  ingestion/  — §7 multi-agent crew             (imports nothing from crag/)

The import boundary between these two packages is as strict as a service
boundary (TDD §5.4.3): if you ever split, that is exactly where the cut goes.
"""
