"""M8 — agent evaluation (§12.3). Two distinct questions, two evaluations:

  Outcome    — was the final answer correct & grounded?  (RAGAS faithfulness)
  Trajectory — did it take a sensible path?               (loop count, retries)

Trajectory metrics live in adapters.runs_repo.metrics() (queried from the agent
schema, exposed at GET /api/agent/metrics). Outcome eval is the offline RAGAS
harness here — knowing the difference is the senior→principal divide.
"""
