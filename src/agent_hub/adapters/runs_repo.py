"""agent.agent_runs / agent.agent_steps persistence (§9.2–9.3).

A durable trajectory log: one run row per CRAG/ingestion invocation, one step
row per node entry/decision. This is audit + evaluation data, NEVER a system of
record. Step summaries are REDACTED — node names, decisions, latencies, counts —
never full document bodies (§14 trace redaction).

A single asyncpg-style psycopg pool is shared process-wide. `step_recorder(run_id)`
returns a StepRecorder bound to a run, auto-incrementing `seq`.
"""

from __future__ import annotations

from psycopg_pool import AsyncConnectionPool

from agent_hub.observability.logging import get_logger

log = get_logger("runs-repo")

_MAX_SUMMARY = 500  # hard cap so a runaway summary never bloats the trace store


def _redact(text: str) -> str:
    return (text or "")[:_MAX_SUMMARY]


class RunsRepository:
    def __init__(self, database_url: str) -> None:
        # search_path=agent so unqualified table names resolve to the agent schema.
        sep = "&" if "?" in database_url else "?"
        self._conninfo = f"{database_url}{sep}options=-csearch_path%3Dagent"
        self._pool: AsyncConnectionPool | None = None

    async def start(self) -> None:
        if self._pool is None:
            self._pool = AsyncConnectionPool(self._conninfo, min_size=1, max_size=5, open=False)
            await self._pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("RunsRepository.start() not called")
        return self._pool

    async def create_run(
        self, kind: str, thread_id: str, actor_user_id: str, correlation_id: str = ""
    ) -> str:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    INSERT INTO agent_runs (kind, thread_id, actor_user_id, correlation_id)
                    VALUES (%s, %s, %s, NULLIF(%s, ''))
                    RETURNING id::text
                    """,
                    (kind, thread_id, actor_user_id, correlation_id),
                )
            ).fetchone()
            return row[0]

    async def finish_run(self, run_id: str, status: str, loops: int) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                UPDATE agent_runs
                   SET status = %s::agent.run_status, loops = %s, ended_at = now()
                 WHERE id = %s::uuid
                """,
                (status, loops, run_id),
            )

    async def add_step(
        self,
        run_id: str,
        seq: int,
        node: str,
        decision: str,
        input_summary: str,
        output_summary: str,
        latency_ms: int,
    ) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO agent_steps
                    (run_id, seq, node, decision, input_summary, output_summary, latency_ms)
                VALUES (%s::uuid, %s, %s, NULLIF(%s,''), %s, %s, %s)
                """,
                (run_id, seq, node, decision, _redact(input_summary), _redact(output_summary), latency_ms),
            )

    async def get_run(self, run_id: str) -> dict | None:
        """Run header + ordered steps — powers GET /api/agent/runs/:id."""
        async with self.pool.connection() as conn:
            run = await (
                await conn.execute(
                    """
                    SELECT id::text, kind::text, thread_id, actor_user_id, status::text,
                           loops, started_at, ended_at, correlation_id
                      FROM agent_runs WHERE id = %s::uuid
                    """,
                    (run_id,),
                )
            ).fetchone()
            if run is None:
                return None
            steps = await (
                await conn.execute(
                    """
                    SELECT seq, node, decision, input_summary, output_summary, latency_ms
                      FROM agent_steps WHERE run_id = %s::uuid ORDER BY seq
                    """,
                    (run_id,),
                )
            ).fetchall()
        cols = ["id", "kind", "thread_id", "actor_user_id", "status", "loops",
                "started_at", "ended_at", "correlation_id"]
        result = dict(zip(cols, run, strict=True))
        result["started_at"] = result["started_at"].isoformat() if result["started_at"] else None
        result["ended_at"] = result["ended_at"].isoformat() if result["ended_at"] else None
        result["steps"] = [
            {"seq": s[0], "node": s[1], "decision": s[2],
             "input_summary": s[3], "output_summary": s[4], "latency_ms": s[5]}
            for s in steps
        ]
        return result

    async def metrics(self) -> dict:
        """Trajectory metrics for CRAG runs (§12.4) — the senior→principal divide.

        correction_rate   — % of runs that needed ≥1 query rewrite (retrieval/query quality)
        abstention_rate   — % of runs that hit MAX_LOOPS and abstained honestly
        loop_depth        — distribution of corrective iterations (fat tail = tuning problem)
        per_node_latency  — mean latency_ms per node (which node dominates wall-clock)
        """
        async with self.pool.connection() as conn:
            agg = await (
                await conn.execute(
                    """
                    SELECT
                      count(*)                                              AS total,
                      count(*) FILTER (WHERE loops > 0)                     AS corrected,
                      count(*) FILTER (WHERE status = 'ABSTAINED')          AS abstained,
                      coalesce(avg(loops), 0)                               AS avg_loops
                    FROM agent_runs WHERE kind = 'CRAG'
                    """
                )
            ).fetchone()
            depth = await (
                await conn.execute(
                    "SELECT loops, count(*) FROM agent_runs WHERE kind='CRAG' GROUP BY loops ORDER BY loops"
                )
            ).fetchall()
            node_lat = await (
                await conn.execute(
                    """
                    SELECT node, round(avg(latency_ms)) AS mean_ms, count(*) AS n
                      FROM agent_steps GROUP BY node ORDER BY mean_ms DESC NULLS LAST
                    """
                )
            ).fetchall()
        total = agg[0] or 0
        return {
            "total_crag_runs": total,
            "correction_rate": round((agg[1] or 0) / total, 3) if total else 0.0,
            "abstention_rate": round((agg[2] or 0) / total, 3) if total else 0.0,
            "avg_loops": round(float(agg[3] or 0), 3),
            "loop_depth_distribution": {str(d[0]): d[1] for d in depth},
            "per_node_latency_ms": {n[0]: {"mean": n[1], "count": n[2]} for n in node_lat},
        }

    def step_recorder(self, run_id: str) -> RunStepRecorder:
        return RunStepRecorder(self, run_id)


class RunStepRecorder:
    """StepRecorder bound to one run; assigns monotonically increasing seq."""

    def __init__(self, repo: RunsRepository, run_id: str) -> None:
        self._repo = repo
        self._run_id = run_id
        self._seq = 0

    async def record(
        self,
        node: str,
        decision: str = "",
        input_summary: str = "",
        output_summary: str = "",
        latency_ms: int = 0,
    ) -> None:
        self._seq += 1
        try:
            await self._repo.add_step(
                self._run_id, self._seq, node, decision, input_summary, output_summary, latency_ms
            )
        except Exception as exc:  # observability must never break the run
            log.warning("step.record_failed", node=node, error=str(exc))


class NullStepRecorder:
    """Used pre-M4 (and in tests) so graphs run identically without a DB."""

    async def record(self, *args, **kwargs) -> None:
        return None
