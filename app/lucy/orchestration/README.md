# Orchestration

Workflow 2 (Brief-to-Campaign Launch) execution engine.

## Submodules

| Path | Responsibility | Tasks |
|------|----------------|-------|
| `engine/` | Sequential + parallel task graph executor | P1-04, P1-05 |
| `state/` | Supabase workflow_runs, tasks, checkpoints persistence | P1-02, P1-06 |
| `hitl/` | Pause/resume, approve/reject/revise, revision router | P2-* |
| `streaming/` | NDJSON workflow event emission | P3-01, P3-02 |
| `workflows/` | Per-workflow definitions (campaign_planner first) | P1-01, P7-02 |

## Integration points

- **Router** (`lucy.agents.router_agent`) — extended to trigger workflow runs (P1-08)
- **Agents** — adapted via standardized handoff pattern (P7-01)
- **Domain contracts** — `lucy.domain.workflows`
