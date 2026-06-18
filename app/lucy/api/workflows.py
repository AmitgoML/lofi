"""Workflow API endpoints — status, checkpoints, launch (P2-04, P3-03).

Not mounted until implementation; scaffold only.
"""

from fastapi import APIRouter

router = APIRouter()

# Endpoints to implement:
# - POST /workflows/{workflow_id}/runs
# - GET  /workflows/runs/{run_id}
# - POST /workflows/runs/{run_id}/checkpoints/{checkpoint_id}/decide
# - GET  /workflows/runs/{run_id}/events  (NDJSON stream)
