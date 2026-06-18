"""Pipeline health and operator endpoints (P4-07, P4-08).

Not mounted until implementation; scaffold only.
"""

from fastapi import APIRouter

router = APIRouter()

# Endpoints to implement:
# - GET /pipeline/health
# - GET /pipeline/freshness
# - POST /pipeline/backfill  (operator-triggered)
