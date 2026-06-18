# Competitive intelligence

SearchAPI.io competitor ad ingestion, deduplication, and agent query layer.

## Submodules

| Path | Responsibility |
|------|----------------|
| `connectors/` | SearchAPI.io ingestion (P5-02) |
| `storage/` | Competitor ad records schema access (P5-01) |
| `queries/` | SQL tool interface for Market Analyst (P5-04) |

Migrations for competitor tables live in `lofi-v2-dev/supabase/migrations/`.
