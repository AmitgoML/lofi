# Data pipeline

## Layers

| Layer | Path | Store |
|-------|------|-------|
| L1 Raw-immutable | `layers/l1_raw/` | S3 |
| L2 Source-aligned | `layers/l2_source_aligned/` | Supabase (`google_ads.*`, `meta_ads.*`, …) |
| L3 Canonical | `layers/l3_canonical/` | `campaign_metrics`, `audience_metrics`, mapping tables |
| L4 Serving | `layers/l4_serving/` | Supabase views for agents + dashboards |

## Platform build order

1. Google Ads → 2. Meta → 3. Spotify → 4. TikTok (§9.1)

## Supporting modules

- `connectors/` — unified auth + per-platform read connectors (P4-01)
- `mapping/` — campaign + creative mapping tables (§4.2, §4.3)
- `normalization/` — currency, timezone, geo resolution, KPI derivation (§7.2–7.3)
- `scheduling/` — hourly totals, daily breakdowns, rolling restatement (§10)
