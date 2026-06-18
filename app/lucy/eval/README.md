# Evaluation framework

LLM-as-judge scoring, regression testing, CI-compatible eval runs (P8).

Requires golden I/O examples from P0-06 before judges can be calibrated.

## Submodules

| Path | Responsibility |
|------|----------------|
| `datasets/` | Golden + synthetic eval scenarios (P8-01) |
| `judges/` | Calibrated rubrics per agent (P8-02) |
| `runner/` | Reproducible test runner + CI mode (P8-03, P8-04) |
