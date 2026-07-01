"""Unit tests for the deterministic performance-signal helpers."""

from lofi.agents.performance_analyst import compute_anomalies, compute_trend, confidence_label
from lofi.schemas.performance_analyst import ConfidenceLevel


class TestConfidenceLabel:
    def test_high_requires_both_sample_size_and_spend(self) -> None:
        assert confidence_label(30, 500.0) == ConfidenceLevel.HIGH

    def test_high_sample_size_with_low_spend_is_only_medium(self) -> None:
        assert confidence_label(30, 50.0) == ConfidenceLevel.MEDIUM

    def test_medium_on_sample_size_alone(self) -> None:
        assert confidence_label(14, 0.0) == ConfidenceLevel.MEDIUM

    def test_medium_on_spend_alone(self) -> None:
        assert confidence_label(1, 100.0) == ConfidenceLevel.MEDIUM

    def test_directional_when_both_low(self) -> None:
        assert confidence_label(5, 10.0) == ConfidenceLevel.DIRECTIONAL

    def test_no_spend_column_falls_back_to_sample_size_only(self) -> None:
        assert confidence_label(30) == ConfidenceLevel.HIGH
        assert confidence_label(14) == ConfidenceLevel.MEDIUM
        assert confidence_label(1) == ConfidenceLevel.DIRECTIONAL


def _daily_rows(values: list[float], metric: str = "cost", start: str = "2026-01-01") -> list[dict]:
    from datetime import date, timedelta

    base = date.fromisoformat(start)
    return [{"date": (base + timedelta(days=i)).isoformat(), metric: v} for i, v in enumerate(values)]


class TestComputeAnomalies:
    def test_flags_a_spike_beyond_two_std_dev(self) -> None:
        # 7 days with a little natural variance (std > 0), then one huge spike.
        rows = _daily_rows([95.0, 105.0, 98.0, 102.0, 97.0, 103.0, 100.0, 1000.0], metric="cost")

        anomalies = compute_anomalies(rows, metric_keys=("cost",))

        assert len(anomalies) == 1
        assert anomalies[0].metric == "cost"
        assert anomalies[0].direction == "spike"
        assert anomalies[0].date == rows[-1]["date"]

    def test_flags_a_drop(self) -> None:
        rows = _daily_rows([95.0, 105.0, 98.0, 102.0, 97.0, 103.0, 100.0, 1.0], metric="cost")

        anomalies = compute_anomalies(rows, metric_keys=("cost",))

        assert len(anomalies) == 1
        assert anomalies[0].direction == "drop"

    def test_no_anomaly_when_stable(self) -> None:
        rows = _daily_rows([100.0] * 10, metric="cost")

        assert compute_anomalies(rows, metric_keys=("cost",)) == []

    def test_returns_empty_without_enough_history(self) -> None:
        rows = _daily_rows([100.0] * 5, metric="cost")

        assert compute_anomalies(rows, metric_keys=("cost",)) == []

    def test_rows_without_date_are_excluded(self) -> None:
        rows = [{"cost": 100.0} for _ in range(10)]

        assert compute_anomalies(rows, metric_keys=("cost",)) == []


class TestComputeTrend:
    def test_computes_week_over_week_change(self) -> None:
        rows = _daily_rows([10.0] * 7 + [20.0] * 7, metric="roas")

        trend = compute_trend(rows, metric_keys=("roas",))

        assert len(trend) == 1
        assert trend[0].prior_period_avg == 10.0
        assert trend[0].last_period_avg == 20.0
        assert trend[0].change_pct == 100.0

    def test_returns_empty_with_fewer_than_two_periods(self) -> None:
        rows = _daily_rows([10.0] * 10, metric="roas")

        assert compute_trend(rows, metric_keys=("roas",)) == []

    def test_rows_without_date_are_excluded(self) -> None:
        rows = [{"roas": 10.0} for _ in range(20)]

        assert compute_trend(rows, metric_keys=("roas",)) == []

    def test_zero_prior_average_yields_null_change_pct(self) -> None:
        rows = _daily_rows([0.0] * 7 + [5.0] * 7, metric="roas")

        trend = compute_trend(rows, metric_keys=("roas",))

        assert trend[0].change_pct is None
