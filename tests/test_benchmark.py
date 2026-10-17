"""Replay accounting and pressure simulation, without claiming IBM measurements."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pandas as pd
import pytest

from scalper_detector.benchmark import benchmark_orders
from scalper_detector.data import PaymentFeatureStore


class RecordingSession:
    """Exercise real causal state while isolating replay accounting from a model."""

    def __init__(self):
        self.feature_store = PaymentFeatureStore()
        self.orders = []
        self.last_neural_ms = 0.01

    def score_order(self, order):
        enriched = self.feature_store.update(order)
        self.orders.append(enriched)
        return {"order_id": enriched["order_id"], "scalper_probability": 0.5}


class RecordingDetector:
    def __init__(self):
        self.sessions = []

    def new_session(self):
        session = RecordingSession()
        self.sessions.append(session)
        return session


@pytest.fixture
def orders():
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame([
        {"order_id": f"order-{index}", "timestamp": (origin + timedelta(seconds=index * 120)).isoformat(),
         "card_id": "tokenized-card", "card_bin": "001234", "merchant_id": f"site-{index % 3}",
         "category": "tickets", "amount": 100.0, "is_scalper": index % 2}
        for index in range(6)
    ])


def test_replay_counts_warmup_and_labels_cpu_backend_honestly(orders):
    detector = RecordingDetector()
    report = benchmark_orders(detector, orders.iloc[::-1], max_orders=3, warmup=2)
    assert report["backend"] == "pytorch_cpu_plus_numpy_forest"
    assert report["telum_measured"] is False
    assert "not measured" in report["hardware_comparison"]["telum"]
    assert "concurrent load" in report["exclusions"]
    assert report["platform"] and report["hardware"]
    assert len(detector.sessions) == 2
    for key in ("normal", "simulated_drop_day_burst"):
        replay = report[key]
        assert replay["orders_measured"] == 3
        assert replay["warmup_orders"] == 2
        for statistic, quantiles in (("end_to_end_ms", ("p50", "p95", "p99")),
                                    ("neural_only_ms", ("p50", "p95"))):
            values = [replay[statistic][quantile] for quantile in quantiles]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            assert values == sorted(values)
        assert replay["serial_replay_orders_per_second"] > 0
        assert math.isfinite(replay["serial_replay_orders_per_second"])
        assert replay["simulated_serial_queue_wait_p95_ms"] >= 0
    for session in detector.sessions:
        assert [entry["order_id"] for entry in session.orders] == [f"order-{i}" for i in range(5)]
        assert all("is_scalper" not in entry for entry in session.orders)


def test_sparse_trace_burst_reaches_requested_offered_rate_and_rebuilds_causal_state(orders):
    detector = RecordingDetector()
    report = benchmark_orders(detector, orders, max_orders=3, warmup=2,
                              burst_multiplier=10, burst_orders_per_second=1500)
    normal = report["normal"]
    burst = report["simulated_drop_day_burst"]
    assert report["burst_requested_min_orders_per_second"] == 1500
    assert normal["arrival_time_compression"] == 1
    assert normal["trace_arrival_orders_per_second"] == pytest.approx(1 / 120)
    assert burst["arrival_time_compression"] >= 10
    # Timestamp arithmetic has finite resolution; permit tiny rate rounding.
    assert burst["trace_arrival_orders_per_second"] >= 1500 * 0.999
    normal_session, burst_session = detector.sessions
    assert normal_session.orders[-1]["card_orders_1m"] == 1
    assert burst_session.orders[-1]["card_orders_1m"] == 5
    assert normal_session.orders[-1]["history_count"] == 5
    assert burst_session.orders[-1]["history_count"] == 5
    assert burst_session.orders[-1]["card_merchants_5m"] == 3
    assert normal_session.orders[0]["timestamp"] == burst_session.orders[0]["timestamp"]


def test_replay_uses_available_orders_without_inventing_extra_work(orders):
    report = benchmark_orders(RecordingDetector(), orders, max_orders=100, warmup=1)
    assert report["normal"]["orders_measured"] == 5
    assert report["simulated_drop_day_burst"]["orders_measured"] == 5


def test_retimed_replay_rebuilds_bin_counts_instead_of_using_original_clock_totals(orders):
    orders["processor_bin_orders_1m"] = 100
    orders["processor_bin_orders_5m"] = 200
    detector = RecordingDetector()
    report = benchmark_orders(detector, orders, max_orders=3, warmup=2)
    normal, burst = detector.sessions
    assert normal.orders[-1]["bin_orders_1m"] == 1
    assert burst.orders[-1]["bin_orders_1m"] == 5
    assert "Recomputed from replay input" in report["bin_counter_context"]


def test_single_order_has_latency_but_no_measurable_arrival_rate(orders):
    report = benchmark_orders(RecordingDetector(), orders.iloc[:1], max_orders=1, warmup=0)
    for key in ("normal", "simulated_drop_day_burst"):
        assert report[key]["orders_measured"] == 1
        assert report[key]["trace_arrival_orders_per_second"] is None
        assert report[key]["simulated_serial_queue_wait_p95_ms"] == 0


def test_insufficient_warmup_data_rejected(orders):
    with pytest.raises(ValueError, match="more orders"):
        benchmark_orders(RecordingDetector(), orders.iloc[:2], warmup=2)


@pytest.mark.parametrize("option,value", [
    ("max_orders", 0), ("max_orders", -1), ("max_orders", 1.5), ("max_orders", True),
    ("warmup", -1), ("warmup", 1.5), ("warmup", True),
    ("burst_multiplier", 0.5), ("burst_multiplier", float("nan")),
    ("burst_multiplier", float("inf")), ("burst_multiplier", True),
    ("burst_multiplier", "bad"), ("burst_multiplier", None),
    ("burst_orders_per_second", 0), ("burst_orders_per_second", -1),
    ("burst_orders_per_second", float("nan")), ("burst_orders_per_second", float("inf")),
    ("burst_orders_per_second", True), ("burst_orders_per_second", "bad"),
    ("burst_orders_per_second", None),
])
def test_invalid_benchmark_options_raise_clear_value_errors(orders, option, value):
    arguments = {"warmup": 0, option: value}
    with pytest.raises(ValueError, match=option):
        benchmark_orders(RecordingDetector(), orders, **arguments)
