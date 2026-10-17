"""Measured local CPU replay and explicit, simulated drop-day arrival pressure."""

from __future__ import annotations

import platform
from numbers import Real
from time import perf_counter

import numpy as np
import pandas as pd

from .data import LABEL, validate_events


def benchmark_orders(detector, frame: pd.DataFrame, *, max_orders: int = 300,
                     burst_multiplier: float = 10.0, warmup: int = 20,
                     burst_orders_per_second: float = 1000.0) -> dict:
    if isinstance(max_orders, bool) or not isinstance(max_orders, int) or max_orders < 1:
        raise ValueError("max_orders must be positive.")
    if (isinstance(burst_multiplier, bool) or not isinstance(burst_multiplier, Real)
            or not np.isfinite(burst_multiplier) or burst_multiplier < 1):
        raise ValueError("burst_multiplier must be finite and at least 1.")
    if (isinstance(burst_orders_per_second, bool) or not isinstance(burst_orders_per_second, Real)
            or not np.isfinite(burst_orders_per_second) or burst_orders_per_second <= 0):
        raise ValueError("burst_orders_per_second must be positive and finite.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer.")
    events = validate_events(frame.drop(columns=[LABEL], errors="ignore")).head(max_orders + warmup)
    # External counters describe the original clock. Recompute from replay
    # input in both cases instead of carrying stale counts into a time warp.
    events = events.drop(columns=["processor_bin_orders_1m", "processor_bin_orders_5m"], errors="ignore")
    if len(events) <= warmup:
        raise ValueError("Provide more orders than the requested benchmark warmup.")
    measured_span = (events.timestamp.iloc[-1] - events.timestamp.iloc[warmup]).total_seconds()
    measured_intervals = len(events) - warmup - 1
    # Sparse payment traces need much more than a ten-fold compression to
    # simulate a release burst. Compress to at least the requested offered rate.
    compression = max(float(burst_multiplier),
                      burst_orders_per_second * measured_span / measured_intervals if measured_intervals > 0 else 1.0)

    def replay(multiplier: float) -> dict:
        stream = events.copy()
        origin = stream.timestamp.iloc[0]
        stream["timestamp"] = origin + (stream.timestamp - origin) / multiplier
        stream["seconds_since_release"] = stream.seconds_since_release / multiplier
        session = detector.new_session()
        records = stream.to_dict("records")
        for order in records[:warmup]:
            session.score_order(order)
        elapsed_ms, neural_ms, arrivals = [], [], []
        wall_start = perf_counter()
        for order in records[warmup:]:
            start = perf_counter()
            session.score_order(order)
            elapsed_ms.append((perf_counter() - start) * 1000)
            neural_ms.append(session.last_neural_ms)
            arrivals.append((order["timestamp"] - origin).total_seconds() * 1000)
        wall_seconds = perf_counter() - wall_start
        # A serial queue simulation driven by measured service times. No real
        # concurrent clients, traffic generator, or hardware load is implied.
        finish = arrivals[0]
        waits = []
        for arrival, duration in zip(arrivals, elapsed_ms):
            start = max(arrival, finish)
            waits.append(start - arrival)
            finish = start + duration
        arrival_span = (arrivals[-1] - arrivals[0]) / 1000
        return {"orders_measured": len(elapsed_ms), "warmup_orders": warmup,
                "arrival_time_compression": multiplier,
                "end_to_end_ms": {"p50": float(np.percentile(elapsed_ms, 50)),
                                  "p95": float(np.percentile(elapsed_ms, 95)),
                                  "p99": float(np.percentile(elapsed_ms, 99))},
                "neural_only_ms": {"p50": float(np.percentile(neural_ms, 50)),
                                   "p95": float(np.percentile(neural_ms, 95))},
                "serial_replay_orders_per_second": len(elapsed_ms) / wall_seconds,
                "trace_arrival_orders_per_second": (len(arrivals) - 1) / arrival_span if arrival_span > 0 else None,
                "simulated_serial_queue_wait_p95_ms": float(np.percentile(waits, 95))}

    return {"backend": "pytorch_cpu_plus_numpy_forest", "hardware": platform.machine(),
            "platform": platform.platform(), "telum_measured": False,
            "scope": "Single-process serial replay: validation, causal lookup, preprocessing, LSTM, forest, calibration, reasons, policy",
            "bin_counter_context": "Recomputed from replay input; supplied population counters are excluded when retiming arrivals",
            "exclusions": ["network", "persistent feature-store I/O", "real authorization system", "concurrent load"],
            "burst_requested_min_orders_per_second": float(burst_orders_per_second),
            "normal": replay(1.0), "simulated_drop_day_burst": replay(compression),
            "hardware_comparison": {"cpu": "measured above", "telum": "not measured; requires target IBM hardware and compiled model"}}
