"""Processor BIN context preserves past population activity in partial feeds."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scalper_detector.data import (
    NUMERIC_FEATURES,
    PaymentFeatureStore,
    engineer_events,
    linked_components,
    validate_events,
)


def _orders() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame([
        {
            "order_id": f"order-{number}",
            "timestamp": (start + timedelta(seconds=seconds)).isoformat(),
            "card_id": f"card-{number}",
            "card_bin": "001234",
            "merchant_id": f"merchant-{number}",
            "category": "tickets",
            "amount": 160.0,
            "event_id": "concert-shared",
            "quantity": 2,
            "ticket_limit": 4,
            "split_group": partition,
            "is_scalper": number % 2,
        }
        for number, (seconds, partition) in enumerate(
            [(0, "train"), (10, "validation"), (70, "test")]
        )
    ])


@pytest.mark.parametrize("missing", [None, np.nan, ""])
def test_paired_missing_population_counts_fall_back_to_local_feed(missing):
    orders = _orders()
    orders["processor_bin_orders_1m"] = missing
    orders["processor_bin_orders_5m"] = missing
    validated = validate_events(orders)
    assert validated.processor_bin_orders_1m.isna().all()
    assert validated.processor_bin_orders_5m.isna().all()
    features = engineer_events(orders)
    assert features.bin_orders_1m.tolist() == [1, 2, 2]
    assert features.bin_orders_5m.tolist() == [1, 2, 3]


def test_absent_population_columns_use_local_causal_counts():
    features = engineer_events(_orders())
    assert features.bin_orders_1m.tolist() == [1, 2, 2]
    assert features.bin_orders_5m.tolist() == [1, 2, 3]
    assert features.card_orders_1m.tolist() == [1, 1, 1]
    assert features.cluster_orders_1m.tolist() == [1, 1, 1]


def test_supplied_processor_counts_override_only_bin_population_features():
    orders = _orders()
    local = engineer_events(orders)
    orders["processor_bin_orders_1m"] = [91, 97, 112]
    orders["processor_bin_orders_5m"] = [300, 309, 337]
    features = engineer_events(orders)
    assert features.bin_orders_1m.tolist() == [91, 97, 112]
    assert features.bin_orders_5m.tolist() == [300, 309, 337]
    local_features = [name for name in NUMERIC_FEATURES if not name.startswith("bin_")]
    pd.testing.assert_frame_equal(features[local_features], local[local_features])
    assert features.linked_cards_count.eq(1).all()
    assert len(set(linked_components(orders).values())) == 3

    online = PaymentFeatureStore()
    observed = [online.update(row) for row in orders.to_dict("records")]
    assert [row["bin_orders_1m"] for row in observed] == [91, 97, 112]
    assert [row["cluster_orders_5m"] for row in observed] == [1, 1, 1]


def test_missing_rows_use_local_counts_without_inheriting_processor_totals():
    orders = _orders()
    orders["processor_bin_orders_1m"] = [100, np.nan, np.nan]
    orders["processor_bin_orders_5m"] = [500, np.nan, np.nan]
    features = engineer_events(orders)
    assert features.bin_orders_1m.tolist() == [100, 2, 2]
    assert features.bin_orders_5m.tolist() == [500, 2, 3]


@pytest.mark.parametrize(
    "provided",
    [
        {"processor_bin_orders_1m": 2},
        {"processor_bin_orders_5m": 2},
        {"processor_bin_orders_1m": 2, "processor_bin_orders_5m": np.nan},
        {"processor_bin_orders_1m": np.nan, "processor_bin_orders_5m": 2},
    ],
)
def test_population_context_requires_both_counts_or_neither(provided):
    order = _orders().iloc[[0]].assign(**provided)
    with pytest.raises(ValueError, match="both processor_bin_orders"):
        validate_events(order)


@pytest.mark.parametrize(
    "one_minute,five_minutes",
    [
        (0, 1), (-1, 1), (1.5, 2),
        (1, 0), (1, -1), (1, 2.5),
        (np.inf, np.inf), (1, np.inf), ("invalid", 2),
    ],
)
def test_population_context_rejects_nonpositive_fractional_or_nonfinite_counts(
    one_minute, five_minutes,
):
    order = _orders().iloc[[0]].assign(
        processor_bin_orders_1m=one_minute,
        processor_bin_orders_5m=five_minutes,
    )
    with pytest.raises(ValueError, match="processor_bin_orders"):
        validate_events(order)


def test_five_minute_population_count_cannot_be_smaller_than_one_minute():
    order = _orders().iloc[[0]].assign(
        processor_bin_orders_1m=5, processor_bin_orders_5m=4,
    )
    with pytest.raises(ValueError, match="cannot be less"):
        validate_events(order)


def test_integer_count_strings_from_csv_are_accepted():
    order = _orders().iloc[[0]].assign(
        processor_bin_orders_1m="12", processor_bin_orders_5m="45",
    )
    features = engineer_events(order)
    assert features.bin_orders_1m.iloc[0] == 12
    assert features.bin_orders_5m.iloc[0] == 45


def test_past_processor_context_is_not_rewritten_by_future_counts():
    orders = _orders().assign(
        processor_bin_orders_1m=[31, 33, 999],
        processor_bin_orders_5m=[100, 120, 10_000],
    )
    prefix = engineer_events(orders.iloc[:2])
    complete = engineer_events(orders)
    pd.testing.assert_frame_equal(
        prefix[list(NUMERIC_FEATURES)], complete.iloc[:2][list(NUMERIC_FEATURES)],
    )
    assert complete.bin_orders_1m.tolist() == [31, 33, 999]


def test_unlabeled_global_bin_context_can_follow_a_partial_component_feed():
    orders = _orders()
    background = engineer_events(orders.drop(columns="is_scalper"))
    enriched = orders.assign(
        processor_bin_orders_1m=background.bin_orders_1m.to_numpy(),
        processor_bin_orders_5m=background.bin_orders_5m.to_numpy(),
    )
    # This test component sees only its own order; explicit past-population
    # context retains preceding activity from the other independent buyers.
    partial = engineer_events(enriched[enriched.split_group.eq("test")])
    without_context = engineer_events(orders[orders.split_group.eq("test")])
    assert partial.bin_orders_1m.iloc[0] == 2
    assert partial.bin_orders_5m.iloc[0] == 3
    assert without_context.bin_orders_1m.iloc[0] == 1
    assert without_context.bin_orders_5m.iloc[0] == 1
    assert partial.card_orders_1m.iloc[0] == partial.cluster_orders_1m.iloc[0] == 1
    assert partial.linked_cards_count.iloc[0] == 1


def test_global_population_counts_ignore_label_values_and_split_metadata():
    orders = _orders()
    original = engineer_events(orders)
    changed = engineer_events(orders.assign(is_scalper=1 - orders.is_scalper,
                                            split_group="different-assignment"))
    unlabeled = engineer_events(orders.drop(columns=["is_scalper", "split_group"]))
    for alternative in (changed, unlabeled):
        pd.testing.assert_frame_equal(
            original[list(NUMERIC_FEATURES)], alternative[list(NUMERIC_FEATURES)],
        )
