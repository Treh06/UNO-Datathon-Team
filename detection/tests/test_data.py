"""Payment validation, causal cross-merchant graphs, and history integrity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import torch

from scalper_detector.data import (
    LABEL, NUMERIC_FEATURES, REQUIRED_COLUMNS, ActivitySequences, EventPreprocessor,
    PaymentFeatureStore, engineer_events, linked_components, read_events,
    split_accounts, validate_events,
)


def order(number=0, *, seconds=None, card="card-a", merchant="merchant-a", **changes):
    return {
        "order_id": f"order-{number}",
        "timestamp": (datetime(2025, 1, 1, tzinfo=timezone.utc)
                      + timedelta(seconds=number if seconds is None else seconds)).isoformat(),
        "card_id": card, "card_bin": "001234", "merchant_id": merchant,
        "category": "tickets", "amount": 100.0, **changes,
    }


@pytest.fixture
def events():
    return pd.DataFrame([
        order(0, card="a", is_scalper=0), order(1, card="b", is_scalper=1),
        order(2, card="a", is_scalper=1), order(3, card="b", is_scalper=0),
    ])


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_validation_requires_payment_fields(events, column):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_events(events.drop(columns=column))


@pytest.mark.parametrize("column", ["order_id", "card_id", "card_bin", "merchant_id", "category"])
@pytest.mark.parametrize("value", [None, "   "])
def test_required_tokens_cannot_be_missing(events, column, value):
    events.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        validate_events(events)


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1, "invalid"])
def test_amount_requires_positive_finite_number(events, value):
    events["amount"] = events.amount.astype(object)
    events.loc[0, "amount"] = value
    with pytest.raises(ValueError, match="amount"):
        validate_events(events)


@pytest.mark.parametrize("value", [None, "not-a-date"])
def test_invalid_dates_rejected(events, value):
    events.loc[0, "timestamp"] = value
    with pytest.raises(ValueError, match="timestamp"):
        validate_events(events)


@pytest.mark.parametrize("value", [None, -1, 2, "scalper"])
def test_invalid_per_order_labels_rejected(events, value):
    events[LABEL] = events[LABEL].astype(object)
    events.loc[0, LABEL] = value
    with pytest.raises(ValueError, match=LABEL):
        validate_events(events)


def test_labels_may_change_on_one_card_and_are_optional_for_inference(events):
    actual = validate_events(events, require_labels=True)
    assert actual[actual.card_id.eq("a")][LABEL].tolist() == [0, 1]
    assert LABEL not in validate_events(events.drop(columns=LABEL))
    with pytest.raises(ValueError, match="Training requires"):
        validate_events(events.drop(columns=LABEL), require_labels=True)


def test_optional_defaults_and_known_flags_survive_revalidation(events):
    actual = validate_events(events)
    pd.testing.assert_frame_equal(actual, validate_events(actual))
    assert actual.quantity.eq(1).all()
    assert actual.quantity_available.eq(0).all()
    assert actual.card_age_days.eq(0).all() and actual.card_age_known.eq(0).all()
    assert actual.seconds_since_release.eq(0).all() and actual.release_time_known.eq(0).all()
    assert actual.ticket_limit.eq(0).all() and actual.ticket_limit_known.eq(0).all()
    assert actual.event_id_known.eq(0).all()
    assert actual.card_type.eq("unknown").all()
    for column in ("shipping_hash", "billing_hash", "email_hash", "device_id", "event_id"):
        assert actual[column].eq("").all()
    events["quantity"] = [2, None, 3, None]
    events["card_age_days"] = [0, None, 40, None]
    events["seconds_since_release"] = [0, None, 10, None]
    events["ticket_limit"] = [4, None, 6, None]
    events["event_id"] = ["performance-a", None, "performance-b", ""]
    known = validate_events(events)
    assert known.quantity_available.tolist() == [1, 0, 1, 0]
    assert known.card_age_known.tolist() == [1, 0, 1, 0]
    assert known.release_time_known.tolist() == [1, 0, 1, 0]
    assert known.ticket_limit_known.tolist() == [1, 0, 1, 0]
    assert known.event_id_known.tolist() == [1, 0, 1, 0]
    pd.testing.assert_frame_equal(known, validate_events(known))


@pytest.mark.parametrize("column,value", [
    ("quantity", 1.5), ("quantity", 0), ("quantity", -1),
    ("card_age_days", -1), ("seconds_since_release", -1),
    ("quantity_available", 2), ("card_age_known", np.nan),
    ("ticket_limit", 0), ("ticket_limit", -1), ("ticket_limit", 1.5),
    ("ticket_limit", np.inf), ("ticket_limit_known", 2),
])
def test_invalid_optional_fields_rejected(events, column, value):
    events[column] = value
    with pytest.raises(ValueError, match="quantity|card_age|seconds_since_release|ticket_limit"):
        validate_events(events)


@pytest.mark.parametrize("category", ["sneakers", "electronics", "collectibles", "unknown", "concert"])
def test_non_ticket_categories_are_rejected_by_every_preprocessor_entry(events, category):
    preprocessor = EventPreprocessor().fit(events)
    wrong = events.assign(category=category)
    with pytest.raises(ValueError, match="Only category='tickets'"):
        validate_events(wrong)
    with pytest.raises(ValueError, match="Only category='tickets'"):
        preprocessor.transform(wrong)
    engineered = engineer_events(events).assign(category=category)
    with pytest.raises(ValueError, match="Only category='tickets'"):
        preprocessor.transform_engineered(engineered)


def test_ticket_category_is_normalized_without_consuming_invalid_online_orders():
    assert validate_events(pd.DataFrame([order(0, category=" Tickets ")])).category.item() == "tickets"
    store = PaymentFeatureStore()
    with pytest.raises(ValueError, match="tickets"):
        store.update(order(0, category="sneakers"))
    assert store.update(order(0))["history_count"] == 1


def test_event_known_flag_preserves_unknown_state_and_rejects_false_known(events):
    unknown = validate_events(events.assign(event_id="do-not-use", event_id_known=0))
    assert unknown.event_id.eq("").all() and unknown.event_id_known.eq(0).all()
    pd.testing.assert_frame_equal(unknown, validate_events(unknown))
    with pytest.raises(ValueError, match="event_id cannot be missing"):
        validate_events(events.assign(event_id="", event_id_known=1))
    with pytest.raises(ValueError, match="event_id_known"):
        validate_events(events.assign(event_id_known=2))


def test_order_ids_are_unique_while_target_event_ids_may_repeat(events):
    duplicates = events.copy()
    duplicates.loc[1, "order_id"] = duplicates.loc[0, "order_id"]
    with pytest.raises(ValueError, match="Duplicate order_id"):
        validate_events(duplicates)
    events["event_id"] = ["event-a", "event-a", "", ""]
    assert validate_events(events).event_id.tolist() == ["event-a", "event-a", "", ""]
    store = PaymentFeatureStore()
    assert len([store.update(row) for row in events.to_dict("records")]) == 4


def test_csv_preserves_zero_tokens_and_global_stable_order(events, tmp_path):
    events["card_id"] = ["001", "002", "001", "002"]
    events["merchant_id"] = "0005"
    events.loc[1, "timestamp"] = events.loc[0, "timestamp"]
    path = tmp_path / "payments.csv"
    events.iloc[[2, 1, 0, 3]].to_csv(path, index=False)
    actual = read_events(str(path), require_labels=True)
    assert set(actual.card_id) == {"001", "002"}
    assert actual.card_bin.eq("001234").all()
    assert actual.merchant_id.eq("0005").all()
    assert actual.timestamp.is_monotonic_increasing
    assert actual.order_id.tolist() == ["order-1", "order-0", "order-2", "order-3"]
    assert str(actual.timestamp.dt.tz) == "UTC"


def test_card_velocity_is_cross_merchant_and_window_boundaries_are_inclusive():
    frame = engineer_events(pd.DataFrame([
        order(0, seconds=0), order(1, seconds=30, merchant="merchant-b"),
        order(2, seconds=60, merchant="merchant-c"), order(3, seconds=61),
        order(4, seconds=3601),
    ]))
    assert frame.card_orders_1m.tolist() == [1, 2, 3, 3, 1]
    assert frame.card_orders_5m.tolist() == [1, 2, 3, 4, 1]
    assert frame.card_merchants_5m.tolist() == [1, 2, 3, 3, 1]
    assert frame.card_orders_1h.tolist() == [1, 2, 3, 4, 4]
    assert frame.seconds_since_previous.tolist() == [0, 30, 30, 1, 3540]
    assert frame.history_count.tolist() == [1, 2, 3, 4, 5]


def test_window_boundaries_preserve_subsecond_precision():
    frame = engineer_events(pd.DataFrame([
        order(0, timestamp="2025-01-01T00:00:00.900000000Z"),
        order(1, timestamp="2025-01-01T00:01:00.900000000Z"),
        order(2, timestamp="2025-01-01T00:01:00.900000001Z"),
    ]))
    assert frame.card_orders_1m.tolist() == [1, 2, 2]


def test_linked_cards_accumulate_cross_site_activity():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", shipping_hash="shared", merchant="site-a"),
        order(1, card="b", shipping_hash="shared", merchant="site-b"),
        order(2, card="b", merchant="site-c"),
    ]))
    assert frame.linked_cards_count.tolist() == [1, 2, 2]
    assert frame.shipping_linked_cards.tolist() == [1, 2, 0]
    assert frame.cluster_orders_1m.tolist() == [1, 2, 3]
    assert frame.cluster_merchants_5m.tolist() == [1, 2, 3]
    assert frame.cluster_cards_1h.tolist() == [1, 2, 2]
    assert frame.card_orders_1m.tolist() == [1, 1, 2]


def test_bridge_event_merges_both_prior_cluster_histories():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", shipping_hash="address-a", merchant="site-a"),
        order(1, card="b", email_hash="email-b", merchant="site-b"),
        order(2, card="c", shipping_hash="address-a", merchant="site-c"),
        order(3, card="d", email_hash="email-b", merchant="site-d"),
        order(4, card="c", email_hash="email-b", merchant="site-e"),
    ]))
    assert frame.cluster_orders_1m.tolist() == [1, 1, 2, 2, 5]
    assert frame.linked_cards_count.tolist() == [1, 1, 2, 2, 4]
    assert frame.iloc[-1].cluster_cards_1h == 4
    assert frame.iloc[-1].cluster_merchants_5m == 5
    assert set(linked_components(frame).values()) == {"a"}


def test_missing_identity_and_shared_bins_do_not_link_cards():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", shipping_hash=None, email_hash="", device_id=None),
        order(1, card="b", shipping_hash="", email_hash=None, device_id=""),
    ]))
    assert frame.bin_orders_1m.tolist() == [1, 2]
    assert frame.cluster_orders_1m.tolist() == [1, 1]
    assert frame.linked_cards_count.tolist() == [1, 1]
    assert frame.shipping_linked_cards.eq(0).all()
    assert frame.email_linked_cards.eq(0).all() and frame.device_linked_cards.eq(0).all()
    assert len(set(linked_components(frame).values())) == 2


def test_link_namespaces_keep_devices_separate_but_addresses_can_cross_fields():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", shipping_hash="same-value"),
        order(1, card="b", device_id="same-value"),
        order(2, card="c", billing_hash="same-value"),
    ]))
    assert frame.linked_cards_count.tolist() == [1, 1, 2]
    assert frame.cluster_orders_1m.tolist() == [1, 1, 2]


def test_periodic_uniform_baskets_have_high_regularity():
    frame = engineer_events(pd.DataFrame([order(i, seconds=i * 10) for i in range(4)]))
    assert frame.interval_regularity.tolist() == [0, 0, 1, 1]
    assert frame.basket_uniformity_5m.tolist() == [0, 1, 1, 1]
    varied = engineer_events(pd.DataFrame([order(0), order(1, amount=300), order(2, seconds=100)]))
    assert 0 < varied.iloc[-1].interval_regularity < 1
    assert 0 < varied.iloc[-1].basket_uniformity_5m < 1


def test_future_orders_cannot_rewrite_prior_numeric_or_graph_features(events):
    first = events.iloc[:2].copy()
    first["shipping_hash"] = ["address-a", "address-b"]
    bridge = pd.DataFrame([order(4, card="a", shipping_hash="address-b", device_id="new-device", is_scalper=1)])
    before = engineer_events(first)
    after = engineer_events(pd.concat([first, bridge], ignore_index=True))
    pd.testing.assert_frame_equal(before, after.iloc[:2].reset_index(drop=True))
    preprocessor = EventPreprocessor().fit(events)
    np.testing.assert_array_equal(preprocessor.transform(first)[1], preprocessor.transform(pd.concat([first, bridge]))[1][:2])


def test_online_features_match_batch_and_reject_late_or_repeated_events(events):
    batch = engineer_events(events)
    store = PaymentFeatureStore()
    actual = pd.DataFrame([store.update(row) for row in events.to_dict("records")])
    pd.testing.assert_frame_equal(batch, actual)
    with pytest.raises(ValueError, match="Duplicate order_id"):
        store.update(events.iloc[-1].to_dict())
    with pytest.raises(ValueError, match="nondecreasing"):
        store.update(order(8, seconds=0))
    next_order = store.update(order(9, seconds=4, card="a"))
    assert next_order["history_count"] == 3
    assert next_order["card_orders_1m"] == 3


def _grouped_orders():
    rows = []
    for label in (0, 1):
        for group in range(5):
            identity = f"address-{label}-{group}"
            for card in range(2):
                token = f"card-{label}-{group}-{card}"
                rows.append(order(len(rows), card=token, is_scalper=label))
                rows.append(order(len(rows), card=token, shipping_hash=identity, is_scalper=label))
    return pd.DataFrame(rows)


def test_partitioning_keeps_complete_future_linked_components_together():
    original = _grouped_orders()
    partitions = split_accounts(original, seed=21)
    assert set(partitions) == {"train", "validation", "calibration", "test"}
    repeated = split_accounts(original, seed=21)
    observed = pd.concat([frame.assign(partition=name) for name, frame in partitions.items()])
    assert len(observed) == len(original)
    assert observed.order_id.nunique() == len(original)
    assert observed.groupby("card_id").partition.nunique().eq(1).all()
    assert observed.groupby("split_group").partition.nunique().eq(1).all()
    with_hash = observed[observed.shipping_hash.ne("")]
    assert with_hash.groupby("shipping_hash").partition.nunique().eq(1).all()
    for name, frame in partitions.items():
        assert set(frame[LABEL]) == {0, 1}
        pd.testing.assert_frame_equal(frame, repeated[name])
        assert frame.timestamp.is_monotonic_increasing


def test_partitioning_uses_component_max_order_label_and_checks_support(events):
    frame = _grouped_orders()
    # A card/order can be normal inside a component containing positive orders.
    frame.loc[(frame.card_id.str.startswith("card-1")) & frame.shipping_hash.isna(), LABEL] = 0
    split_accounts(frame)
    with pytest.raises(ValueError, match="at least four"):
        split_accounts(pd.DataFrame([order(0, card="a", is_scalper=0), order(1, card="b", is_scalper=1)]))
    with pytest.raises(ValueError, match="all-normal and positive"):
        split_accounts(frame.assign(is_scalper=0))


def test_preprocessor_ignores_tokens_labels_and_preserves_roundtrip(events):
    fitted = EventPreprocessor().fit(events)
    renamed = events.copy()
    for column in ("order_id", "card_id", "card_bin", "merchant_id"):
        renamed[column] = "renamed-" + renamed[column]
    renamed[LABEL] = 1 - renamed[LABEL]
    changed = EventPreprocessor().fit(renamed)
    assert fitted.to_dict() == changed.to_dict()
    np.testing.assert_array_equal(fitted.transform(renamed)[1], fitted.transform(events.drop(columns=LABEL))[1])
    restored = EventPreprocessor.from_dict(fitted.to_dict())
    np.testing.assert_array_equal(restored.transform(events)[1], fitted.transform(events)[1])
    assert restored.feature_names == fitted.feature_names
    assert all("card_id" not in name and "merchant_id" not in name and LABEL not in name for name in fitted.feature_names)


def test_unknown_card_types_and_online_transform_use_saved_state(events):
    fitted = EventPreprocessor().fit(events)
    state_before = fitted.to_dict()
    new = pd.DataFrame([order(10, card_type="prepaid")])
    enriched, values = fitted.transform(new)
    assert values.dtype == np.float32
    assert values.shape == (1, len(fitted.feature_names))
    assert np.isfinite(values).all()
    assert values[0, fitted.feature_names.index("card_type=<unknown>")] == 1
    assert values[0, fitted.feature_names.index("category=tickets")] == 1
    np.testing.assert_array_equal(values, fitted.transform_engineered(enriched))
    assert fitted.to_dict() == state_before
    with pytest.raises(ValueError, match="Fit the preprocessor"):
        EventPreprocessor().transform(new)


def test_windows_gather_same_card_from_noncontiguous_chronological_rows(events):
    frame = validate_events(events)
    features = np.array([[1], [10], [2], [20]], dtype=np.float32)
    windows = ActivitySequences(frame, features, sequence_length=3)
    expected = [([1, 0, 0], 1, 0), ([10, 0, 0], 1, 1), ([1, 2, 0], 2, 1), ([10, 20, 0], 2, 0)]
    for index, (values, length, label) in enumerate(expected):
        window, actual_length, actual_label = windows[index]
        torch.testing.assert_close(window[:, 0], torch.tensor(values, dtype=torch.float32))
        assert actual_length == length and actual_label.item() == label
    latest = ActivitySequences(frame, features, sequence_length=1, latest_only=True)
    assert latest.endpoints.tolist() == [2, 3]
    assert len(latest) == 2
    assert latest[0][0].item() == 2 and latest[1][0].item() == 20


def test_window_shape_checks_and_unlabeled_input(events):
    frame = validate_events(events).drop(columns=LABEL)
    features = np.ones((len(frame), 2), dtype=np.float32)
    assert ActivitySequences(frame, features)[0][2].item() == 0
    with pytest.raises(ValueError, match="positive"):
        ActivitySequences(frame, features, sequence_length=0)
    with pytest.raises(ValueError, match="Feature rows"):
        ActivitySequences(frame, features[:-1])
    with pytest.raises(ValueError, match="Feature rows"):
        ActivitySequences(frame, features[:, 0])


def test_all_requested_numeric_features_are_present_and_finite(events):
    enriched = engineer_events(events)
    assert np.isfinite(enriched[list(NUMERIC_FEATURES)].to_numpy()).all()


def test_finite_large_amounts_do_not_overflow_basket_statistics_or_scaling():
    frame = pd.DataFrame([order(0, amount=1e308), order(1, amount=1e307)])
    preprocessor = EventPreprocessor().fit(frame)
    enriched, features = preprocessor.transform(frame)
    assert np.isfinite(enriched[list(NUMERIC_FEATURES)].to_numpy()).all()
    assert np.isfinite(features).all()


EVENT_FEATURES = (
    "card_event_orders_5m", "cluster_event_orders_5m", "cluster_event_merchants_5m", "cluster_event_cards_5m",
    "card_event_tickets_1h", "cluster_event_tickets_1h", "cluster_event_tickets_24h",
    "cluster_event_quantity_coverage_1h", "cluster_event_limit_excess",
)


def test_same_event_cross_site_metrics_exclude_other_performances_and_unknown_quantities():
    frame = pd.DataFrame([
        order(0, card="a", merchant="site-a", event_id="concert-a", quantity=2, ticket_limit=4, shipping_hash="shared"),
        order(1, card="b", merchant="site-b", event_id="concert-a", ticket_limit=4, shipping_hash="shared"),
        order(2, card="a", merchant="other-site", event_id="concert-b", quantity=10, ticket_limit=4, shipping_hash="shared"),
        order(3, card="c", merchant="site-c", event_id="concert-a", quantity=3, ticket_limit=4, shipping_hash="shared"),
    ])
    enriched = engineer_events(frame)
    first = enriched.iloc[0]
    assert first.card_event_orders_5m == 1 and first.cluster_event_tickets_24h == 2
    assert first.cluster_event_limit_excess == 0
    missing_quantity = enriched.iloc[1]
    assert missing_quantity.quantity == 1 and missing_quantity.quantity_available == 0
    assert missing_quantity.cluster_event_tickets_1h == 2
    assert missing_quantity.cluster_event_quantity_coverage_1h == 0.5
    latest = enriched.iloc[-1]
    assert latest.cluster_orders_5m == 4
    assert latest.card_event_orders_5m == 1
    assert latest.cluster_event_orders_5m == 3
    assert latest.cluster_event_merchants_5m == 3
    assert latest.cluster_event_cards_5m == 3
    assert latest.card_event_tickets_1h == 3
    assert latest.cluster_event_tickets_1h == 5
    assert latest.cluster_event_tickets_24h == 5
    assert latest.cluster_event_quantity_coverage_1h == pytest.approx(2 / 3)
    assert latest.cluster_event_limit_excess == 1
    store = PaymentFeatureStore()
    online = pd.DataFrame([store.update(row) for row in frame.to_dict("records")])
    pd.testing.assert_frame_equal(enriched, online)


def test_missing_event_never_forms_an_unknown_event_group_even_with_known_limit():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", shipping_hash="shared", quantity=4, ticket_limit=2),
        order(1, card="b", shipping_hash="shared", quantity=6, ticket_limit=2, event_id=None),
        order(2, card="a", shipping_hash="shared", quantity=8, ticket_limit=2, event_id=""),
    ]))
    assert frame.event_id_known.eq(0).all()
    assert (frame[list(EVENT_FEATURES)].to_numpy() == 0).all()
    assert frame.iloc[-1].cluster_orders_5m == 3


def test_limit_excess_requires_current_known_limit_and_counts_only_known_tickets():
    frame = engineer_events(pd.DataFrame([
        order(0, event_id="concert-a", quantity=8),
        order(1, event_id="concert-a", quantity=None, ticket_limit=10),
        order(2, event_id="concert-a", quantity=4, ticket_limit=10),
        order(3, event_id="concert-a", quantity=1),
    ]))
    assert frame.cluster_event_tickets_24h.tolist() == [8, 8, 12, 13]
    assert frame.cluster_event_limit_excess.tolist() == [0, 0, 2, 0]
    assert frame.cluster_event_quantity_coverage_1h.tolist() == [1, 0.5, pytest.approx(2 / 3), 0.75]


def test_event_day_retention_preserves_exact_boundaries_without_inflating_hour_features():
    frame = engineer_events(pd.DataFrame([
        order(0, card="a", seconds=0, event_id="concert-a", quantity=2, shipping_hash="shared"),
        order(1, card="b", seconds=3601, event_id="concert-a", quantity=3, shipping_hash="shared"),
        order(2, card="a", seconds=86400, event_id="concert-a", quantity=4, shipping_hash="shared"),
        order(3, card="a", timestamp="2025-01-02T00:00:00.000000001Z", event_id="concert-a", quantity=1),
    ]))
    boundary = frame.iloc[2]
    assert boundary.cluster_event_tickets_24h == 9
    assert boundary.cluster_event_tickets_1h == 4
    assert boundary.card_event_tickets_1h == 4
    assert boundary.cluster_orders_1h == 1 and boundary.card_orders_1h == 1
    assert boundary.cluster_cards_1h == 1
    after = frame.iloc[3]
    assert after.cluster_event_tickets_24h == 8
    assert after.cluster_event_tickets_1h == 5
    assert after.cluster_orders_1h == 2 and after.cluster_cards_1h == 1


def test_late_identity_bridge_merges_earlier_event_ticket_histories_causally():
    frame = pd.DataFrame([
        order(0, card="a", seconds=0, event_id="concert-a", quantity=2, shipping_hash="address-a"),
        order(1, card="b", seconds=100, event_id="concert-a", quantity=3, email_hash="email-b"),
        order(2, card="c", seconds=5000, event_id="concert-a", quantity=1, ticket_limit=4,
              shipping_hash="address-a", email_hash="email-b"),
    ])
    before = engineer_events(frame.iloc[:2])
    after = engineer_events(frame)
    pd.testing.assert_frame_equal(before, after.iloc[:2].reset_index(drop=True))
    bridge = after.iloc[-1]
    assert bridge.cluster_event_tickets_24h == 6
    assert bridge.cluster_event_tickets_1h == 1
    assert bridge.cluster_event_orders_5m == 1
    assert bridge.cluster_event_limit_excess == 2
    assert bridge.linked_cards_count == 3


def test_four_component_partitions_have_expected_large_sample_proportions():
    frame = pd.DataFrame([
        order(index, card=f"card-{index}", is_scalper=int(index >= 100))
        for index in range(200)
    ])
    partitions = split_accounts(frame, seed=7)
    assert {key: len(values) for key, values in partitions.items()} == {
        "train": 120, "validation": 30, "calibration": 20, "test": 30,
    }
    for values in partitions.values():
        assert values[LABEL].mean() == 0.5


def test_exactly_four_components_per_class_populates_every_partition():
    frame = pd.DataFrame([order(index, card=f"card-{index}", is_scalper=int(index >= 4)) for index in range(8)])
    partitions = split_accounts(frame)
    assert all(len(partition) == 2 and set(partition[LABEL]) == {0, 1} for partition in partitions.values())


def test_different_event_fan_context_is_distinct_from_same_event_card_rotation():
    different = pd.DataFrame([
        order(index, card=f"card-{index % 3}", seconds=index * 15, shipping_hash="shared",
              event_id=f"performance-{index}", quantity=2)
        for index in range(6)
    ])
    same = different.assign(event_id="one-performance")
    diverse = engineer_events(different).iloc[-1]
    concentrated = engineer_events(same).iloc[-1]
    assert diverse.cluster_distinct_events_5m == 6
    assert diverse.cluster_same_event_order_fraction_5m == pytest.approx(1 / 6)
    assert diverse.cluster_event_identity_coverage_5m == 1
    assert concentrated.cluster_distinct_events_5m == 1
    assert concentrated.cluster_same_event_order_fraction_5m == 1
    assert concentrated.cluster_event_identity_coverage_5m == 1
    assert diverse.cluster_orders_5m == concentrated.cluster_orders_5m == 6


def test_event_concentration_excludes_unknown_targets_and_reports_missing_coverage():
    frame = engineer_events(pd.DataFrame([
        order(0, event_id="performance-a"), order(1, event_id=None),
        order(2, event_id="performance-b"), order(3, event_id="performance-a"),
    ]))
    assert frame.cluster_distinct_events_5m.tolist() == [1, 1, 2, 2]
    assert frame.cluster_same_event_order_fraction_5m.tolist() == [1, 0, 0.5, pytest.approx(2 / 3)]
    assert frame.cluster_event_identity_coverage_5m.tolist() == [1, 0.5, pytest.approx(2 / 3), 0.75]
    unknown = engineer_events(pd.DataFrame([order(0), order(1)]))
    for feature in ("cluster_distinct_events_5m", "cluster_same_event_order_fraction_5m",
                    "cluster_event_identity_coverage_5m"):
        assert unknown[feature].eq(0).all()


def test_event_concentration_updates_only_after_a_causal_identity_bridge():
    frame = pd.DataFrame([
        order(0, card="a", event_id="performance-a", shipping_hash="address-a"),
        order(1, card="b", event_id="performance-b", email_hash="email-b"),
        order(2, card="a", event_id="performance-a", email_hash="email-b"),
    ])
    prefix = engineer_events(frame.iloc[:2])
    complete = engineer_events(frame)
    pd.testing.assert_frame_equal(prefix, complete.iloc[:2].reset_index(drop=True))
    assert complete.cluster_distinct_events_5m.tolist() == [1, 1, 2]
    assert complete.iloc[-1].cluster_same_event_order_fraction_5m == pytest.approx(2 / 3)
    store = PaymentFeatureStore()
    online = pd.DataFrame([store.update(row) for row in frame.to_dict("records")])
    pd.testing.assert_frame_equal(online, complete)
