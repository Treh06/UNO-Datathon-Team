"""Synthetic payment orders are reproducible, coherent, and group-splittable."""

import numpy as np
import pandas as pd
import pytest

from scalper_detector.synthetic import generate_synthetic_data, generate_ticket_scenarios


def test_synthetic_payment_orders_are_reproducible_and_coherent():
    first = generate_synthetic_data(n_accounts=80, seed=7)
    pd.testing.assert_frame_equal(first, generate_synthetic_data(n_accounts=80, seed=7))
    assert not first.equals(generate_synthetic_data(n_accounts=80, seed=8))
    required = {"order_id", "timestamp", "card_id", "card_bin", "merchant_id", "category", "amount"}
    assert required <= set(first.columns)
    assert first.card_id.nunique() == 80
    assert first.order_id.is_unique
    assert first.groupby("card_id").size().between(12, 28).all()
    assert first.groupby("card_id").is_scalper.nunique().eq(1).all()
    assert first.groupby("card_id").is_scalper.first().sum() == 16
    assert first.groupby("card_id").card_bin.nunique().eq(1).all()
    assert set(first.category) == {"tickets"}
    assert set(first.performance_type) == {"concert", "sports", "theatre"}
    assert first.timestamp.str.endswith("Z").all()
    assert first.timestamp.is_monotonic_increasing
    assert first.amount.gt(0).all()
    assert first.quantity.dropna().between(1, 8).all()
    assert first.ticket_limit.dropna().isin([4, 6, 8]).all()
    assert first.ticket_limit.isna().any()
    assert first.seconds_since_release.dropna().ge(0).all()
    assert not {"event_type", "retail_price", "unit_price", "account_id"} & set(first.columns)

    observed = first.dropna(subset=["event_id", "seconds_since_release"])
    inferred_release = pd.to_datetime(observed.timestamp, utc=True) - pd.to_timedelta(
        observed.seconds_since_release, unit="s"
    )
    spread = inferred_release.groupby(observed.event_id).agg(["min", "max"])
    assert (spread["max"] - spread["min"]).dt.total_seconds().le(0.002).all()
    known_events = first.dropna(subset=["event_id"])
    assert known_events.groupby("event_id").ticket_limit.nunique().le(1).all()
    assert known_events.groupby("event_id").performance_date.nunique().eq(1).all()
    assert (pd.to_datetime(first.timestamp, utc=True)
            < pd.to_datetime(first.performance_date, utc=True)).all()
    for _, orders in first.groupby("card_id"):
        known_ages = orders.card_age_days.dropna()
        assert known_ages.is_monotonic_increasing
        assert orders.merchant_id.nunique() >= 2


@pytest.mark.parametrize("fraction", [0.0001, 0.2, 0.5, 0.9999])
def test_minimum_dataset_has_four_disjoint_buyer_groups_per_class(fraction):
    orders = generate_synthetic_data(n_accounts=20, scalper_fraction=fraction)
    # Shipping is scoped to a latent group and guaranteed on each card's first
    # event. Shared BIN prefixes are deliberately excluded from identity groups.
    known_shipping = orders.dropna(subset=["shipping_hash"])
    groups = known_shipping.groupby("shipping_hash").agg(
        cards=("card_id", "nunique"), label_count=("is_scalper", "nunique"),
        label=("is_scalper", "first"),
    )
    assert groups.cards.between(1, 4).all()
    assert groups.label_count.eq(1).all()
    assert groups.groupby("label").size().ge(4).all()
    assert known_shipping.card_id.nunique() == 20


def test_fast_high_quantity_behavior_and_bins_overlap_between_classes():
    orders = generate_synthetic_data(n_accounts=200, seed=42)
    bins_by_class = []
    for _, group in orders.groupby("is_scalper"):
        assert group.seconds_since_release.lt(3_600).any()
        assert group.quantity.ge(4).any()
        assert group.quantity.eq(1).any()
        assert group.seconds_since_release.gt(86_400).any()
        bins_by_class.append(set(group.card_bin))
    assert bins_by_class[0] & bins_by_class[1]
    assert orders[["shipping_hash", "email_hash", "device_id"]].isna().any().all()
    shipping_groups = orders.dropna(subset=["shipping_hash"]).groupby("shipping_hash")
    assert shipping_groups.card_id.nunique().gt(1).any()
    assert shipping_groups.merchant_id.nunique().ge(3).any()


@pytest.mark.parametrize("fraction", [0.0001, 0.2, 0.5, 0.9999])
def test_ticket_histories_support_four_component_isolated_partitions(fraction):
    from scalper_detector.data import EventPreprocessor, split_accounts, validate_events

    orders = validate_events(generate_synthetic_data(20, scalper_fraction=fraction), require_labels=True)
    partitions = split_accounts(orders)
    assert set(partitions) == {"train", "validation", "calibration", "test"}
    preprocessor = EventPreprocessor().fit(partitions["train"])
    observed_components = set()
    for partition in partitions.values():
        components = set(partition.split_group)
        assert not observed_components & components
        observed_components.update(components)
        assert set(partition.is_scalper) == {0, 1}
        _, features = preprocessor.transform(partition)
        assert np.isfinite(features).all()


def test_ticket_scenarios_have_explicit_separate_event_behaviors():
    from scalper_detector.data import linked_components, validate_events

    scenarios = generate_ticket_scenarios()
    pd.testing.assert_frame_equal(scenarios, generate_ticket_scenarios())
    assert set(scenarios.category) == {"tickets"}
    assert "is_scalper" not in scenarios
    assert scenarios.timestamp.is_monotonic_increasing
    assert scenarios.order_id.is_unique
    assert set(scenarios.scenario_name) == {
        "casual_fan", "family_booking", "cross_site_card_rotation", "different_events_fan",
    }
    rotation = scenarios[scenarios.scenario_name.eq("cross_site_card_rotation")]
    assert rotation.card_id.nunique() == 3
    assert rotation.groupby("card_id").size().eq(3).all()
    assert rotation.quantity.eq(2).all()
    assert rotation.quantity.sum() == 18
    assert rotation.event_id.nunique() == 1
    assert rotation.merchant_id.nunique() == 3
    times = pd.to_datetime(rotation.timestamp, utc=True)
    assert (times.max() - times.min()).total_seconds() <= 90
    family = scenarios[scenarios.scenario_name.eq("family_booking")]
    assert len(family) == 1 and family.quantity.iloc[0] == family.ticket_limit.iloc[0] == 4
    casual = scenarios[scenarios.scenario_name.eq("casual_fan")]
    assert len(casual) == 1 and casual.quantity.iloc[0] == 2
    avid = scenarios[scenarios.scenario_name.eq("different_events_fan")]
    assert len(avid) == avid.event_id.nunique() == 6
    assert avid.groupby("event_id").quantity.sum().le(avid.ticket_limit.iloc[0]).all()
    validated = validate_events(scenarios)
    components = linked_components(validated)
    scenarios["component"] = scenarios.card_id.map(components)
    assert scenarios.groupby("scenario_name").component.nunique().eq(1).all()
    assert scenarios.groupby("component").scenario_name.nunique().eq(1).all()


def test_legitimate_multi_event_bursts_are_substantial_training_examples():
    from scalper_detector.data import engineer_events, linked_components

    orders = generate_synthetic_data(200, seed=42)
    features = engineer_events(orders)
    normal = features[features.is_scalper.eq(0)].copy()
    components = linked_components(orders)
    normal["buyer_component"] = normal.card_id.map(components)
    diverse = (normal.cluster_distinct_events_5m.ge(5)
               & normal.cluster_same_event_order_fraction_5m.le(0.35)
               & normal.cluster_event_identity_coverage_5m.ge(0.60))
    # Broad coverage is intentional: velocity/regularity/amount uniformity alone
    # must not be a shortcut for the positive target in the training population.
    covered = normal.loc[diverse, "buyer_component"].nunique()
    assert covered / normal.buyer_component.nunique() >= 0.20
    assert (diverse & normal.basket_uniformity_5m.ge(0.95)
            & normal.interval_regularity.ge(0.90)).sum() >= 8
    assert (diverse & normal.quantity_available.eq(1) & normal.quantity.le(2)).sum() >= 30
    assert normal.loc[diverse, "cluster_merchants_5m"].ge(3).any()
    assert not {"bundle_fan", "fan_mode", "persona"} & set(orders.columns)

    # Shared onsale calendars are a catalog property, not a normal-only signal.
    observed = orders.dropna(subset=["event_id", "seconds_since_release"])
    release = (pd.to_datetime(observed.timestamp, utc=True)
               - pd.to_timedelta(observed.seconds_since_release, unit="s")).dt.round("s")
    catalog = observed.assign(release=release).groupby("event_id").agg(
        release=("release", "first"), label_count=("is_scalper", "nunique"),
    )
    shared_onsales = catalog.groupby("release").filter(lambda events: len(events) >= 5)
    assert len(shared_onsales) >= 20
    assert shared_onsales.label_count.eq(2).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_accounts": 19}, {"n_accounts": 20.5}, {"n_accounts": True},
        {"seed": -1}, {"seed": 1.2}, {"seed": True},
        {"scalper_fraction": 0}, {"scalper_fraction": 1},
        {"scalper_fraction": np.nan}, {"scalper_fraction": np.inf},
        {"scalper_fraction": "0.2"},
    ],
)
def test_invalid_generation_parameters_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_data(**kwargs)
