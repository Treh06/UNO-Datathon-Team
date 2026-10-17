"""Generate fictional ticket authorizations and independent ticket scenarios.

All identities, merchants, card tokens, BIN examples, orders, and labels here
are synthetic. No real payment data or card numbers are generated or accessed.
Labels describe sampled latent buyer personas, not a rule calculated from the
exported features. Enthusiastic collectors, legitimate group buyers, and subtle
scalpers deliberately overlap. These examples cannot validate real-world risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Persona:
    delay_seconds: float
    extra_quantity: float
    repeat_probability: float
    coordinated_probability: float
    coordination_jitter: float
    cross_merchant_probability: float
    card_age_days: float


_SCALPER_PERSONAS = (
    _Persona(90, 0.9, 0.88, 0.90, 22, 0.90, 340),
    _Persona(7_200, 0.6, 0.74, 0.70, 900, 0.82, 700),
    _Persona(43_200, 0.3, 0.44, 0.44, 7_200, 0.62, 850),
)

_NORMAL_PERSONAS = (
    _Persona(345_600, 0.8, 0.10, 0.30, 3_600, 0.45, 650),  # Casual fans.
    _Persona(3_600, 2.4, 0.22, 0.65, 60, 0.60, 800),  # Family bookings.
    _Persona(240, 0.7, 0.16, 0.78, 35, 0.75, 500),  # Fast, avid fans.
    _Persona(43_200, 3.1, 0.24, 0.60, 120, 0.65, 900),  # Corporate/group buyers.
)

_PERFORMANCE_TYPES = ("concert", "sports", "theatre")

# A BIN is a shared issuer/product prefix, never a buyer identity or card number.
_EXAMPLE_BINS = ("400100", "400200", "411100", "420100", "510100", "520100", "540100", "550100")

_COLUMNS = [
    "order_id",
    "timestamp",
    "card_id",
    "card_bin",
    "merchant_id",
    "category",
    "amount",
    "shipping_hash",
    "billing_hash",
    "email_hash",
    "device_id",
    "event_id",
    "card_type",
    "quantity",
    "seconds_since_release",
    "card_age_days",
    "ticket_limit",
    "performance_type",
    "performance_date",
    "is_scalper",
]


def _card_group_sizes(count: int, rng: np.random.Generator) -> list[int]:
    """Mostly one/two-card buyers, with at least four identities per class."""
    sizes = []
    weights = np.asarray([0.50, 0.35, 0.11, 0.04])
    while count:
        # Reserve one card for every still-required independent buyer identity.
        maximum = min(4, count - max(0, 3 - len(sizes)))
        probabilities = weights[:maximum] / weights[:maximum].sum()
        size = int(rng.choice(np.arange(1, maximum + 1), p=probabilities))
        sizes.append(size)
        count -= size
    return sizes


def generate_synthetic_data(
    n_accounts: int = 1000,
    seed: int = 42,
    scalper_fraction: float = 0.2,
) -> pd.DataFrame:
    """Return globally chronological ticket authorizations with persona labels.

    ``n_accounts`` is the number of tokenized cards (the legacy argument name is
    retained for CLI/API compatibility). Each card has 12–28 authorizations.
    Cards are arranged into disjoint buyer groups of one to four cards, connected
    through shared shipping/email/device identifiers. Identity availability
    varies by order; each card's first order exposes a group shipping token.

    ``n_accounts`` must be an integer >=20, ``seed`` a nonnegative integer, and
    ``scalper_fraction`` strictly between zero and one. Rounded positive-card
    counts are clipped to leave >=4 cards and >=4 independent buyer groups per
    class for group-isolated train/validation/calibration/test splits. Every row
    has category ``tickets``. Shared concert/sports/theatre performances have a
    common onsale time, ticket price, and per-buyer ticket limit across sites.
    Example BINs, catalog, dates, limits, and ID formats do not encode labels.

    Scalper personas tend to accumulate the same event through repeat orders
    and linked cards/sites. Normal families can make large bookings; avid fans
    buy many different events quickly. Some legitimate shared-identity groups
    contain multiple real buyers, so exceeding a graph-level limit is not proof
    of scalping. Optional fields are missing independently of label. No resale
    prices, listings, or clickstream are generated. About 30% of normal groups
    also make rapid, regular, low-quantity bookings for distinct performances
    whose season/venue onsales occur together. Both classes share that catalog.
    """
    if isinstance(n_accounts, bool) or not isinstance(n_accounts, Integral) or n_accounts < 20:
        raise ValueError("n_accounts must be an integer of at least 20 tokenized cards")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(scalper_fraction, bool)
        or not isinstance(scalper_fraction, Real)
        or not np.isfinite(scalper_fraction)
        or not 0 < scalper_fraction < 1
    ):
        raise ValueError("scalper_fraction must be a finite number between 0 and 1")

    rng = np.random.default_rng(int(seed))
    n_cards = int(n_accounts)
    n_scalper_cards = int(np.clip(round(n_cards * scalper_fraction), 4, n_cards - 4))
    groups = [(size, 0) for size in _card_group_sizes(n_cards - n_scalper_cards, rng)]
    groups += [(size, 1) for size in _card_group_sizes(n_scalper_cards, rng)]
    rng.shuffle(groups)
    card_tokens = rng.permutation(n_cards)
    identity_tokens = rng.permutation(len(groups))

    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    catalog = []
    type_indices = {kind: [] for kind in _PERFORMANCE_TYPES}
    bundles_by_type = {kind: [] for kind in _PERFORMANCE_TYPES}

    def add_performance(kind: str, release: datetime, unit_amount: float) -> int:
        index = len(catalog)
        type_indices[kind].append(index)
        catalog.append(
            {
                "event_id": f"performance_{index:03d}",
                "performance_type": kind,
                "performance_date": (release + timedelta(days=180)).date().isoformat(),
                "unit_amount": unit_amount,
                "ticket_limit": int(rng.choice([4, 6, 8], p=[0.55, 0.30, 0.15])),
                "release": release,
            }
        )
        return index

    for kind in _PERFORMANCE_TYPES:
        for _ in range(80):
            release = epoch + timedelta(seconds=float(rng.uniform(0, 180 * 86_400)))
            add_performance(kind, release, float(rng.uniform(35, 280)))
        # A venue/season can release several distinct performances together.
        # These releases and face prices exist before any buyer labels are used
        # and are available to both legitimate fans and inventory accumulators.
        for _ in range(8):
            release = epoch + timedelta(seconds=float(rng.uniform(0, 180 * 86_400)))
            face_price = float(rng.uniform(35, 280))
            common_price = rng.random() < 0.70
            bundle = []
            for _ in range(10):
                price = face_price if common_price else face_price * float(rng.uniform(0.80, 1.20))
                bundle.append(add_performance(kind, release, price))
            bundles_by_type[kind].append(bundle)

    normal_groups = [index for index, (_, label) in enumerate(groups) if not label]
    fan_count = max(1, int(round(0.30 * len(normal_groups))))
    bundle_fan_groups = set(rng.choice(normal_groups, size=fan_count, replace=False).tolist())

    rows = []
    card_offset = 0
    for group_index, (group_size, label) in enumerate(groups):
        if label:
            persona_index = int(rng.choice(3, p=[0.55, 0.25, 0.20]))
            persona = _SCALPER_PERSONAS[persona_index]
            target_min, target_max = ((3, 7), (5, 11), (10, 19))[persona_index]
            forced_repeats = (3, 2, 1)[persona_index]
        else:
            persona = _NORMAL_PERSONAS[int(rng.choice(4, p=[0.42, 0.23, 0.20, 0.15]))]
            target_min, target_max, forced_repeats = 8, 16, 0
        # These opaque synthetic tokens are scoped to this buyer group. No
        # identifier is reused by another latent buyer, even within one class.
        group_token = f"{int(identity_tokens[group_index]):08x}"
        identities = {
            "shipping_hash": f"ship_synth_{group_token}",
            "billing_hash": f"bill_synth_{group_token}",
            "email_hash": f"mail_synth_{group_token}",
            "device_id": f"device_synth_{group_token}",
        }
        primary_type = str(rng.choice(_PERFORMANCE_TYPES))
        targets = rng.choice(
            type_indices[primary_type],
            size=int(rng.integers(target_min, target_max)),
            replace=False,
        )
        if label and rng.random() < 0.20:
            bundle = bundles_by_type[primary_type][int(rng.integers(8))]
            targets = rng.choice(bundle, size=min(len(targets), len(bundle)), replace=False)
        merchants = rng.choice(16, size=int(rng.integers(3, 6)), replace=False)
        delay_center = persona.delay_seconds * float(rng.lognormal(0, 0.8))
        quantity_center = persona.extra_quantity * float(rng.lognormal(0, 0.4))
        repeat_probability = float(
            np.clip(persona.repeat_probability + rng.normal(0, 0.08), 0.10, 0.97)
        )
        campaign_delays = {
            int(index): float(np.clip(rng.lognormal(np.log(delay_center), 1.1), 0.1, 100 * 86_400))
            for index in targets
        }
        normal_remaining: dict[int, int] = {}
        event_order_counts: dict[int, int] = {}
        # A token graph can combine distinct legitimate buyers in a household,
        # workplace, or shared device. Keep this ambiguity in the normal class.
        separate_legitimate_buyers = not label and group_size > 1 and rng.random() < 0.15
        normal_bookings_per_event = 3 if separate_legitimate_buyers else 2

        for local_card in range(group_size):
            card_id = f"card_synth_{int(card_tokens[card_offset]):06d}"
            card_offset += 1
            card_bin = str(rng.choice(_EXAMPLE_BINS))
            card_type = str(rng.choice(["credit", "debit", "prepaid"], p=[0.65, 0.30, 0.05]))
            card_billing = identities["billing_hash"] if rng.random() < 0.45 else f"bill_{card_id}"
            fan_targets = []
            fan_times = np.empty(0)
            fan_quantity = int(rng.choice([1, 2]))
            fan_uniform_quantity = rng.random() < 0.75
            fan_fee_factor = float(rng.uniform(0.98, 1.08))
            if group_index in bundle_fan_groups:
                available_bundles = [
                    [index for index in bundle if index not in normal_remaining]
                    for bundle in bundles_by_type[primary_type]
                ]
                available_bundles = [bundle for bundle in available_bundles if len(bundle) >= 6]
                selected = available_bundles[int(rng.integers(len(available_bundles)))]
                burst_count = int(rng.integers(6, min(10, len(selected)) + 1))
                fan_targets = rng.choice(selected, size=burst_count, replace=False).tolist()
                gaps = (
                    np.ones(burst_count - 1)
                    if rng.random() < 0.55
                    else rng.uniform(0.4, 1.6, size=burst_count - 1)
                )
                gaps *= float(rng.uniform(60, 180)) / gaps.sum()
                fan_times = float(rng.lognormal(np.log(40), 0.65)) + np.r_[0.0, np.cumsum(gaps)]
            card_orders = []
            for order_number in range(int(rng.integers(12, 29))):
                fan_booking = order_number < len(fan_targets)
                if fan_booking:
                    target_index = int(fan_targets[order_number])
                    normal_remaining[target_index] = catalog[target_index]["ticket_limit"]
                elif not label:
                    reusable = [
                        index
                        for index, left in normal_remaining.items()
                        if left > 0 and event_order_counts[index] < normal_bookings_per_event
                    ]
                    if reusable and rng.random() < repeat_probability:
                        target_index = int(rng.choice(reusable))
                    else:
                        unused = [
                            index
                            for index in type_indices[primary_type]
                            if index not in normal_remaining
                        ]
                        if not unused or rng.random() > 0.72:
                            unused = [
                                index
                                for index in range(len(catalog))
                                if index not in normal_remaining
                            ]
                        target_index = int(rng.choice(unused))
                        multiplier = 2 if separate_legitimate_buyers else 1
                        normal_remaining[target_index] = (
                            catalog[target_index]["ticket_limit"] * multiplier
                        )
                elif order_number < forced_repeats:
                    target_index = int(targets[0])
                elif rng.random() < repeat_probability:
                    target_index = int(rng.choice(targets))
                elif rng.random() < 0.72:
                    target_index = int(rng.choice(type_indices[primary_type]))
                else:
                    target_index = int(rng.integers(len(catalog)))
                item = catalog[target_index]
                if not label and target_index not in campaign_delays:
                    campaign_delays[target_index] = float(rng.lognormal(np.log(delay_center), 1.1))
                coordinated = target_index in campaign_delays and (
                    order_number < forced_repeats or rng.random() < persona.coordinated_probability
                )
                if fan_booking:
                    delay = float(fan_times[order_number])
                elif coordinated:
                    # Shared target, linked cards, and nearby cross-site orders
                    # form a campaign; benign group buyers can do this too.
                    delay = campaign_delays[target_index] + float(
                        rng.uniform(1, max(2, persona.coordination_jitter))
                    )
                else:
                    delay = float(rng.lognormal(np.log(delay_center), 1.25))
                delay = float(np.clip(delay, 0.1, 120 * 86_400))
                if (
                    fan_booking
                    or order_number < forced_repeats
                    or rng.random() < persona.cross_merchant_probability
                ):
                    merchant = int(merchants[(order_number + local_card) % len(merchants)])
                else:
                    merchant = int(merchants[0])
                quantity = int(np.clip(1 + rng.poisson(quantity_center), 1, item["ticket_limit"]))
                if fan_booking:
                    quantity = fan_quantity if fan_uniform_quantity else int(rng.choice([1, 2]))
                if not label:
                    quantity = min(quantity, normal_remaining[target_index])
                    normal_remaining[target_index] -= quantity
                event_order_counts[target_index] = event_order_counts.get(target_index, 0) + 1
                fee_factor = fan_fee_factor if fan_booking else float(rng.uniform(0.95, 1.12))
                amount = round(item["unit_amount"] * quantity * fee_factor, 2)
                order = {
                    "timestamp": item["release"] + timedelta(seconds=delay),
                    "card_id": card_id,
                    "card_bin": card_bin,
                    "merchant_id": f"merchant_{merchant:03d}",
                    "category": "tickets",
                    "amount": amount,
                    "shipping_hash": identities["shipping_hash"] if rng.random() < 0.78 else None,
                    "billing_hash": card_billing if rng.random() < 0.70 else None,
                    "email_hash": identities["email_hash"] if rng.random() < 0.78 else None,
                    "device_id": identities["device_id"] if rng.random() < 0.65 else None,
                    "event_id": item["event_id"] if rng.random() < 0.90 else None,
                    "card_type": card_type,
                    "quantity": quantity if rng.random() < 0.90 else None,
                    "seconds_since_release": round(delay, 3) if rng.random() < 0.85 else None,
                    "ticket_limit": item["ticket_limit"] if rng.random() < 0.90 else None,
                    "performance_type": item["performance_type"],
                    "performance_date": item["performance_date"],
                    "is_scalper": label,
                }
                card_orders.append(order)
            card_orders.sort(key=lambda order: order["timestamp"])
            # Guarantee each card belongs to its intended connected component,
            # while preserving realistic missing identities on other orders.
            card_orders[0]["shipping_hash"] = identities["shipping_hash"]
            first_time = card_orders[0]["timestamp"]
            age_at_start = float(rng.lognormal(np.log(persona.card_age_days), 1.0))
            for order in card_orders:
                age = age_at_start + (order["timestamp"] - first_time).total_seconds() / 86_400
                order["card_age_days"] = round(age, 3) if rng.random() < 0.85 else None
                order["timestamp"] = (
                    order["timestamp"].isoformat(timespec="milliseconds").replace("+00:00", "Z")
                )
            rows.extend(card_orders)

    order_tokens = rng.permutation(len(rows))
    for order, token in zip(rows, order_tokens):
        order["order_id"] = f"order_synth_{int(token):08d}"
    return (
        pd.DataFrame(rows, columns=_COLUMNS)
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )


def generate_ticket_scenarios() -> pd.DataFrame:
    """Return deterministic, independent ticket examples for behavior inspection.

    Scenario names are descriptive metadata, never training features or verified
    outcome labels. The rotation example buys 18 tickets for one performance
    through three linked cards/sites; the avid fan buys six different events.
    """
    release = datetime(2025, 9, 1, 12, tzinfo=timezone.utc)
    rows = []

    def add(
        name: str, group: int, card: int, event: int, seconds: int, quantity: int, merchant: int
    ) -> None:
        rows.append(
            {
                "order_id": f"scenario_order_{len(rows):03d}",
                "timestamp": (release + timedelta(seconds=seconds))
                .isoformat()
                .replace("+00:00", "Z"),
                "card_id": f"scenario_card_{card:02d}",
                "card_bin": "400100",
                "merchant_id": f"merchant_{merchant:03d}",
                "category": "tickets",
                "amount": float(quantity * 80),
                "shipping_hash": f"scenario_ship_{group:02d}",
                "billing_hash": f"scenario_bill_{group:02d}",
                "email_hash": f"scenario_email_{group:02d}",
                "device_id": f"scenario_device_{group:02d}",
                "event_id": f"scenario_performance_{event:02d}",
                "card_type": "credit",
                "quantity": quantity,
                "seconds_since_release": float(seconds),
                "card_age_days": 365.0,
                "ticket_limit": 4,
                "performance_type": "concert",
                "performance_date": "2025-12-01",
                "scenario_name": name,
            }
        )

    add("casual_fan", 1, 1, 1, 25, 2, 0)
    add("family_booking", 2, 2, 2, 15, 4, 1)
    for wave in range(3):
        for card_offset in range(3):
            add(
                "cross_site_card_rotation",
                3,
                3 + card_offset,
                3,
                20 + 10 * (wave * 3 + card_offset),
                2,
                (wave + card_offset) % 3,
            )
    for number in range(6):
        add("different_events_fan", 4, 6, 4 + number, 10 + number * 15, 2, number % 3)
    return pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
