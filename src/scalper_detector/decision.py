"""Merchant-controlled recommendations and observed ticket-order evidence.

Reasons describe observed inputs; they are not causal model attributions.
Confidence is a heuristic evidence-quality tier based on history, feature
coverage, and branch agreement, not statistical or calibrated certainty.
No recommendation in this module executes a payment or merchant action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real


@dataclass
class MerchantPolicy:
    """Probability cutoffs configured by the merchant for advisory actions."""

    allow_below: float = 0.35
    step_up_below: float = 0.65
    cap_below: float = 0.85

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = (self.allow_below, self.step_up_below, self.cap_below)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in values):
            raise ValueError("Merchant policy cutoffs must be finite numeric probabilities.")
        if not 0 <= self.allow_below < self.step_up_below < self.cap_below <= 1:
            raise ValueError("Merchant policy requires 0 <= allow_below < step_up_below < cap_below <= 1.")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict) -> "MerchantPolicy":
        if not isinstance(values, dict):
            raise ValueError("Merchant policy must be a JSON object.")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Unknown merchant policy fields: {sorted(set(values) - set(cls.__dataclass_fields__))}") from exc


def _number(features: dict, key: str) -> float | None:
    value = features.get(key)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, Real) or not math.isfinite(value):
        return None
    return float(value)


def _count(features: dict, key: str) -> int:
    value = _number(features, key)
    return int(value) if value is not None and value >= 0 and value.is_integer() else 0


def _event_identity(features: dict) -> str | None:
    value = features.get("event_id")
    if _number(features, "event_id_known") != 1 or not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _observed_reasons(features: dict) -> tuple[list[str], list[str]]:
    candidates: list[tuple[int, str, str]] = []

    def add(priority: int, code: str, detail: str) -> None:
        candidates.append((priority, code, detail))

    event = _event_identity(features)
    if event is not None:
        event_orders = _count(features, "cluster_event_orders_5m")
        event_sites = _count(features, "cluster_event_merchants_5m")
        event_cards = _count(features, "cluster_event_cards_5m")
        tickets = _number(features, "cluster_event_tickets_24h")
        limit = _number(features, "ticket_limit")
        quantity_coverage = _number(features, "cluster_event_quantity_coverage_1h")
        if (tickets is not None and limit is not None and tickets > limit > 0
                and _number(features, "ticket_limit_known") == 1
                and quantity_coverage is not None and quantity_coverage > 0):
            add(120, "event_ticket_limit_exceeded",
                f"Known quantities total {tickets:g} tickets for event {event} across linked cards in 24 hours; "
                f"supplied per-buyer event limit is {limit:g}. Shared tokens do not verify one buyer")
        if event_sites >= 2 and event_orders >= 3:
            add(115, "same_event_cross_site_buying",
                f"{event_orders} linked-card ticket orders for event {event} across {event_sites} sites in five minutes")
        if event_cards >= 3 and event_orders >= 5:
            add(110, "linked_cards_same_event",
                f"{event_cards} linked cards placed {event_orders} ticket orders for event {event} in five minutes")
        card_event_orders = _count(features, "card_event_orders_5m")
        if card_event_orders >= 3:
            add(105, "rapid_ticket_repeats",
                f"{card_event_orders} ticket orders for event {event} on this card in five minutes")

    for priority, feature, code, noun in (
        (100, "shipping_linked_cards", "shared_shipping_many_cards", "shipping"),
        (95, "device_linked_cards", "shared_device_many_cards", "device"),
        (90, "email_linked_cards", "shared_email_many_cards", "email"),
    ):
        count = _count(features, feature)
        if count >= 3:
            add(priority, code, f"{count} cards linked to this {noun} token")

    cluster_orders = _count(features, "cluster_orders_5m")
    cluster_cards = _count(features, "cluster_cards_1h")
    if cluster_orders >= 8 and cluster_cards >= 3:
        add(98, "linked_cluster_order_burst",
            f"{cluster_orders} linked-cluster ticket orders in five minutes; {cluster_cards} cluster cards observed in one hour")
    card_orders = _count(features, "card_orders_1m")
    if card_orders >= 3:
        add(94, "rapid_card_orders", f"{card_orders} ticket orders on this card in one minute")
    merchants = _count(features, "cluster_merchants_5m")
    if merchants >= 3 and cluster_orders >= 5:
        add(92, "linked_cluster_cross_merchant_burst",
            f"{merchants} sites received {cluster_orders} linked-cluster ticket orders in five minutes")
    card_merchants = _count(features, "card_merchants_5m")
    if card_merchants >= 3:
        add(88, "card_cross_merchant_velocity", f"This card placed ticket orders at {card_merchants} sites in five minutes")
    bin_orders = _count(features, "bin_orders_1m")
    if bin_orders >= 20:
        add(60, "bin_order_burst", f"{bin_orders} ticket orders observed for this BIN in one minute")
    linked_cards = _count(features, "linked_cards_count")
    if linked_cards >= 5 and not any(code.startswith("shared_") for _, code, _ in candidates):
        add(85, "multiple_linked_cards", f"{linked_cards} cards linked through observed shared tokens")

    regularity = _number(features, "interval_regularity")
    if regularity is not None and regularity >= 0.9 and cluster_orders >= 5:
        add(75, "regular_payment_intervals",
            f"Regular inter-order intervals across {cluster_orders} linked-cluster orders in five minutes")
    uniformity = _number(features, "basket_uniformity_5m")
    if uniformity is not None and uniformity >= 0.95 and cluster_orders >= 5:
        add(70, "uniform_payment_amounts",
            f"Similar payment amounts across {cluster_orders} linked-cluster orders in five minutes")
    previous = _number(features, "seconds_since_previous")
    if previous is not None and 0 <= previous <= 3 and _count(features, "card_orders_1h") >= 3:
        add(80, "short_card_repeat_interval", f"{previous:g} seconds since the previous order on this card")
    release_delay = _number(features, "seconds_since_release")
    if (_number(features, "release_time_known") == 1 and release_delay is not None
            and 0 <= release_delay <= 60 and cluster_orders >= 5):
        add(78, "linked_cluster_release_rush",
            f"Order arrived {release_delay:g} seconds after release; {cluster_orders} linked-cluster orders in five minutes")
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:3]
    return [item[1] for item in selected], [item[2] for item in selected]


_COVERAGE_KEYS = (
    "card_orders_1m", "card_orders_5m", "card_orders_1h",
    "bin_orders_1m", "bin_orders_5m", "cluster_orders_1m",
    "cluster_orders_5m", "cluster_orders_1h", "card_merchants_5m",
    "cluster_merchants_5m", "cluster_cards_1h", "linked_cards_count",
    "shipping_linked_cards", "email_linked_cards", "device_linked_cards",
    "seconds_since_previous", "interval_regularity", "basket_uniformity_5m",
)


def _confidence(features: dict, neural_probability: float, tree_probability: float) -> str:
    history = _count(features, "history_count") if "history_count" in features else _count(features, "card_orders_1h")
    # A zero linkage count means the corresponding token is absent, rather
    # than evidence of a fully observed single-card address/device/email.
    linkage_keys = {"shipping_linked_cards", "email_linked_cards", "device_linked_cards"}
    covered = sum(
        _number(features, key) is not None and (key not in linkage_keys or _count(features, key) > 0)
        for key in _COVERAGE_KEYS
    )
    optional_flags = ("quantity_available", "card_age_known", "release_time_known")
    covered += sum(_number(features, key) == 1 for key in optional_flags)
    coverage = covered / (len(_COVERAGE_KEYS) + len(optional_flags))
    disagreement = abs(neural_probability - tree_probability)
    if history < 3 or coverage < 0.35 or _event_identity(features) is None:
        return "limited"
    if history < 7 or coverage < 0.6 or disagreement > 0.3:
        return "low"
    event_keys = ("card_event_orders_5m", "cluster_event_orders_5m", "cluster_event_merchants_5m",
                  "cluster_event_cards_5m", "card_event_tickets_1h", "cluster_event_tickets_1h",
                  "cluster_event_tickets_24h")
    event_coverage = sum(_number(features, key) is not None for key in event_keys) / len(event_keys)
    quantity_coverage = _number(features, "cluster_event_quantity_coverage_1h")
    identity_coverage = _number(features, "cluster_event_identity_coverage_5m")
    known_limit = _number(features, "ticket_limit")
    ticket_evidence_complete = (
        _number(features, "quantity_available") == 1
        and _number(features, "ticket_limit_known") == 1
        and known_limit is not None and known_limit > 0
        and quantity_coverage is not None and quantity_coverage >= 0.8
        and identity_coverage is not None and 0.8 <= identity_coverage <= 1
        and event_coverage >= 0.8
    )
    if coverage >= 0.8 and disagreement <= 0.1 and ticket_evidence_complete:
        return "high"
    return "moderate"


def decision_fields(
    probability: float,
    neural_probability: float,
    tree_probability: float,
    features: dict,
    policy: MerchantPolicy | None = None,
) -> dict:
    """Return advisory risk and observed evidence; the merchant makes the decision.

    Confidence describes heuristic evidence quality, independently of risk.
    It must not be interpreted as a calibrated probability of a correct decision.
    """
    probabilities = (probability, neural_probability, tree_probability)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Model probabilities must be finite numbers in [0, 1].")
    if not isinstance(features, dict):
        raise ValueError("features must contain the observed raw feature dictionary.")
    policy = MerchantPolicy() if policy is None else policy
    if not isinstance(policy, MerchantPolicy):
        raise ValueError("policy must be a MerchantPolicy instance.")
    policy.validate()
    if probability < policy.allow_below:
        tier, action = "low", "allow"
    elif probability < policy.step_up_below:
        tier, action = "medium", "step_up_verification"
    elif probability < policy.cap_below:
        tier, action = "high", "quantity_cap_or_throttle"
    else:
        tier, action = "critical", "cancel_or_refund"
    codes, details = _observed_reasons(features) if probability >= policy.allow_below else ([], [])
    return {
        "risk_score": round(float(probability) * 100, 2),
        "risk_tier": tier,
        "confidence_tier": _confidence(features, float(neural_probability), float(tree_probability)),
        "reason_codes": codes,
        "reason_details": details,
        "recommended_action": action,
        "policy_allow_below": float(policy.allow_below),
        "policy_step_up_below": float(policy.step_up_below),
        "policy_cap_below": float(policy.cap_below),
        "merchant_decision_required": True,
    }
