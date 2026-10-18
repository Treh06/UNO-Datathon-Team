"""Merchant advice, observed evidence, and portable neural export checks."""

import json
from types import SimpleNamespace

import pytest
import torch

from scalper_detector.decision import MerchantPolicy, decision_fields
from scalper_detector.deployment import _PaddedExportLSTM, export_onnx
from scalper_detector.model import ScalperLSTM
from scalper_detector.runtime import safe_mode_enabled

NATIVE_LSTM = pytest.param(
    "native",
    marks=pytest.mark.skipif(
        safe_mode_enabled(), reason="Native recurrent kernels are disabled in compatibility mode"
    ),
)


def _complete_evidence():
    return {
        "history_count": 10,
        "card_orders_1m": 1,
        "card_orders_5m": 2,
        "card_orders_1h": 10,
        "bin_orders_1m": 5,
        "bin_orders_5m": 10,
        "cluster_orders_1m": 1,
        "cluster_orders_5m": 2,
        "cluster_orders_1h": 10,
        "card_merchants_5m": 1,
        "cluster_merchants_5m": 1,
        "cluster_cards_1h": 1,
        "linked_cards_count": 1,
        "shipping_linked_cards": 1,
        "email_linked_cards": 1,
        "device_linked_cards": 1,
        "seconds_since_previous": 100,
        "interval_regularity": 0.3,
        "basket_uniformity_5m": 0.4,
        "event_id": "event_123",
        "event_id_known": 1,
        "quantity": 2,
        "quantity_available": 1,
        "ticket_limit": 4,
        "ticket_limit_known": 1,
        "card_event_orders_5m": 1,
        "cluster_event_orders_5m": 2,
        "cluster_event_merchants_5m": 1,
        "cluster_event_cards_5m": 1,
        "card_event_tickets_1h": 2,
        "cluster_event_tickets_1h": 2,
        "cluster_event_tickets_24h": 2,
        "cluster_event_quantity_coverage_1h": 1.0,
        "cluster_event_identity_coverage_5m": 1.0,
    }


@pytest.mark.parametrize(
    "probability,tier,action",
    [
        (0.1, "low", "allow"),
        (0.35, "medium", "step_up_verification"),
        (0.65, "high", "quantity_cap_or_throttle"),
        (0.85, "critical", "cancel_or_refund"),
    ],
)
def test_merchant_policy_boundaries_are_advisory(probability, tier, action):
    result = decision_fields(probability, probability, probability, {})
    assert result["risk_tier"] == tier
    assert result["recommended_action"] == action
    assert result["risk_score"] == round(100 * probability, 2)
    assert result["merchant_decision_required"] is True
    assert result["policy_allow_below"] == 0.35
    assert result["policy_step_up_below"] == 0.65
    assert result["policy_cap_below"] == 0.85
    json.dumps(result, allow_nan=False)


def test_custom_merchant_policy_controls_recommendation():
    policy = MerchantPolicy(allow_below=0.8, step_up_below=0.9, cap_below=0.95)
    assert MerchantPolicy.from_dict(policy.to_dict()) == policy
    assert decision_fields(0.7, 0.7, 0.7, {})["recommended_action"] == "quantity_cap_or_throttle"
    overridden = decision_fields(0.7, 0.7, 0.7, {}, policy)
    assert overridden["recommended_action"] == "allow"
    assert overridden["risk_tier"] == "low"
    assert overridden["risk_score"] == 70.0
    assert overridden["merchant_decision_required"] is True
    assert overridden["policy_allow_below"] == 0.8
    assert overridden["policy_step_up_below"] == 0.9
    assert overridden["policy_cap_below"] == 0.95


@pytest.mark.parametrize(
    "values",
    [
        {"allow_below": -0.1},
        {"cap_below": 1.1},
        {"step_up_below": 0.2},
        {"allow_below": 0.65},
        {"allow_below": float("nan")},
        {"allow_below": True},
        {"allow_below": "0.3"},
        {"unknown_setting": 0.2},
    ],
)
def test_invalid_policy_rejected(values):
    with pytest.raises(ValueError):
        MerchantPolicy.from_dict(values)


def test_observed_reasons_preserve_actual_counts_and_have_a_cap():
    features = _complete_evidence()
    features.update(
        shipping_linked_cards=12,
        cluster_orders_5m=14,
        cluster_cards_1h=6,
        card_orders_1m=5,
        interval_regularity=0.98,
        basket_uniformity_5m=0.99,
    )
    result = decision_fields(0.9, 0.9, 0.9, features)
    assert len(result["reason_codes"]) == len(result["reason_details"]) == 3
    reasons = dict(zip(result["reason_codes"], result["reason_details"]))
    assert reasons["shared_shipping_many_cards"] == "12 cards linked to this shipping token"
    assert "14 linked-cluster ticket orders" in reasons["linked_cluster_order_burst"]
    assert reasons["rapid_card_orders"] == "5 ticket orders on this card in one minute"
    assert decision_fields(0.1, 0.1, 0.1, features)["reason_codes"] == []


def test_uniformity_reasons_describe_payment_amounts_and_optional_inputs_need_flags():
    result = decision_fields(
        0.7,
        0.7,
        0.7,
        {
            "cluster_orders_5m": 5,
            "basket_uniformity_5m": 0.99,
            "quantity": 10,
            "quantity_available": 0,
            "seconds_since_release": 1,
            "release_time_known": 0,
        },
    )
    assert result["reason_codes"] == ["uniform_payment_amounts"]
    assert "payment amounts" in result["reason_details"][0]


def test_confidence_is_distinct_from_risk_and_reflects_evidence_and_agreement():
    evidence = _complete_evidence()
    assert decision_fields(0.95, 0.95, 0.9, evidence)["confidence_tier"] == "high"
    assert decision_fields(0.1, 0.1, 0.12, evidence)["confidence_tier"] == "high"
    assert decision_fields(0.95, 0.95, 0.95, {})["confidence_tier"] == "limited"
    assert decision_fields(0.8, 0.95, 0.05, evidence)["confidence_tier"] == "low"
    sparse_tokens = dict(
        evidence, shipping_linked_cards=0, email_linked_cards=0, device_linked_cards=0
    )
    assert decision_fields(0.95, 0.95, 0.95, sparse_tokens)["confidence_tier"] == "moderate"
    unknown_event = dict(evidence, event_id="", event_id_known=0)
    assert decision_fields(0.95, 0.95, 0.95, unknown_event)["confidence_tier"] == "limited"
    missing_ticket_details = dict(evidence, quantity_available=0, ticket_limit_known=0)
    assert (
        decision_fields(0.95, 0.95, 0.95, missing_ticket_details)["confidence_tier"] == "moderate"
    )
    evidence["history_count"] = 2
    assert decision_fields(0.95, 0.95, 0.95, evidence)["confidence_tier"] == "limited"


def test_family_four_tickets_at_supplied_limit_has_no_abuse_reason():
    evidence = _complete_evidence()
    evidence.update(
        quantity=4,
        ticket_limit=4,
        card_event_orders_5m=1,
        cluster_event_orders_5m=1,
        card_event_tickets_1h=4,
        cluster_event_tickets_1h=4,
        cluster_event_tickets_24h=4,
    )
    result = decision_fields(0.8, 0.8, 0.8, evidence)
    assert result["reason_codes"] == []
    assert result["reason_details"] == []


@pytest.mark.parametrize("coverage", [None, float("nan"), 0.0, 0.2, 0.79])
def test_high_confidence_requires_known_event_identity_coverage(coverage):
    evidence = _complete_evidence()
    evidence["cluster_event_identity_coverage_5m"] = coverage
    result = decision_fields(0.95, 0.95, 0.95, evidence)
    assert result["confidence_tier"] == "moderate"


def test_same_event_ticket_reasons_use_actual_site_card_and_order_counts():
    evidence = _complete_evidence()
    evidence.update(
        cluster_event_orders_5m=6,
        cluster_event_merchants_5m=2,
        cluster_event_cards_5m=3,
        ticket_limit_known=0,
    )
    result = decision_fields(0.9, 0.9, 0.9, evidence)
    assert result["reason_codes"] == ["same_event_cross_site_buying", "linked_cards_same_event"]
    assert (
        result["reason_details"][0]
        == "6 linked-card ticket orders for event event_123 across 2 sites in five minutes"
    )
    assert (
        result["reason_details"][1]
        == "3 linked cards placed 6 ticket orders for event event_123 in five minutes"
    )


def test_known_event_limit_reason_is_observation_not_verified_buyer_violation():
    evidence = _complete_evidence()
    evidence.update(
        cluster_event_tickets_24h=12,
        ticket_limit=4,
        cluster_event_orders_5m=6,
        cluster_event_merchants_5m=2,
        cluster_event_cards_5m=3,
        shipping_linked_cards=12,
    )
    result = decision_fields(0.9, 0.9, 0.9, evidence)
    assert result["reason_codes"] == [
        "event_ticket_limit_exceeded",
        "same_event_cross_site_buying",
        "linked_cards_same_event",
    ]
    detail = result["reason_details"][0]
    assert "12 tickets for event event_123" in detail
    assert "supplied per-buyer event limit is 4" in detail
    assert "do not verify one buyer" in detail


@pytest.mark.parametrize(
    "patch", [{"ticket_limit_known": 0}, {"cluster_event_quantity_coverage_1h": 0}]
)
def test_limit_reason_requires_known_limit_and_observed_ticket_quantities(patch):
    evidence = _complete_evidence()
    evidence.update(cluster_event_tickets_24h=12, ticket_limit=4)
    evidence.update(patch)
    assert (
        "event_ticket_limit_exceeded"
        not in decision_fields(0.9, 0.9, 0.9, evidence)["reason_codes"]
    )


@pytest.mark.parametrize("event,known", [("", 0), ("", 1), (None, 1), ("event_123", 0)])
def test_missing_or_unsupported_event_cannot_create_same_event_evidence(event, known):
    evidence = _complete_evidence()
    evidence.update(
        event_id=event,
        event_id_known=known,
        cluster_event_orders_5m=10,
        cluster_event_merchants_5m=5,
        cluster_event_cards_5m=5,
        cluster_event_tickets_24h=20,
        card_event_orders_5m=4,
    )
    assert decision_fields(0.9, 0.9, 0.9, evidence)["reason_codes"] == []


def test_repeated_same_event_card_orders_have_ticket_specific_reason():
    evidence = _complete_evidence()
    evidence.update(card_event_orders_5m=3)
    result = decision_fields(0.8, 0.8, 0.8, evidence)
    assert result["reason_codes"] == ["rapid_ticket_repeats"]
    assert "3 ticket orders for event event_123 on this card" in result["reason_details"][0]


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_decision_probability_rejected(probability):
    with pytest.raises(ValueError):
        decision_fields(probability, 0.5, 0.5, {})


@pytest.mark.parametrize("lstm_backend", [NATIVE_LSTM, "unfused"])
def test_padded_export_wrapper_matches_packed_model_for_short_histories(lstm_backend):
    torch.manual_seed(7)
    model = ScalperLSTM(4, hidden_size=6, num_layers=2, lstm_backend=lstm_backend).eval()
    wrapper = _PaddedExportLSTM(model).eval()
    features = torch.randn(4, 7, 4)
    lengths = torch.tensor([1, 7, 3, 6])
    with torch.inference_mode():
        expected = model(features, lengths)
        actual = wrapper(features, lengths)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("lstm_backend", [NATIVE_LSTM, "unfused"])
def test_onnx_export_has_numeric_parity_and_target_manifest(tmp_path, monkeypatch, lstm_backend):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from scalper_detector.pipeline import ScalperDetector

    model = ScalperLSTM(4, hidden_size=6, num_layers=2, lstm_backend=lstm_backend).eval()
    detector = SimpleNamespace(
        model=model,
        metadata={
            "format_version": 2,
            "training_config": {"sequence_length": 7},
            "model_config": {"input_size": 4},
        },
    )
    monkeypatch.setattr(ScalperDetector, "load", classmethod(lambda cls, _: detector))
    destination = tmp_path / "neural.onnx"
    manifest = export_onnx(tmp_path, destination)
    onnx.checker.check_model(onnx.load(destination))
    assert manifest["exported_branch"] == "neural_lstm_only"
    assert manifest["lstm_backend"] == lstm_backend
    assert manifest["onnx_checker_passed"] is True
    assert manifest["cpu_numerical_parity"]["verified"] is True
    assert len(manifest["cpu_numerical_parity"]["cases"]) == 3
    assert {case["batch_size"] for case in manifest["cpu_numerical_parity"]["cases"]} == {1, 2, 4}
    assert manifest["hardware_validated"] is False
    assert "IBM zDLC" in manifest["target_validation_required"]
    assert "tree branch" in manifest["remains_on_cpu"]
    assert json.loads(destination.with_suffix(".manifest.json").read_text()) == manifest
