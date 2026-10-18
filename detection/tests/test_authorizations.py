"""Schema conversion and strict separation of requests from simulator truth."""

import importlib.util
import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scalper_detector.authorizations import (
    _records_frame,
    authorization_to_order,
    read_authorizations,
    read_ground_truth,
)
from scalper_detector.data import LABEL, EventPreprocessor


@pytest.fixture
def payload():
    return {
        "txn_id": "000001",
        "event_ts": "2026-11-05T12:00:00Z",
        "card_token": "000123",
        "bin": "001234",
        "merchant_id": "0009",
        "mcc": "7922",
        "event_id": "00042",
        "quantity": 2,
        "amount": 90.5,
        "currency": "USD",
        "ship_addr_hash": "shipping_token",
        "bill_addr_hash": "billing_token",
        "email_hash": "email_token",
        "device_id": "device_token",
        "card_type": "credit",
        "card_issue_date": "2026-11-01",
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _truth(txn_id="000001", label=1):
    return {
        "txn_id": txn_id,
        LABEL: label,
        "actor_id": "actor_offline",
        "tactic": "shared_address_ring",
    }


def test_explicit_field_mapping_preserves_tokens_and_units(payload):
    actual = authorization_to_order(payload)
    assert actual["order_id"] == "000001"
    assert actual["card_id"] == "000123"
    assert actual["card_bin"] == "001234"
    assert actual["merchant_id"] == "0009"
    assert actual["event_id"] == "00042"
    assert actual["shipping_hash"] == payload["ship_addr_hash"]
    assert actual["billing_hash"] == payload["bill_addr_hash"]
    assert actual["email_hash"] == payload["email_hash"]
    assert actual["device_id"] == payload["device_id"]
    assert actual["amount"] == 90.5
    assert actual["quantity"] == 2
    assert actual["timestamp"] == pd.Timestamp("2026-11-05T12:00:00Z")
    assert actual["card_age_days"] == 4.5
    assert actual["card_age_known"] == 1
    assert actual["category"] == "tickets"
    assert actual["ticket_limit_known"] == actual["release_time_known"] == 0


def test_absent_optional_inputs_stay_unknown_and_are_not_invented(payload):
    for key in (
        "card_issue_date",
        "event_id",
        "quantity",
        "ship_addr_hash",
        "bill_addr_hash",
        "email_hash",
        "device_id",
    ):
        payload.pop(key)
    payload.update(
        quantity_cap=8,
        drops=[{"timestamp": "2026-11-05T00:00:00Z"}],
        ticket_limit=8,
        seconds_since_release=12,
        ip_prefix="192.0.2.0/24",
        channel="api",
    )
    actual = authorization_to_order(payload)
    assert actual["quantity_available"] == actual["event_id_known"] == actual["card_age_known"] == 0
    assert actual["release_time_known"] == actual["ticket_limit_known"] == 0
    assert (
        actual["shipping_hash"]
        == actual["billing_hash"]
        == actual["email_hash"]
        == actual["device_id"]
        == ""
    )
    assert "quantity_cap" not in actual
    assert "drops" not in actual
    assert "ip_prefix" not in actual
    assert "channel" not in actual


def test_unknown_source_fields_cannot_shadow_canonical_fields(payload):
    expected = authorization_to_order(payload)
    payload.update(
        card_id="wrong",
        card_bin="wrong",
        order_id="wrong",
        shipping_hash="wrong",
        category="wrong",
        final_spend=999999,
        detection_result={"tactic": "future"},
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame([authorization_to_order(payload)]), pd.DataFrame([expected])
    )


@pytest.mark.parametrize("category", ["sneakers", "cards", "collectibles", "other", None, 1])
def test_explicit_non_ticket_authorization_category_is_rejected(payload, category):
    payload["product_category"] = category
    with pytest.raises(ValueError, match="product_category='tickets'"):
        authorization_to_order(payload)


def test_explicit_ticket_category_preserves_existing_generator_features(payload):
    expected = authorization_to_order(payload)
    payload["product_category"] = "tickets"
    pd.testing.assert_frame_equal(
        pd.DataFrame([authorization_to_order(payload)]), pd.DataFrame([expected])
    )


@pytest.mark.parametrize("amount", ["90.50", Decimal("90.50")])
def test_authorization_decimal_amounts_keep_usd_units(payload, amount):
    payload["amount"] = amount
    assert authorization_to_order(payload)["amount"] == 90.5


@pytest.mark.parametrize(
    "amount", ["NaN", "Infinity", "-Infinity", "0", "-10", "not-money", "1e9999"]
)
def test_invalid_decimal_payment_strings_rejected(payload, amount):
    payload["amount"] = amount
    with pytest.raises(ValueError, match="amount"):
        authorization_to_order(payload)


@pytest.mark.parametrize("field", ["actor_id", "tactic", "is_scalper"])
def test_truth_fields_rejected_in_authorization_payload(payload, field):
    payload[field] = None
    with pytest.raises(ValueError, match="Ground-truth fields are forbidden"):
        authorization_to_order(payload)


@pytest.mark.parametrize("field", ["txn_id", "card_token", "bin", "merchant_id"])
def test_required_identifiers_never_fabricated(payload, field):
    payload.pop(field)
    with pytest.raises(ValueError, match=field):
        authorization_to_order(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("currency", "EUR"),
        ("currency", None),
        ("mcc", "5732"),
        ("mcc", None),
        ("amount", 0),
        ("amount", -1),
        ("amount", float("nan")),
        ("amount", True),
        ("quantity", 0),
        ("quantity", 1.5),
        ("quantity", True),
        ("card_issue_date", "2026-11-06"),
        ("card_issue_date", "2026-02-30"),
        ("card_issue_date", "2026-11-01T00:00:00Z"),
        ("card_issue_date", "20261101"),
        ("event_ts", "2026-11-05T12:00:00"),
        ("event_ts", "not a date"),
    ],
)
def test_invalid_values_rejected(payload, field, value):
    payload[field] = value
    with pytest.raises(ValueError):
        authorization_to_order(payload)


def test_card_age_uses_utc_order_timestamp(payload):
    payload["event_ts"] = "2026-11-05T01:00:00+02:00"
    payload["card_issue_date"] = "2026-11-04"
    actual = authorization_to_order(payload)
    assert actual["timestamp"] == pd.Timestamp("2026-11-04T23:00:00Z")
    assert actual["card_age_days"] == pytest.approx(23 / 24)
    payload["card_issue_date"] = "2026-11-05"
    with pytest.raises(ValueError, match="cannot be after"):
        authorization_to_order(payload)


def test_readers_join_exact_ids_only_on_explicit_request(payload, tmp_path):
    second = dict(payload, txn_id="000002", event_ts="2026-11-05T11:00:00Z")
    requests = _write_jsonl(tmp_path / "requests.jsonl", [payload, second])
    truth_path = _write_jsonl(
        tmp_path / "ground_truth.jsonl", [_truth("000001", 1), _truth("000002", 0)]
    )
    unlabeled = read_authorizations(requests)
    assert LABEL not in unlabeled
    assert unlabeled.order_id.tolist() == ["000002", "000001"]
    labeled = read_authorizations(requests, ground_truth_path=truth_path)
    assert labeled[LABEL].tolist() == [0, 1]
    assert "actor_id" not in labeled and "tactic" not in labeled
    truth = read_ground_truth(truth_path)
    assert truth.columns.tolist() == ["txn_id", "is_scalper", "actor_id", "tactic"]
    assert truth.txn_id.tolist() == ["000001", "000002"]


def test_reader_normalizes_timezone_offsets_before_dataframe_inference(payload, tmp_path):
    first = dict(payload, event_ts="2026-11-05T12:00:00.123456789-01:00")
    second = dict(payload, txn_id="000002", event_ts="2026-11-05T12:30:00Z")
    requests = _write_jsonl(tmp_path / "requests.jsonl", [first, second])

    actual = read_authorizations(requests)

    assert actual.timestamp.tolist() == [
        pd.Timestamp("2026-11-05T12:30:00Z"),
        pd.Timestamp("2026-11-05T13:00:00.123456789Z"),
    ]


def test_record_frame_keeps_timestamp_objects_out_of_datetime_inference():
    timestamp = pd.Timestamp("2026-11-05T12:00:00Z")

    actual = _records_frame([{"timestamp": timestamp}])

    assert actual["timestamp"].dtype == object
    assert actual.iloc[0]["timestamp"] == timestamp


def test_labels_and_offline_metadata_cannot_change_features(payload, tmp_path):
    requests = _write_jsonl(tmp_path / "requests.jsonl", [payload])
    truth_path = _write_jsonl(tmp_path / "truth.jsonl", [_truth()])
    first = read_authorizations(requests, ground_truth_path=truth_path)
    truth = _truth(label=0)
    truth.update(actor_id="different_actor", tactic="different_future_tactic")
    _write_jsonl(truth_path, [truth])
    changed = read_authorizations(requests, ground_truth_path=truth_path)
    pd.testing.assert_frame_equal(first.drop(columns=LABEL), changed.drop(columns=LABEL))
    processor = EventPreprocessor().fit(first)
    np.testing.assert_array_equal(processor.transform(first)[1], processor.transform(changed)[1])


@pytest.mark.parametrize(
    "rows,error",
    [
        ([], "empty"),
        ([_truth(), _truth()], "Duplicate ground-truth"),
        ([_truth("other")], "missing label.*extra txn_id"),
        ([_truth(), _truth("extra")], "extra txn_id"),
    ],
)
def test_ground_truth_must_match_every_request_once(payload, tmp_path, rows, error):
    requests = _write_jsonl(tmp_path / "requests.jsonl", [payload])
    truth = _write_jsonl(tmp_path / "truth.jsonl", rows)
    with pytest.raises(ValueError, match=error):
        read_authorizations(requests, ground_truth_path=truth)


@pytest.mark.parametrize("label", [None, -1, 2, "1", True, 1.0])
def test_invalid_truth_labels_rejected(tmp_path, label):
    path = _write_jsonl(tmp_path / "truth.jsonl", [_truth(label=label)])
    with pytest.raises(ValueError, match="is_scalper"):
        read_ground_truth(path)


@pytest.mark.parametrize("field", ["actor_id", "tactic"])
def test_truth_metadata_required_only_in_separate_truth_file(tmp_path, field):
    row = _truth()
    row.pop(field)
    with pytest.raises(ValueError, match=field):
        read_ground_truth(_write_jsonl(tmp_path / "truth.jsonl", [row]))


def test_blank_lines_allowed_and_equal_time_order_is_stable(payload, tmp_path):
    second = dict(payload, txn_id="000002")
    path = tmp_path / "requests.jsonl"
    path.write_text("\n" + json.dumps(second) + "\n \n" + json.dumps(payload) + "\n")
    assert read_authorizations(path).order_id.tolist() == ["000002", "000001"]


@pytest.mark.parametrize(
    "line,error",
    [
        ("{bad json", "invalid JSON"),
        ("[]", "must be an object"),
        ('{"txn_id":"a","txn_id":"b"}', "Duplicate JSON field"),
        ('{"value":NaN}', "Nonfinite JSON"),
    ],
)
def test_bad_jsonl_reports_path_and_line(tmp_path, line, error):
    path = tmp_path / "requests.jsonl"
    path.write_text("\n" + line + "\n")
    with pytest.raises(ValueError, match=f"requests.jsonl:2:.*{error}"):
        read_authorizations(path)


def test_duplicate_request_ids_rejected_with_line_context(payload, tmp_path):
    path = _write_jsonl(tmp_path / "requests.jsonl", [payload, payload])
    with pytest.raises(ValueError, match="requests.jsonl:2:.*Duplicate request"):
        read_authorizations(path)


def test_actual_generator_output_converts_without_modifying_files(tmp_path):
    source_root = Path(__file__).resolve().parents[2] / "generation_and_testing"
    spec = importlib.util.spec_from_file_location(
        "friend_authorization_generator", source_root / "simulator/generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((source_root / "configs/default.json").read_text())
    config.update(
        end="2026-11-05T03:00:00Z", baseline_orders_per_hour=20, customers_per_split=12, drops=[]
    )
    output = tmp_path / "generated"
    manifest = module.generate(deepcopy(config), output)
    for split in ("train", "validation", "test"):
        requests, truth_path = (
            output / split / "requests.jsonl",
            output / split / "ground_truth.jsonl",
        )
        before = requests.read_bytes(), truth_path.read_bytes()
        canonical = read_authorizations(requests, ground_truth_path=truth_path)
        truth = read_ground_truth(truth_path).set_index("txn_id")
        assert len(canonical) == manifest["splits"][split]["orders"]
        np.testing.assert_array_equal(canonical[LABEL], truth.loc[canonical.order_id, LABEL])
        assert canonical.timestamp.is_monotonic_increasing
        assert canonical.category.eq("tickets").all()
        assert canonical.card_age_known.eq(1).all()
        assert canonical.release_time_known.eq(0).all()
        assert canonical.ticket_limit_known.eq(0).all()
        assert "actor_id" not in canonical and "tactic" not in canonical
        assert (requests.read_bytes(), truth_path.read_bytes()) == before
