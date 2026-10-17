"""Real-model HTTP integration for generated and original authorization inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from datathon_authorization.authorization.main import create_app
from scalper_detector.authorizations import authorization_to_order, read_authorizations
from scalper_detector.pipeline import ScalperDetector, TrainConfig
from scalper_detector.simulation import train_simulation


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SPEC = importlib.util.spec_from_file_location("http_test_simulator", ROOT / "generation_and_testing/simulator/generate.py")
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    config = json.loads((ROOT / "generation_and_testing/configs/default.json").read_text())
    config.update({
        "seed": 19, "start": "2026-11-05T00:00:00Z", "end": "2026-11-05T06:00:00Z",
        "baseline_orders_per_hour": 120, "customers_per_split": 120, "scalper_ratio": 0.35,
        "split_fractions": {"train": 0.5, "validation": 0.25, "test": 0.25},
        "held_out_tactic": "card_testing",
        "tactic_weights": {"shared_address_ring": 1, "max_qty_bulk": 1,
                           "card_testing": 1, "cross_merchant_sweep": 3},
        "drops": [
            {"timestamp": f"2026-11-05T{clock}Z", "orders": 120, "window_seconds": 60, "event_index": event}
            for clock, event in [("01:00:00", 0), ("03:30:00", 1), ("05:00:00", 2)]
        ],
    })
    directory = tmp_path_factory.mktemp("http_model")
    simulation_dir, model_dir = directory / "simulation", directory / "model"
    GENERATOR.generate(config, simulation_dir)
    train_simulation(simulation_dir, model_dir,
                     TrainConfig(epochs=2, hidden_size=6, num_layers=1, sequence_length=3,
                                 batch_size=256, threads=1, patience=2), verbose=False)
    requests_path = simulation_dir / "test/requests.jsonl"
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    return {"model_dir": model_dir, "detector": ScalperDetector.load(model_dir),
            "requests": requests, "requests_path": requests_path}


@pytest.fixture
def client(trained):
    with TestClient(create_app(model_dir=trained["model_dir"])) as connection:
        yield connection


def model_result(response):
    assert response.status_code == 200, response.text
    body = response.json()
    result = body["model_result"]
    assert body["txn_id"] == result["order_id"]
    assert isinstance(body["risk_score"], int) and 0 <= body["risk_score"] <= 100
    assert abs(body["risk_score"] - result["risk_score"]) <= 0.51
    assert body["reason_codes"] == result["reason_codes"]
    assert body["scalper_probability"] == result["scalper_probability"]
    assert body["confidence_tier"] == result["confidence_tier"]
    assert body["recommended_action"] == result["recommended_action"]
    assert body["flagged_for_review"] == result["flagged_for_review"]
    assert body["decision_threshold"] == result["decision_threshold"]
    expected_status = {"allow": "APPROVE", "step_up_verification": "REVIEW",
                       "quantity_cap_or_throttle": "REVIEW", "cancel_or_refund": "DECLINE"}
    assert body["authorization_status"] == expected_status[result["recommended_action"]]
    assert pd.Timestamp(body["processed_ts"]).tzinfo is not None
    assert {"actor_id", "tactic", "is_scalper"}.isdisjoint(body)
    assert {"actor_id", "tactic", "is_scalper"}.isdisjoint(result)
    return body, result


def assert_matches_offline(result, expected):
    for field in ("scalper_probability", "neural_probability", "tree_probability"):
        assert result[field] == pytest.approx(float(expected[field]), rel=1e-5, abs=2e-6)
    for field in ("order_id", "reason_codes", "reason_details", "events_observed", "events_in_window",
                  "same_event_sites_5m", "observed_event_tickets_24h", "recommended_action",
                  "flagged_for_review", "confidence_tier"):
        assert result[field] == expected[field]


def test_health_reports_loaded_ticket_model_and_fresh_stats(client):
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["ready"] is True and payload["status"] == "ok"
    assert payload["mode"] == "simulation"
    assert payload["single_worker_required"] is True
    assert payload["model"]["task"] == "ticket_scalping"
    assert payload["model"]["format_version"] == 3
    assert client.get("/stats").json() == {
        "requests_processed": 0, "decisions": {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0},
    }


def test_online_demo_matches_trained_detector_offline(trained, client, tmp_path):
    from live_demo.__main__ import run_demo
    from live_demo.traffic import DemoConfig
    from live_demo.test_live_demo import Clock

    clock = Clock()
    posted = []

    def send(_endpoint, payload, _timeout):
        assert {"is_scalper", "actor_id", "tactic"}.isdisjoint(payload)
        posted.append(payload)
        response = client.post("/authorize", json=payload)
        assert response.status_code == 200, response.text
        return response.json()

    output = tmp_path / "online"
    summary = run_demo(DemoConfig(duration=10, rate=5, drop_after=2, drop_duration=4), output,
                       post=send, clock=clock, sleep=clock.sleep, run_id="1234",
                       start=datetime(2026, 11, 9, tzinfo=timezone.utc), emit=lambda _: None)
    assert summary["status"] == "completed"
    assert summary["responses"] > 50
    canonical = read_authorizations(output / "requests.jsonl")
    expected = trained["detector"].predict(canonical).set_index("order_id", drop=False)
    saved = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    assert len(saved) == len(posted) == summary["responses"]
    for response in saved:
        assert_matches_offline(response["model_result"], expected.loc[response["txn_id"]])
    assert client.get("/stats").json()["requests_processed"] == len(posted)


def test_entire_generated_test_stream_matches_offline_without_loading_ground_truth(trained, client):
    canonical = read_authorizations(trained["requests_path"])
    assert {"is_scalper", "actor_id", "tactic"}.isdisjoint(canonical.columns)
    offline = trained["detector"].predict(canonical).set_index("order_id")
    assert len(offline) == len(trained["requests"])
    for request in trained["requests"]:
        body, result = model_result(client.post("/authorize", json=request))
        expected = offline.loc[request["txn_id"]].to_dict()
        expected["order_id"] = request["txn_id"]
        assert_matches_offline(result, expected)
        assert body["authorization_status"] in {"APPROVE", "REVIEW", "DECLINE"}
    stats = client.get("/stats").json()
    assert stats["requests_processed"] == len(offline)
    assert sum(stats["decisions"].values()) == len(offline)


def cross_site_requests(template):
    requests = []
    origin = datetime(2027, 1, 1, tzinfo=timezone.utc)
    for index in range(6):
        request = deepcopy(template)
        request.update({
            "txn_id": f"T{900000000000 + index:012d}",
            "event_ts": (origin + timedelta(seconds=10 * index)).isoformat().replace("+00:00", "Z"),
            "card_token": f"tok_http_ring_{index % 3}", "card_issue_date": "2025-01-01",
            "merchant_id": f"M{index % 3:04d}", "event_id": "EVT_HTTP_RING", "quantity": 2,
            "amount": "100.00", "ship_addr_hash": "addr_http_ring", "bill_addr_hash": "addr_http_ring",
            "email_hash": "em_http_ring", "device_id": "dev_http_ring", "product_category": "tickets",
        })
        requests.append(request)
    return requests


def test_http_uses_cross_site_linked_card_and_current_order_context(trained, client):
    requests = cross_site_requests(trained["requests"][0])
    canonical = pd.DataFrame([authorization_to_order(request) for request in requests])
    offline = trained["detector"].predict(canonical).set_index("order_id")
    results = []
    for request in requests:
        _, result = model_result(client.post("/authorize", json=request))
        expected = offline.loc[request["txn_id"]].to_dict()
        expected["order_id"] = request["txn_id"]
        assert_matches_offline(result, expected)
        results.append(result)
    assert results[0]["observed_event_tickets_24h"] == 2
    assert results[-1]["observed_event_tickets_24h"] == 12
    assert results[-1]["same_event_sites_5m"] == 3
    assert results[-1]["events_observed"] == 2
    assert results[-1]["event_quantity_coverage_1h"] == 1
    assert results[-1]["event_ticket_limit"] is None


def test_original_two_sample_transactions_and_decimal_amount_strings(trained, client):
    path = ROOT / "datathon_authorization/authorization/sample_transactions.json"
    requests = json.loads(path.read_text())
    canonical = pd.DataFrame([authorization_to_order(request) for request in requests])
    offline = trained["detector"].predict(canonical).set_index("order_id")
    for request in requests:
        assert len(request["txn_id"]) == 11 and isinstance(request["amount"], str)
        _, result = model_result(client.post("/authorize", json=request))
        expected = offline.loc[request["txn_id"]].to_dict()
        expected["order_id"] = request["txn_id"]
        assert_matches_offline(result, expected)


@pytest.mark.parametrize("missing", ["omitted", "null"])
def test_missing_performance_id_is_accepted_with_limited_evidence(trained, client, missing):
    request = deepcopy(trained["requests"][0])
    if missing == "omitted":
        request.pop("event_id", None)
    else:
        request["event_id"] = None
    _, result = model_result(client.post("/authorize", json=request))
    assert result["confidence_tier"] == "limited"
    assert result["same_event_sites_5m"] == 0
    assert result["observed_event_tickets_24h"] == 0


@pytest.mark.parametrize("missing", ["omitted", "null"])
def test_missing_card_type_matches_offline_unknown_encoding(trained, client, missing):
    request = deepcopy(trained["requests"][0])
    if missing == "omitted":
        request.pop("card_type", None)
    else:
        request["card_type"] = None
    expected = trained["detector"].predict(pd.DataFrame([authorization_to_order(request)])).iloc[0].to_dict()
    _, result = model_result(client.post("/authorize", json=request))
    assert_matches_offline(result, expected)


def test_timezone_offsets_are_normalized_before_scoring(trained, client):
    request = deepcopy(trained["requests"][0])
    original = authorization_to_order(request)
    instant = datetime.fromisoformat(request["event_ts"].replace("Z", "+00:00"))
    request["event_ts"] = instant.astimezone(timezone(timedelta(hours=1))).isoformat()
    expected = trained["detector"].predict(pd.DataFrame([original])).iloc[0].to_dict()
    _, result = model_result(client.post("/authorize", json=request))
    assert pd.Timestamp(result["timestamp"]) == original["timestamp"]
    assert_matches_offline(result, expected)


@pytest.mark.parametrize("field,value", [
    ("is_scalper", 1), ("actor_id", "offline-only"), ("tactic", "card_testing"),
    ("product_category", "sneakers"), ("product_category", "electronics"),
    ("mcc", "5999"), ("currency", "EUR"), ("quantity", 0), ("quantity", -1),
    ("quantity", True), ("quantity", 1.5), ("quantity", "2"),
    ("amount", True), ("amount", "NaN"), ("amount", "Infinity"), ("amount", "bad"),
    ("amount", {"value": 12}), ("bin", 900001), ("event_ts", "2026-11-05T05:00:00"),
    ("event_ts", "not-a-date"), ("card_issue_date", "2030-01-01"),
])
def test_invalid_input_is_rejected_before_consuming_any_history(trained, client, field, value):
    valid = deepcopy(trained["requests"][0])
    invalid = deepcopy(valid)
    invalid[field] = value
    response = client.post("/authorize", json=invalid)
    assert response.status_code == 422, response.text
    assert client.get("/stats").json()["requests_processed"] == 0
    _, result = model_result(client.post("/authorize", json=valid))
    assert result["events_observed"] == 1


@pytest.mark.parametrize("field,value", [
    ("amount", float("nan")), ("amount", float("inf")), ("amount", float("-inf")),
    ("quantity", float("inf")), ("unexpected", float("nan")),
    ("unexpected", {"value": float("nan"), "items": [float("inf"), float("-inf")]}),
])
def test_raw_nonfinite_json_validation_returns_standard_422_without_consuming_state(trained, client, field, value):
    valid = deepcopy(trained["requests"][0])
    invalid = deepcopy(valid)
    invalid[field] = value
    # Send raw bytes because httpx's json= argument rejects nonfinite numbers
    # before the request reaches the application's validation/error handler.
    response = client.post("/authorize", content=json.dumps(invalid), headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/json")

    def reject_nonfinite_response(value):
        pytest.fail(f"Validation response contains a nonstandard JSON number: {value}")

    body = json.loads(response.text, parse_constant=reject_nonfinite_response)
    assert isinstance(body["detail"], list) and body["detail"]
    assert client.get("/stats").json()["requests_processed"] == 0
    assert client.get("/health").json()["ready"] is True
    _, result = model_result(client.post("/authorize", json=valid))
    assert result["events_observed"] == 1


def test_duplicate_and_late_requests_return_conflict_without_state_contamination(trained, client):
    first, second = deepcopy(trained["requests"][:2])
    model_result(client.post("/authorize", json=first))
    assert client.post("/authorize", json=first).status_code == 409
    late = deepcopy(first)
    late["txn_id"] = "T999999999999"
    late["event_ts"] = (pd.Timestamp(first["event_ts"]).to_pydatetime() - timedelta(seconds=1)).isoformat()
    assert client.post("/authorize", json=late).status_code == 409
    assert client.get("/stats").json()["requests_processed"] == 1
    _, result = model_result(client.post("/authorize", json=second))
    expected = trained["detector"].predict(pd.DataFrame([authorization_to_order(first), authorization_to_order(second)])).iloc[-1].to_dict()
    assert_matches_offline(result, expected)
    assert client.get("/stats").json()["requests_processed"] == 2


def test_app_instances_have_independent_sessions_and_counters(trained):
    request = deepcopy(trained["requests"][0])
    with TestClient(create_app(detector=trained["detector"])) as first:
        with TestClient(create_app(detector=trained["detector"])) as second:
            _, one = model_result(first.post("/authorize", json=request))
            assert second.get("/stats").json()["requests_processed"] == 0
            _, two = model_result(second.post("/authorize", json=request))
            assert one["events_observed"] == two["events_observed"] == 1
            assert one["scalper_probability"] == two["scalper_probability"]
            assert first.get("/stats").json()["requests_processed"] == 1
            assert second.get("/stats").json()["requests_processed"] == 1


def test_concurrent_duplicate_requests_are_atomic(trained, client):
    request = deepcopy(trained["requests"][0])
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: client.post("/authorize", json=request), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert client.get("/stats").json()["requests_processed"] == 1


def test_missing_model_fails_during_startup_without_heuristic_fallback(tmp_path, monkeypatch):
    missing = tmp_path / "no_model"
    monkeypatch.setenv("SCALPER_MODEL_DIR", str(missing))
    with pytest.raises((OSError, RuntimeError, ValueError)):
        with TestClient(create_app()):
            pytest.fail("An API without trained model artifacts must not become ready")


def test_unexpected_scoring_failure_stops_readiness_and_discards_uncommitted_context(trained, client, monkeypatch):
    first, failing, later = deepcopy(trained["requests"][:3])
    model_result(client.post("/authorize", json=first))
    state = client.app.state.authorization_state
    original_score = state.session.score_order
    stats_before = client.get("/stats").json()
    committed_ids = set(state.seen_ids)
    committed_timestamp = state.last_event_ts

    def fail_after_observing(order):
        original_score(order)
        raise RuntimeError("Injected inference/response failure after context update")

    monkeypatch.setattr(state.session, "score_order", fail_after_observing)
    response = client.post("/authorize", json=failing)
    assert response.status_code == 503
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["ready"] is False
    assert client.get("/stats").json() == stats_before
    assert state.seen_ids == committed_ids
    assert failing["txn_id"] not in state.seen_ids
    assert state.last_event_ts == committed_timestamp
    assert state.session is None
    assert client.post("/authorize", json=later).status_code == 503
    assert client.post("/authorize", json=failing).status_code == 503
    assert client.get("/stats").json() == stats_before
