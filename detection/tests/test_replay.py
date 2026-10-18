"""Replay validates whole inputs, stops on transport errors, and isolates truth."""

import json
import subprocess
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from datathon_authorization import replay


@pytest.fixture
def requests():
    return [
        {
            "txn_id": f"T{number:010d}",
            "event_ts": f"2026-11-05T12:00:{number:02d}Z",
            "card_token": "tok_replay_test",
            "bin": "900001",
            "card_type": "credit",
            "card_issue_date": "2026-01-01",
            "merchant_id": "M0001",
            "mcc": "7922",
            "product_category": "tickets",
            "event_id": "EVT_0001",
            "quantity": 2,
            "amount": "90.00",
            "currency": "USD",
            "ship_addr_hash": "ship_1",
            "bill_addr_hash": "bill_1",
            "email_hash": "email_1",
            "device_id": "device_1",
            "ip_prefix": "192.0.2.0/24",
            "channel": "web",
        }
        for number in range(1, 4)
    ]


def _write_requests(path, rows, *, array=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows) if array else "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    return path


def _response(payload, *, probability=0.8, flag=True, threshold=0.5):
    return {
        "txn_id": payload["txn_id"],
        "authorization_status": "REVIEW",
        "risk_score": 80,
        "reason_codes": ["same_event_cross_site_buying"],
        "processed_ts": "2026-11-05T13:00:00Z",
        "scalper_probability": probability,
        "flagged_for_review": flag,
        "decision_threshold": threshold,
        "model_result": {"confidence_tier": "limited"},
    }


class _HTTPResponse:
    def __init__(self, value, *, status=200):
        self.status = status
        self.body = value if isinstance(value, bytes) else json.dumps(value).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _http(monkeypatch, handler=None):
    calls = []

    def send(request, *, timeout):
        payload = json.loads(request.data)
        calls.append(
            {
                "url": request.full_url,
                "payload": payload,
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        return (
            handler(payload, len(calls))
            if handler is not None
            else _HTTPResponse(_response(payload))
        )

    monkeypatch.setattr(replay, "urlopen", send)
    return calls


def _truth(path, rows):
    return _write_requests(
        path,
        [
            {"txn_id": txn_id, "is_scalper": label, "actor_id": f"actor-{number}", "tactic": tactic}
            for number, (txn_id, label, tactic) in enumerate(rows)
        ],
    )


def test_importing_replay_sends_no_requests():
    code = (
        "import urllib.request\n"
        "def forbidden(*args, **kwargs): raise AssertionError('HTTP on import')\n"
        "urllib.request.urlopen = forbidden\n"
        "import datathon_authorization.replay\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("array", [False, True])
def test_json_array_and_jsonl_replay_preserve_order_and_save_full_responses(
    tmp_path,
    requests,
    monkeypatch,
    array,
):
    source = _write_requests(tmp_path / "requests.json", requests, array=array)
    destination = tmp_path / "responses.jsonl"
    calls = _http(monkeypatch)
    results = replay.replay_requests(
        source, destination, url="http://example.test/api", timeout=4.5
    )
    saved = [json.loads(line) for line in destination.read_text().splitlines()]
    assert saved == results
    assert [row["txn_id"] for row in saved] == [row["txn_id"] for row in requests]
    assert all(call["url"] == "http://example.test/api/authorize" for call in calls)
    assert all(call["method"] == "POST" and call["timeout"] == 4.5 for call in calls)
    assert all(not {"is_scalper", "actor_id", "tactic"} & set(call["payload"]) for call in calls)
    assert saved[0]["model_result"] == {"confidence_tier": "limited"}


def test_generator_id_optional_fields_and_equal_timestamps_are_supported(tmp_path, requests):
    rows = deepcopy(requests[:2])
    for number, row in enumerate(rows):
        row["txn_id"] = f"T{number:012d}"
        row["event_ts"] = "2026-11-05T12:00:00Z"
        for field in ("product_category", "device_id", "ip_prefix"):
            row.pop(field)
    actual = replay.read_requests(_write_requests(tmp_path / "requests.jsonl", rows))
    assert [row["txn_id"] for row in actual] == [row["txn_id"] for row in rows]


@pytest.mark.parametrize("problem", ["duplicate", "time", "truth", "currency", "missing_id"])
def test_bad_later_input_prevents_every_http_call(tmp_path, requests, monkeypatch, capsys, problem):
    if problem == "duplicate":
        requests[1]["txn_id"] = requests[0]["txn_id"]
    elif problem == "time":
        requests[1]["event_ts"] = "2026-11-05T11:00:00Z"
    elif problem == "truth":
        requests[1]["is_scalper"] = 1
    elif problem == "currency":
        requests[1]["currency"] = "EUR"
    else:
        requests[1].pop("txn_id")
    source = _write_requests(tmp_path / "requests.jsonl", requests)
    destination = tmp_path / "responses.jsonl"
    calls = _http(monkeypatch)
    with pytest.raises(SystemExit) as error:
        replay.main(["--input", str(source), "--output", str(destination)])
    assert error.value.code == 2
    assert not calls
    assert not destination.exists()
    assert "Error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "[1]",
        "not json",
        '[{"txn_id":"a","txn_id":"b"}]',
        '{"amount":NaN}',
        '{"amount":Infinity}',
    ],
)
def test_strict_json_input_errors_prevent_http(tmp_path, monkeypatch, text):
    source = tmp_path / "input.json"
    source.write_text(text)
    calls = _http(monkeypatch)
    with pytest.raises(ValueError):
        replay.replay_requests(source, tmp_path / "responses.jsonl")
    assert calls == []


@pytest.mark.parametrize("status", [201, 409, 422, 500])
def test_http_errors_exit_nonzero_and_stop_before_later_requests(
    tmp_path,
    requests,
    monkeypatch,
    capsys,
    status,
):
    source = _write_requests(tmp_path / "requests.jsonl", requests)
    destination = tmp_path / "responses.jsonl"

    def fail(payload, number):
        if status == 201:
            return _HTTPResponse(_response(payload), status=status)
        raise HTTPError(
            "http://example.test/authorize",
            status,
            "failure",
            {},
            BytesIO(b'{"detail":"request rejected"}'),
        )

    calls = _http(monkeypatch, fail)
    with pytest.raises(SystemExit) as error:
        replay.main(["--input", str(source), "--output", str(destination)])
    assert error.value.code == 2
    assert len(calls) == 1
    assert destination.read_text() == ""
    message = capsys.readouterr().err
    assert f"HTTP {status}" in message
    assert "no later requests were sent" in message
    if status == 409:
        assert "Restart the server" in message and "in-memory session" in message


@pytest.mark.parametrize(
    "failure", [URLError("offline"), TimeoutError("timed out"), OSError("reset")]
)
def test_network_failure_preserves_completed_responses_and_stops(
    tmp_path,
    requests,
    monkeypatch,
    failure,
):
    source = _write_requests(tmp_path / "requests.jsonl", requests)
    destination = tmp_path / "responses.jsonl"

    def send(payload, number):
        if number == 2:
            raise failure
        return _HTTPResponse(_response(payload))

    calls = _http(monkeypatch, send)
    with pytest.raises(replay.ReplayError, match="Stopped after 1 saved response"):
        replay.replay_requests(source, destination)
    assert len(calls) == 2
    assert len(destination.read_text().splitlines()) == 1


@pytest.mark.parametrize("problem", ["json", "txn_id", "probability", "flag", "threshold"])
def test_incompatible_api_response_stops_replay(tmp_path, requests, monkeypatch, problem):
    source = _write_requests(tmp_path / "requests.jsonl", requests)

    def send(payload, number):
        result = _response(payload)
        if problem == "json":
            return _HTTPResponse(b"not json")
        if problem == "txn_id":
            result["txn_id"] = "wrong"
        elif problem == "probability":
            result.pop("scalper_probability")
        elif problem == "flag":
            result["flagged_for_review"] = "yes"
        elif problem == "threshold":
            result["decision_threshold"] = 2
        return _HTTPResponse(result)

    calls = _http(monkeypatch, send)
    with pytest.raises(replay.ReplayError):
        replay.replay_requests(source, tmp_path / "responses.jsonl")
    assert len(calls) == 1


def test_offline_evaluation_joins_exact_ids_and_uses_actual_server_flags(tmp_path, requests):
    truth = _truth(
        tmp_path / "truth.jsonl",
        [
            (requests[2]["txn_id"], 0, "normal"),
            (requests[0]["txn_id"], 1, "shared_address_ring"),
            (requests[1]["txn_id"], 0, "legit_bulk"),
        ],
    )
    # Deliberately disagree with a 0.5 cutoff: confusion counts must use actual
    # server flags, while proper probability metrics must retain probabilities.
    responses = [
        _response(requests[0], probability=0.9, flag=False),
        _response(requests[1], probability=0.2, flag=True),
        _response(requests[2], probability=0.1, flag=False),
    ]
    report = replay.evaluate_responses(
        responses, truth, requests_path=tmp_path / "test/requests.jsonl"
    )
    assert report["overall"]["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 0}
    assert report["overall"]["brier_score"] == pytest.approx(0.02)
    assert report["per_tactic"]["shared_address_ring"]["recall"] == 0
    assert report["per_tactic"]["card_testing"]["recall"] is None
    assert report["legit_bulk"]["false_positive_rate"] == 1
    assert report["source_split"] == "test"
    assert report["offline_only"] is True
    assert "synthetic" in report["interpretation"]


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate"])
def test_offline_evaluation_rejects_nonexact_truth_ids(tmp_path, requests, problem):
    rows = [(row["txn_id"], 0, "normal") for row in requests]
    if problem == "missing":
        rows.pop()
    elif problem == "extra":
        rows.append(("extra", 0, "normal"))
    else:
        rows.append(rows[0])
    truth = _truth(tmp_path / "truth.jsonl", rows)
    with pytest.raises(ValueError, match="match saved responses exactly|Duplicate ground-truth"):
        replay.evaluate_responses([_response(row) for row in requests], truth)


def test_truth_is_read_only_after_response_output_is_complete(
    tmp_path,
    requests,
    monkeypatch,
    capsys,
):
    import scalper_detector.authorizations as authorizations

    source = _write_requests(tmp_path / "test/requests.jsonl", requests)
    truth = _truth(tmp_path / "truth.jsonl", [(row["txn_id"], 0, "normal") for row in requests])
    destination, evaluation = tmp_path / "responses.jsonl", tmp_path / "evaluation.json"
    calls = _http(monkeypatch)
    original = authorizations.read_ground_truth

    def offline_only(path):
        assert len(calls) == len(requests)
        assert len(destination.read_text().splitlines()) == len(requests)
        return original(path)

    monkeypatch.setattr(authorizations, "read_ground_truth", offline_only)
    replay.main(
        [
            "--input",
            str(source),
            "--output",
            str(destination),
            "--ground-truth",
            str(truth),
            "--evaluation-output",
            str(evaluation),
        ]
    )
    assert json.loads(evaluation.read_text())["overall"]["n_samples"] == len(requests)
    assert all(not {"is_scalper", "actor_id", "tactic"} & set(call["payload"]) for call in calls)
    assert "offline evaluation" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--ground-truth", "--evaluation-output"])
def test_offline_evaluation_arguments_must_be_paired(monkeypatch, flag):
    calls = _http(monkeypatch)
    with pytest.raises(SystemExit) as error:
        replay.main([flag, "unused"])
    assert error.value.code == 2
    assert not calls


def test_outputs_cannot_overwrite_input_or_ground_truth(tmp_path, requests, monkeypatch):
    source = _write_requests(tmp_path / "requests.jsonl", requests)
    before = source.read_bytes()
    calls = _http(monkeypatch)
    with pytest.raises(SystemExit) as error:
        replay.main(["--input", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert not calls
    assert source.read_bytes() == before
