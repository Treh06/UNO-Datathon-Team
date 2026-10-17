"""Replay validated ticket requests; optionally evaluate saved responses offline.

Use a fresh, single-worker API server for each independent source split. Its
in-memory history persists across replay calls. Importing this module sends
nothing. Ground truth is never read until all response JSONL has been saved.
"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_SCALPER_TACTICS = (
    "shared_address_ring",
    "max_qty_bulk",
    "card_testing",
    "cross_merchant_sweep",
)


class ReplayError(RuntimeError):
    """A replay failed after zero or more successfully saved responses."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON value {value!r} is not permitted.")


def _decode(text: str) -> object:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def read_requests(path: str | Path) -> list[dict[str, Any]]:
    """Validate every JSON-array/JSONL request before the first network call.

    Preserve input order, including equal timestamps; reject duplicate IDs,
    decreasing event time, ground-truth fields, or incompatible API requests.
    """
    from datathon_authorization.authorization.models import Transaction
    from scalper_detector.authorizations import authorization_to_order

    location = Path(path)
    text = location.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Request file is empty: {location}")
    try:
        if text.lstrip().startswith("["):
            records = _decode(text)
        else:
            records = []
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(_decode(line))
                except ValueError as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{location}: invalid JSON: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError(f"{location}: expected a nonempty JSON array or JSONL request file.")

    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous = None
    for number, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("each request must be a JSON object")
            # Use the API schema as well as the detector adapter, so an invalid
            # later record cannot cause a partially submitted local input file.
            payload = Transaction.model_validate(record).model_dump(mode="json")
            order = authorization_to_order(payload)
            txn_id, timestamp = order["order_id"], order["timestamp"]
            if txn_id in seen:
                raise ValueError(f"duplicate txn_id {txn_id!r}")
            if previous is not None and timestamp < previous:
                raise ValueError("requests must be chronological; event_ts decreased")
            seen.add(txn_id)
            previous = timestamp
            requests.append(payload)
        except ValueError as exc:
            raise ValueError(f"{location}: request {number}: {exc}") from exc
    return requests


def _endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--url must be an HTTP(S) server URL without a query or fragment.")
    base = url.rstrip("/")
    return base if parsed.path.rstrip("/").endswith("/authorize") else base + "/authorize"


def _model_values(response: dict[str, Any]) -> tuple[float, bool, float]:
    """Validate and extract the server's calibrated score and review decision."""
    probability = response.get("scalper_probability")
    flag = response.get("flagged_for_review")
    threshold = response.get("decision_threshold")
    for name, value in (("scalper_probability", probability), ("decision_threshold", threshold)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"API response requires a finite {name} in [0, 1]. Use a model-backed server."
            )
    if not isinstance(flag, bool):
        raise ValueError("API response requires boolean flagged_for_review.")
    return float(probability), flag, float(threshold)


def _post(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Submit one request once and require a matching model-backed response."""
    request = Request(
        endpoint,
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ReplayError(f"HTTP {response.status}: /authorize must return HTTP 200.")
            result = _decode(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        hint = (
            " Restart the server to reset its in-memory session before replaying "
            "duplicate IDs or an independent source split."
            if exc.code == 409
            else ""
        )
        raise ReplayError(f"HTTP {exc.code} for {payload['txn_id']}: {detail}.{hint}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ReplayError(f"Network failure for {payload['txn_id']}: {exc}") from exc
    except (ValueError, UnicodeError) as exc:
        raise ReplayError(f"Invalid JSON response for {payload['txn_id']}: {exc}") from exc
    if not isinstance(result, dict) or result.get("txn_id") != payload["txn_id"]:
        raise ReplayError(f"API response txn_id does not match {payload['txn_id']!r}.")
    try:
        _model_values(result)
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    return result


def replay_requests(
    input_path: str | Path,
    output_path: str | Path,
    *,
    url: str = "http://127.0.0.1:8000",
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Save each successful response; stop immediately on any request failure."""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, Real)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive and finite.")
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Response output must not overwrite the request input.")
    endpoint = _endpoint(url)
    requests = read_requests(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    with destination.open("w", encoding="utf-8") as handle:
        for payload in requests:
            try:
                response = _post(endpoint, payload, float(timeout))
            except ReplayError as exc:
                raise ReplayError(
                    f"{exc} Stopped after {len(saved)} saved response(s) in {destination}; "
                    "no later requests were sent."
                ) from exc
            handle.write(json.dumps(response, allow_nan=False) + "\n")
            handle.flush()
            saved.append(response)
    return saved


def evaluate_responses(
    responses: list[dict[str, Any]],
    ground_truth_path: str | Path,
    *,
    requests_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join explicit offline truth by exact txn_id; use actual server flags."""
    from scalper_detector.authorizations import read_ground_truth
    from scalper_detector.metrics import binary_metrics

    ids = [response.get("txn_id") for response in responses]
    if (
        not ids
        or any(not isinstance(txn_id, str) or not txn_id for txn_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Responses must have nonempty, unique txn_id values.")
    values = [_model_values(response) for response in responses]
    truth = read_ground_truth(ground_truth_path).set_index("txn_id")
    missing, extra = set(ids) - set(truth.index), set(truth.index) - set(ids)
    if missing or extra:
        raise ValueError(
            f"Ground truth must match saved responses exactly: {len(missing)} missing label(s), "
            f"{len(extra)} extra txn_id(s)."
        )
    truth = truth.loc[ids]
    labels = truth["is_scalper"].to_numpy()
    probabilities = [value[0] for value in values]
    flags = [int(value[1]) for value in values]
    overall = binary_metrics(labels, probabilities)
    actual_decisions = binary_metrics(labels, flags)
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
        overall[key] = actual_decisions[key]
    tactics = sorted(set(_SCALPER_TACTICS) | set(truth.loc[truth.is_scalper.eq(1), "tactic"]))
    per_tactic = {}
    for tactic in tactics:
        mask = (truth.tactic.eq(tactic) & truth.is_scalper.eq(1)).to_numpy()
        count = int(mask.sum())
        flagged = sum(flag for flag, selected in zip(flags, mask) if selected)
        per_tactic[tactic] = {
            "orders": count,
            "flagged": flagged,
            "recall": flagged / count if count else None,
        }
    bulk = (truth.tactic.eq("legit_bulk") & truth.is_scalper.eq(0)).to_numpy()
    bulk_count = int(bulk.sum())
    bulk_flagged = sum(flag for flag, selected in zip(flags, bulk) if selected)
    source = Path(requests_path) if requests_path is not None else None
    source_split = (
        source.parent.name
        if source is not None and source.parent.name in {"train", "validation", "test"}
        else None
    )
    return {
        "offline_only": True,
        "requests_source": str(source) if source is not None else None,
        "source_split": source_split,
        "ground_truth_source": str(ground_truth_path),
        "decision_basis": "Actual API flagged_for_review values; probability metrics use scalper_probability",
        "decision_thresholds": sorted({value[2] for value in values}),
        "overall": overall,
        "per_tactic": per_tactic,
        "legit_bulk": {
            "orders": bulk_count,
            "flagged": bulk_flagged,
            "false_positive_rate": bulk_flagged / bulk_count if bulk_count else None,
        },
        "interpretation": "Offline synthetic campaign-label evaluation; these results do not establish real-world ticket-scalper accuracy.",
    }


def main(argv: list[str] | None = None) -> None:
    """Replay a validated request file and optionally evaluate its saved responses."""
    parser = argparse.ArgumentParser(
        description="Replay ticket requests into a model-backed local authorization API.",
        epilog="Start a fresh single-worker server for each independent source split; replay preserves input event time.",
    )
    parser.add_argument(
        "--input", default=str(Path(__file__).parent / "authorization/sample_transactions.json")
    )
    parser.add_argument("--output", default="artifacts/api_responses.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--ground-truth", help="Offline evaluation only, read after all responses are saved"
    )
    parser.add_argument("--evaluation-output")
    args = parser.parse_args(argv)
    if bool(args.ground_truth) != bool(args.evaluation_output):
        parser.error("--ground-truth and --evaluation-output must be supplied together")
    try:
        inputs = {Path(args.input).resolve()}
        if args.ground_truth:
            inputs.add(Path(args.ground_truth).resolve())
        outputs = [Path(args.output).resolve()]
        if args.evaluation_output:
            outputs.append(Path(args.evaluation_output).resolve())
        if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
            raise ValueError(
                "Response/evaluation outputs must be distinct and must not overwrite input or ground truth."
            )
        responses = replay_requests(args.input, args.output, url=args.url, timeout=args.timeout)
        print(f"Saved {len(responses)} API responses to {args.output}")
        if args.ground_truth:
            evaluation = evaluate_responses(responses, args.ground_truth, requests_path=args.input)
            destination = Path(args.evaluation_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                f"Saved offline evaluation to {destination}; synthetic results are not real-world validation."
            )
    except (ValueError, OSError, ReplayError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
