"""Paced online demo against the existing, real model-backed HTTP API."""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from datathon_authorization.replay import ReplayError, _endpoint, _model_values, _post

from .traffic import DemoConfig, Traffic


def preflight(url: str, timeout: float) -> None:
    """Require a ready API with an empty chronological scoring session."""
    base = _endpoint(url).removesuffix("/authorize")
    for path in ("/health", "/stats"):
        try:
            with urlopen(base + path, timeout=timeout) as response:
                body = json.load(response)
        except (OSError, URLError, ValueError) as exc:
            raise ReplayError(
                f"Cannot check {path}: {exc}. Start the trained-model API first."
            ) from exc
        if not isinstance(body, dict):
            raise ReplayError(f"Invalid {path} response: expected a JSON object")
        if path == "/health" and body.get("ready") is not True:
            raise ReplayError("Authorization API is not ready; check its trained model.")
        if path == "/stats" and body.get("requests_processed") != 0:
            raise ReplayError(
                "Start a fresh API before the demo: its transaction history is not empty."
            )


def percentile(values: list[float], fraction: float) -> float | None:
    """Return the nearest observed percentile, or None for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _validate_response(response: Any, txn_id: str) -> None:
    """Check fields used by demo logging and metrics before saving a response."""
    if (
        not isinstance(response, dict)
        or response.get("txn_id") != txn_id
        or response.get("authorization_status") not in {"APPROVE", "REVIEW", "DECLINE"}
    ):
        raise ReplayError("Authorization API returned an incompatible response")
    _model_values(response)
    score = response.get("risk_score")
    reasons = response.get("reason_codes")
    if (
        type(score) is not int
        or not 0 <= score <= 100
        or not isinstance(reasons, list)
        or not all(isinstance(code, str) for code in reasons)
    ):
        raise ReplayError("Authorization API returned invalid score or reasons")


def run_demo(
    config: DemoConfig,
    output: str | Path,
    *,
    url: str = "http://127.0.0.1:8000",
    post: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    start: datetime | None = None,
    run_id: str | None = None,
    emit: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """Pace one ordered traffic stream and preserve logs, including partial runs.

    The transport, clock, and output callback can be supplied by callers for
    deterministic runs. Requests are never retried after an ambiguous failure.
    """
    config.validate()
    endpoint = _endpoint(url)
    folder = Path(output)
    # Creating a new run directory is intentional: partial sessions stay inspectable.
    folder.mkdir(parents=True, exist_ok=False)
    traffic = Traffic(config, start=start, run_id=run_id)
    manifest = {
        "config": config.to_dict(),
        "run_id": traffic.run_id,
        "start": traffic.start.isoformat(),
        "url": endpoint,
        "synthetic": True,
        "mode": "online",
        "status": "running",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts: Counter[str] = Counter()
    tactics: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    tactic_flags: Counter[str] = Counter()
    latencies: list[float] = []
    lags: list[float] = []
    began = clock()
    failure = None
    status = "completed"
    post = post or _post
    emit("LIVE synthetic purchases -> trained authorization API (Ctrl+C stops)")
    emit("elapsed phase    truth                    decision score  HTTP ms  lag ms  reasons")
    try:
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context((folder / f"{name}.jsonl").open("x", encoding="utf-8"))
                for name in ("requests", "ground_truth", "responses", "timings")
            }

            def save(name: str, row: dict[str, Any]) -> None:
                handles[name].write(json.dumps(row, allow_nan=False) + "\n")
                handles[name].flush()

            for due in traffic.arrivals():
                # No unbounded producer queue: one ordered request at a time is
                # required by the API's shared causal feature history.
                while clock() - began < due:
                    sleep(max(0.0, min(0.25, due - (clock() - began))))
                if clock() - began >= config.duration:
                    status = "capacity_limited"
                    break
                payload, truth = traffic.purchase(due)
                lag_ms = max(0.0, (clock() - began - due) * 1000)
                save("requests", payload)
                save("ground_truth", truth)
                sent = clock()
                try:
                    response = post(endpoint, payload, config.timeout)
                    _validate_response(response, payload["txn_id"])
                except (ReplayError, OSError, ValueError) as exc:
                    save(
                        "timings",
                        {
                            "txn_id": payload["txn_id"],
                            "error": str(exc),
                            "outcome": "unknown; not retried",
                        },
                    )
                    raise
                elapsed_ms = (clock() - sent) * 1000
                save("responses", response)
                save(
                    "timings",
                    {
                        "txn_id": payload["txn_id"],
                        "scheduled_seconds": due,
                        "http_ms": elapsed_ms,
                        "schedule_lag_ms": lag_ms,
                    },
                )
                counts[response["authorization_status"]] += 1
                tactics[truth["tactic"]] += 1
                flag = bool(response["flagged_for_review"])
                tactic_flags[truth["tactic"]] += int(flag)
                if truth["is_scalper"]:
                    outcome = "tp" if flag else "fn"
                else:
                    outcome = "fp" if flag else "tn"
                outcomes[outcome] += 1
                latencies.append(elapsed_ms)
                lags.append(lag_ms)
                phase = "DROP" if traffic.in_drop(due) else "normal"
                emit(
                    f"{due:6.1f}s {phase:7} {truth['tactic']:24} {response['authorization_status']:8} "
                    f"{response['risk_score']:5} {elapsed_ms:8.1f} {lag_ms:7.1f}  "
                    f"{','.join(response['reason_codes'][:3])}"
                )
            # Honour the scenario's wall duration even if its last arrival was earlier.
            if status == "completed":
                while clock() - began < config.duration:
                    sleep(max(0.0, min(0.25, config.duration - (clock() - began))))
    except KeyboardInterrupt:
        status = "interrupted"
    except (ReplayError, OSError, ValueError) as exc:
        status, failure = "failed", str(exc)
    total = sum(counts.values())
    predicted_positive = outcomes["tp"] + outcomes["fp"]
    actual_positive = outcomes["tp"] + outcomes["fn"]
    per_tactic = {}
    for tactic, count in sorted(tactics.items()):
        metric = "false_positive_rate" if tactic in {"normal", "legit_bulk"} else "recall"
        per_tactic[tactic] = {
            "orders": count,
            "flagged": tactic_flags[tactic],
            metric: tactic_flags[tactic] / count,
        }
    summary = {
        "status": status,
        "error": failure,
        "attempted": traffic.sequence,
        "responses": total,
        "elapsed_seconds": clock() - began,
        "decisions": dict(counts),
        "tactics": dict(tactics),
        "confusion_matrix": {key: outcomes[key] for key in ("tp", "fp", "tn", "fn")},
        "precision": outcomes["tp"] / predicted_positive if predicted_positive else None,
        "recall": outcomes["tp"] / actual_positive if actual_positive else None,
        "per_tactic": per_tactic,
        "http_ms": {f"p{p}": percentile(latencies, p / 100) for p in (50, 95, 99)},
        "schedule_lag_ms": {f"p{p}": percentile(lags, p / 100) for p in (50, 95, 99)},
        "interpretation": (
            "Synthetic campaign labels; metrics cover saved responses only. "
            "HTTP latency includes network and server work."
        ),
    }
    manifest["status"] = status
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit(f"{status}: {total} responses; {dict(counts)}; logs: {folder}")
    if failure:
        raise ReplayError(f"{failure}. Partial logs preserved at {folder}; no automatic retries.")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run a scenario from command-line options and report its completion status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    defaults = DemoConfig()
    for name in defaults.to_dict():
        value = getattr(defaults, name)
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int if type(value) is int else float, default=value
        )
    args = parser.parse_args(argv)
    config = DemoConfig(**{key: getattr(args, key) for key in defaults.to_dict()})
    try:
        config.validate()
        preflight(args.url, config.timeout)
        output = args.output or Path("artifacts/live_demo") / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        summary = run_demo(config, output, url=args.url)
        if summary["status"] in {"interrupted", "capacity_limited"}:
            parser.exit(130 if summary["status"] == "interrupted" else 3)
    except (ValueError, OSError, ReplayError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
