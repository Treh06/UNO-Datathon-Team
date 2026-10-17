"""Fast deterministic checks; run with python -m unittest live_demo.test_live_demo."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from datathon_authorization.replay import ReplayError
from live_demo.__main__ import preflight, run_demo
from live_demo.traffic import TACTICS, DemoConfig, Traffic


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


def response(payload):
    return {
        "txn_id": payload["txn_id"],
        "authorization_status": "APPROVE",
        "risk_score": 10,
        "scalper_probability": 0.1,
        "flagged_for_review": False,
        "decision_threshold": 0.5,
        "reason_codes": [],
    }


class DemoTests(unittest.TestCase):
    def test_repeatable_and_label_free_contract(self):
        config = DemoConfig(duration=200, rate=10)
        start = datetime(2026, 10, 17, tzinfo=timezone.utc)
        a, b = [Traffic(config, start=start, run_id="123") for _ in range(2)]
        arrivals = list(a.arrivals())
        self.assertEqual(arrivals, list(b.arrivals()))
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0 <= x < config.duration for x in arrivals))
        seen, tactics = set(), set()
        for elapsed in arrivals:
            request, truth = a.purchase(elapsed)
            self.assertEqual((request, truth), b.purchase(elapsed))
            self.assertFalse(set(request) & {"actor_id", "tactic", "is_scalper"})
            self.assertNotIn(request["txn_id"], seen)
            seen.add(request["txn_id"])
            self.assertRegex(request["txn_id"], r"^T\d+$")
            self.assertRegex(request["event_ts"], r"\.\d{3}Z$")
            self.assertEqual(request["mcc"], "7922")
            self.assertGreater(request["amount"], 0)
            self.assertLessEqual(request["card_issue_date"], request["event_ts"][:10])
            tactics.add(truth["tactic"])
        self.assertEqual(tactics, set(TACTICS) | {"normal", "legit_bulk"})

    def test_burst_rate_and_campaigns(self):
        config = DemoConfig(
            duration=300, rate=30, scalper_ratio=1, drop_after=100, drop_duration=100
        )
        traffic = Traffic(config, run_id="4")
        arrivals = list(traffic.arrivals())
        normal = sum(x < 100 for x in arrivals)
        burst = sum(100 <= x < 200 for x in arrivals)
        self.assertGreater(burst, normal * 4)
        groups = {}
        for due in arrivals[:2000]:
            req, truth = traffic.purchase(due)
            groups.setdefault((truth["tactic"], truth["actor_id"]), []).append(req)
        for tactic in TACTICS:
            orders = next(
                rows for (name, _), rows in groups.items() if name == tactic and len(rows) >= 18
            )
            if tactic == "shared_address_ring":
                self.assertEqual(len({r["ship_addr_hash"] for r in orders}), 1)
                self.assertGreater(len({r["card_token"] for r in orders}), 5)
            elif tactic == "card_testing":
                self.assertTrue(all(r["amount"] < 3 for r in orders[:12]))
                self.assertTrue(all(r["amount"] > 3 for r in orders[12:]))
            elif tactic == "cross_merchant_sweep":
                self.assertEqual(len({r["card_token"] for r in orders}), 1)
                self.assertEqual(len({r["merchant_id"] for r in orders}), config.merchants)

    def test_pacing_logging_and_separate_truth(self):
        clock = Clock()
        sent = []

        def post(endpoint, payload, timeout):
            self.assertEqual(endpoint, "http://localhost:8000/authorize")
            sent.append((clock(), payload))
            clock.sleep(0.001)
            return response(payload)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            summary = run_demo(
                DemoConfig(duration=2, rate=20),
                output,
                url="http://localhost:8000",
                post=post,
                clock=clock,
                sleep=clock.sleep,
                run_id="1",
                emit=lambda _: None,
            )
            timings = [
                json.loads(line) for line in (output / "timings.jsonl").read_text().splitlines()
            ]
            self.assertGreater(len(sent), 10)
            self.assertTrue(
                all(
                    sent_at >= timing["scheduled_seconds"]
                    for (sent_at, _), timing in zip(sent, timings)
                )
            )
            self.assertEqual(summary["responses"], len(sent))
            self.assertEqual(summary["status"], "completed")
            self.assertAlmostEqual(clock(), 2)
            self.assertEqual(len(timings), len(sent))
            with self.assertRaises(FileExistsError):
                run_demo(DemoConfig(), output, post=post)

    def test_network_failure_not_retried_and_partial_logs_survive(self):
        clock = Clock()
        calls = []

        def post(endpoint, payload, timeout):
            calls.append(payload)
            if len(calls) == 2:
                raise ReplayError("timeout")
            return response(payload)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaises(ReplayError):
                run_demo(
                    DemoConfig(duration=2, rate=20),
                    output,
                    post=post,
                    clock=clock,
                    sleep=clock.sleep,
                    emit=lambda _: None,
                )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                (summary["status"], summary["attempted"], summary["responses"]), ("failed", 2, 1)
            )
            self.assertEqual(len(calls), 2)

    def test_slow_server_reports_capacity_limit(self):
        clock = Clock()

        def post(endpoint, payload, timeout):
            clock.sleep(1)
            return response(payload)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(
                DemoConfig(duration=2, rate=100),
                Path(tmp) / "run",
                post=post,
                clock=clock,
                sleep=clock.sleep,
                emit=lambda _: None,
            )
            self.assertEqual(result["status"], "capacity_limited")
            self.assertGreater(result["schedule_lag_ms"]["p99"], 500)

    def test_preflight_rejects_unready_or_used_server(self):
        def get(url, timeout):
            return BytesIO(
                json.dumps(
                    {"ready": True} if url.endswith("health") else {"requests_processed": 4}
                ).encode()
            )

        with patch("live_demo.__main__.urlopen", get), self.assertRaises(ReplayError):
            preflight("http://localhost:8000", 1)

    def test_interrupt_saves_partial_summary(self):
        clock = Clock()

        def post(endpoint, payload, timeout):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = run_demo(
                DemoConfig(duration=2, rate=20),
                output,
                post=post,
                clock=clock,
                sleep=clock.sleep,
                emit=lambda _: None,
            )
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["attempted"], 1)
            self.assertEqual(result["responses"], 0)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["status"], "interrupted"
            )

    def test_malformed_response_is_failure(self):
        clock = Clock()

        def post(endpoint, payload, timeout):
            return {"txn_id": payload["txn_id"], "authorization_status": "APPROVE"}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaises(ReplayError):
                run_demo(
                    DemoConfig(duration=2, rate=20),
                    output,
                    post=post,
                    clock=clock,
                    sleep=clock.sleep,
                    emit=lambda _: None,
                )
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "failed")

    def test_invalid_settings(self):
        for setting in (
            {"rate": 0},
            {"duration": float("nan")},
            {"scalper_ratio": 1.1},
            {"merchants": 1},
        ):
            with self.assertRaises(ValueError):
                DemoConfig(**setting).validate()


if __name__ == "__main__":
    unittest.main()
