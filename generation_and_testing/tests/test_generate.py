"""Regression checks for synthetic traffic and isolated chronological splits."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from generation_and_testing.simulator.generate import (
    TACTICS,
    Generator,
    generate,
    parse_time,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/default.json").read_text())

    def test_schema_order_counts_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(self.config, tmp)
            seen_ids, actors, cards, addresses = set(), set(), set(), set()
            previous_time = None
            for split, summary in manifest["splits"].items():
                folder = Path(tmp) / split
                requests = read_rows(folder / "requests.jsonl")
                labels = read_rows(folder / "ground_truth.jsonl")
                self.assertEqual(len(requests), summary["orders"])
                self.assertEqual([r["txn_id"] for r in requests], [r["txn_id"] for r in labels])
                split_actors = {r["actor_id"] for r in labels}
                split_cards = {r["card_token"] for r in requests}
                split_addresses = {r["ship_addr_hash"] for r in requests}
                self.assertFalse(actors & split_actors)
                self.assertFalse(cards & split_cards)
                self.assertFalse(addresses & split_addresses)
                actors.update(split_actors)
                cards.update(split_cards)
                addresses.update(split_addresses)
                for req, label in zip(requests, labels):
                    self.assertFalse(set(req) & {"is_scalper", "actor_id", "tactic"})
                    self.assertNotIn(req["txn_id"], seen_ids)
                    seen_ids.add(req["txn_id"])
                    timestamp = parse_time(req["event_ts"])
                    if previous_time is not None:
                        self.assertGreaterEqual(timestamp, previous_time)
                    previous_time = timestamp
                    self.assertGreaterEqual(timestamp, parse_time(summary["start"]))
                    self.assertLess(timestamp, parse_time(summary["end_exclusive"]))
                    self.assertRegex(req["event_ts"], r"\.\d{3}Z$")
                    self.assertRegex(req["card_token"], r"^tok_[0-9a-f]{20}$")
                    self.assertRegex(req["bin"], r"^\d{6}$")
                    self.assertLessEqual(req["card_issue_date"], req["event_ts"][:10])
                    self.assertGreater(req["amount"], 0)
                    self.assertEqual(label["is_scalper"], int(label["tactic"] in TACTICS))
            self.assertEqual(len(seen_ids), 11640)

    def test_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            generate(self.config, a)
            generate(self.config, b)
            for path in a.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (b / path.relative_to(a)).read_bytes())

    def test_held_out_tactic(self):
        self.config["held_out_tactic"] = "card_testing"
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(self.config, tmp)
            self.assertNotIn("card_testing", result["splits"]["train"]["tactics"])
            self.assertNotIn("card_testing", result["splits"]["validation"]["tactics"])
            self.assertGreater(result["splits"]["test"]["tactics"]["card_testing"], 0)

    def test_tactic_behaviors(self):
        start, end = validate_config(self.config)
        sim = Generator(self.config, "test", start, end)
        for tactic in TACTICS:
            sim.emit_session(tactic, 24)
        groups = {t: [r for r, label in sim.rows if label["tactic"] == t] for t in TACTICS}
        ring = groups["shared_address_ring"]
        self.assertEqual(len({r["ship_addr_hash"] for r in ring}), 1)
        self.assertGreater(len({r["card_token"] for r in ring}), 2)
        bulk = groups["max_qty_bulk"]
        self.assertGreater(sum(r["quantity"] == 8 for r in bulk), 12)
        testing = groups["card_testing"]
        self.assertTrue(all(r["amount"] < 3 for r in testing[:16]))
        self.assertTrue(all(r["amount"] > 3 for r in testing[16:]))
        sweep = groups["cross_merchant_sweep"]
        self.assertEqual(len({r["card_token"] for r in sweep}), 1)
        self.assertEqual(len({r["merchant_id"] for r in sweep}), 20)

    def test_bad_config_and_existing_output(self):
        for key, value in (("scalper_ratio", 2), ("merchants", 0), ("held_out_tactic", "unknown")):
            config = deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                generate(self.config, tmp)
            self.assertEqual((Path(tmp) / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
