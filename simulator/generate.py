"""Generate synthetic JSONL requests and isolated evaluation labels (stdlib only)."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random


TACTICS = ("shared_address_ring", "max_qty_bulk", "card_testing", "cross_merchant_sweep")
SPLITS = ("train", "validation", "test")
CARD_TYPES = ("credit", "debit", "prepaid", "virtual")


def parse_time(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_config(c):
    start, end = parse_time(c["start"]), parse_time(c["end"])
    if end - start < timedelta(seconds=3):
        raise ValueError("date range must span at least three seconds")
    for key in ("merchants", "events", "customers_per_split", "quantity_cap"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("scalper_ratio", "legit_bulk_ratio", "optional_field_missing_rate"):
        if not 0 <= c[key] <= 1:
            raise ValueError(f"{key} must be between zero and one")
    if not 0 <= c["baseline_orders_per_hour"] < float("inf"):
        raise ValueError("baseline_orders_per_hour must be finite and nonnegative")
    fractions = c["split_fractions"]
    if set(fractions) != set(SPLITS) or any(v <= 0 for v in fractions.values()):
        raise ValueError("split_fractions must contain positive train, validation, test fractions")
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to one")
    if c.get("held_out_tactic") not in (None, *TACTICS):
        raise ValueError("unknown held_out_tactic")
    weights = c["tactic_weights"]
    if set(weights) != set(TACTICS) or any(not 0 <= w < float("inf") for w in weights.values()):
        raise ValueError("tactic_weights must specify four finite, nonnegative weights")
    for split in SPLITS:
        available = [t for t in TACTICS if split == "test" or t != c.get("held_out_tactic")]
        if sum(weights[t] for t in available) <= 0:
            raise ValueError(f"no positive tactic weights available for {split}")
    for drop in c["drops"]:
        timestamp = parse_time(drop["timestamp"])
        if not start <= timestamp < end:
            raise ValueError("drop timestamp must be inside the simulation range")
        if type(drop["orders"]) is not int or drop["orders"] < 0:
            raise ValueError("drop orders must be a nonnegative integer")
        if not 0 < drop["window_seconds"] < float("inf"):
            raise ValueError("drop window_seconds must be finite and positive")
        if type(drop["event_index"]) is not int or not 0 <= drop["event_index"] < c["events"]:
            raise ValueError("drop event_index is outside the event pool")
    return start, end


class Generator:
    def __init__(self, config, split, start, end):
        self.c, self.split, self.start, self.end = config, split, start, end
        self.rng = random.Random(f"{config['seed']}:{split}")
        self.counter = 0
        self.rows = []
        self.prices = [self.rng.choice((4500, 6500, 9500, 12500, 18500)) for _ in range(config["events"])]
        self.customers = [self.actor("customer", i) for i in range(config["customers_per_split"])]

    def token(self, prefix, identity):
        digest = hashlib.sha256(f"{self.c['seed']}:{self.split}:{identity}".encode()).hexdigest()[:20]
        return f"{prefix}_{digest}"

    def card(self, identity, recent=False):
        return {
            "card_token": self.token("tok", identity),
            # Deliberately synthetic six-digit issuer metadata; never a PAN.
            "bin": self.rng.choice(("900001", "900002", "900003", "900004")),
            "card_type": self.rng.choice(("prepaid", "virtual") if recent else CARD_TYPES),
            "card_issue_date": (self.start - timedelta(days=self.rng.randint(1, 20) if recent else self.rng.randint(1, 1800))).date().isoformat(),
        }

    def actor(self, kind, index):
        identity = f"{kind}:{index}"
        # Households deliberately share addresses across distinct customers.
        household = f"household:{index // 3}" if kind == "customer" else identity
        return {
            "actor_id": self.token("actor", identity),
            "card": self.card(identity),
            "ship_addr_hash": self.token("addr", household),
            "bill_addr_hash": self.token("addr", household),
            "email_hash": self.token("em", identity),
            "device_id": self.token("dev", household),
            "ip_prefix": self.rng.choice(("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")),
            "channel": self.rng.choices(("web", "app", "api"), (65, 33, 2))[0],
        }

    def daily_anchor(self):
        # Rejection sampling: overnight is quieter, afternoon/evening busiest.
        seconds = (self.end - self.start).total_seconds()
        while True:
            timestamp = self.start + timedelta(seconds=self.rng.random() * seconds)
            hour = timestamp.hour
            weight = 0.18 if hour < 7 else (0.6 if hour < 12 else (1 if hour < 23 else 0.4))
            if self.rng.random() < weight:
                return timestamp

    def normal(self, actor, n):
        return [(actor["card"], self.rng.choices((1, 2, 3, 4, 5, 6), (28, 35, 15, 15, 4, 3))[0], None, None) for _ in range(n)]

    def legit_bulk(self, actor, n):
        # Corporate buyers and families overlap ring and bulk behavior.
        cards = [actor["card"], self.card(f"{actor['actor_id']}:corporate")]
        return [(self.rng.choice(cards), self.rng.randint(4, 6), None, None) for _ in range(n)]

    def shared_address_ring(self, actor, n):
        cards = [self.card(f"{actor['actor_id']}:{i}") for i in range(max(2, n // 2))]
        return [(self.rng.choice(cards), self.rng.randint(1, self.c["quantity_cap"]), None, None) for _ in range(n)]

    def max_qty_bulk(self, actor, n):
        event = self.rng.randrange(self.c["events"])
        cap = self.c["quantity_cap"]
        return [(actor["card"], cap if self.rng.random() < 0.8 else self.rng.randint(1, cap), event, None) for _ in range(n)]

    def card_testing(self, actor, n):
        cards = [self.card(f"{actor['actor_id']}:test:{i}", recent=True) for i in range(max(2, n // 2))]
        probes = max(1, n * 2 // 3)
        return [(cards[i % len(cards)], 1 if i < probes else self.rng.randint(2, self.c["quantity_cap"]), None,
                 self.rng.choice((50, 99, 100, 199, 299)) if i < probes else None) for i in range(n)]

    def cross_merchant_sweep(self, actor, n):
        return [(actor["card"], self.rng.randint(1, self.c["quantity_cap"]), None, None) for _ in range(n)]

    def emit_session(self, tactic, n, drop=None):
        malicious = tactic in TACTICS
        actor = self.actor("session", self.counter) if malicious else self.rng.choice(self.customers)
        self.counter += 1
        specs = getattr(self, tactic)(actor, n)
        anchor = parse_time(drop["timestamp"]) if drop else self.daily_anchor()
        window = drop["window_seconds"] if drop else (180 if malicious else 900)
        if drop:
            # Both distributions overlap; automation is typically faster.
            delay = min(window * 0.85, self.rng.expovariate(1 / (1.5 if malicious else 18)))
            span = min(window - delay, self.rng.uniform(0.2, 12 if malicious else 70))
        else:
            delay, span = 0, self.rng.uniform(1, window)
        merchant_start = self.rng.randrange(self.c["merchants"])
        home_merchant = self.rng.randrange(self.c["merchants"])
        for i, (card, quantity, event, amount_cents) in enumerate(specs):
            timestamp = anchor + timedelta(seconds=delay + span * i / max(1, n - 1))
            timestamp = min(timestamp, self.end - timedelta(milliseconds=1))
            event = drop["event_index"] if drop else (event if event is not None else self.rng.randrange(self.c["events"]))
            merchant = (merchant_start + i) % self.c["merchants"] if tactic == "cross_merchant_sweep" else home_merchant
            cents = amount_cents if amount_cents is not None else self.prices[event] * quantity
            request = {
                "event_ts": iso(timestamp), **card,
                "merchant_id": f"M{merchant:04d}", "mcc": "7922",
                "event_id": f"EVT_{event:04d}", "quantity": quantity,
                "amount": cents / 100, "currency": "USD",
                **{k: actor[k] for k in ("ship_addr_hash", "bill_addr_hash", "email_hash", "device_id", "ip_prefix", "channel")},
            }
            # Address mismatch is possible in both classes; no perfect shortcut.
            if self.rng.random() < 0.15:
                request["bill_addr_hash"] = self.token("addr", f"billing:{actor['actor_id']}")
            # Some rings rotate devices; some legitimate households share them.
            if malicious and self.rng.random() < 0.25:
                request["device_id"] = self.token("dev", f"rotate:{self.counter}:{i}")
            for key in ("event_id", "device_id", "ip_prefix"):
                if self.rng.random() < self.c["optional_field_missing_rate"]:
                    request.pop(key)
            label = {"is_scalper": int(malicious), "actor_id": actor["actor_id"], "tactic": tactic}
            self.rows.append((request, label))

    def population(self, count, drop=None):
        bad = round(count * self.c["scalper_ratio"])
        bulk = round((count - bad) * self.c["legit_bulk_ratio"])
        groups = [("normal", count - bad - bulk), ("legit_bulk", bulk)]
        available = [t for t in TACTICS if self.split == "test" or t != self.c.get("held_out_tactic")]
        while bad:
            tactic = self.rng.choices(available, [self.c["tactic_weights"][t] for t in available])[0]
            size = min(bad, self.rng.randint(8, 28))
            groups.append((tactic, size))
            bad -= size
        for tactic, remaining in groups:
            while remaining:
                size = min(remaining, self.rng.randint(2, 5) if tactic == "legit_bulk" else (1 if tactic == "normal" else remaining))
                self.emit_session(tactic, size, drop)
                remaining -= size

    def run(self):
        hours = (self.end - self.start).total_seconds() / 3600
        self.population(round(hours * self.c["baseline_orders_per_hour"]))
        for drop in self.c["drops"]:
            if self.start <= parse_time(drop["timestamp"]) < self.end:
                self.population(drop["orders"], drop)
        self.rows.sort(key=lambda pair: pair[0]["event_ts"])
        return self.rows


def generate(config, output):
    start, end = validate_config(config)
    output = Path(output)
    # Avoid accidentally replacing training artifacts or leaving stale files.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    manifest = {"schema_version": 1, "config": config, "splits": {},
                "notes": ["Entirely synthetic; no real card numbers or customer data.",
                          "Actor and entity pools are isolated across chronological splits.",
                          "Synthetic test performance does not establish real-world performance.",
                          "Amounts are JSON numbers in currency units; generated using integer cents."]}
    cursor, txn_index = start, 0
    for split in SPLITS:
        boundary = end if split == "test" else cursor + (end - start) * config["split_fractions"][split]
        rows = Generator(config, split, cursor, boundary).run()
        folder = output / split
        folder.mkdir()
        counts = Counter()
        with (folder / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as requests, (folder / "ground_truth.jsonl").open("w", encoding="utf-8", newline="\n") as labels:
            for request, label in rows:
                txn_index += 1
                txn_id = f"T{txn_index:012d}"
                requests.write(json.dumps({"txn_id": txn_id, **request}, separators=(",", ":")) + "\n")
                labels.write(json.dumps({"txn_id": txn_id, **label}, separators=(",", ":")) + "\n")
                counts[label["tactic"]] += 1
        manifest["splits"][split] = {"start": iso(cursor), "end_exclusive": iso(boundary), "orders": len(rows), "tactics": dict(sorted(counts.items()))}
        cursor = boundary
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "default.json")
    parser.add_argument("--output", type=Path, default=Path("data/simulation"))
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        manifest = generate(config, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    for split, summary in manifest["splits"].items():
        print(f"{split}: {summary['orders']:,} orders, tactics={summary['tactics']}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
