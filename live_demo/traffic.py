"""Bounded entity pools and continuing campaigns; no training files are read."""

from __future__ import annotations

import hashlib
import math
import random
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

TACTICS = ("shared_address_ring", "max_qty_bulk", "card_testing", "cross_merchant_sweep")


@dataclass
class DemoConfig:
    """Traffic volume, entity pools, and release-window settings for one run."""

    duration: float = 120.0
    rate: float = 2.0
    scalper_ratio: float = 0.25
    seed: int = 2026
    customers: int = 120
    merchants: int = 12
    events: int = 6
    drop_after: float = 30.0
    drop_duration: float = 20.0
    drop_multiplier: float = 5.0
    timeout: float = 10.0

    def validate(self) -> DemoConfig:
        """Reject invalid scenario settings and return this configuration."""
        for name in ("duration", "rate", "drop_duration", "drop_multiplier", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.drop_after) or self.drop_after < 0:
            raise ValueError("drop_after must be finite and nonnegative")
        if not math.isfinite(self.scalper_ratio) or not 0 <= self.scalper_ratio <= 1:
            raise ValueError("scalper_ratio must be in [0, 1]")
        for name, minimum in (("customers", 1), ("merchants", 2), ("events", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Traffic:
    """Generate reproducible arrivals and purchases for a synthetic session."""

    def __init__(
        self,
        config: DemoConfig,
        start: datetime | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config.validate()
        self.start = start or datetime.now(timezone.utc)
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        self.start = self.start.astimezone(timezone.utc)
        self.run_id = run_id or str(uuid.uuid4().int)
        if not self.run_id.isascii() or not self.run_id.isdigit() or len(self.run_id) > 50:
            raise ValueError("run_id must contain 1 to 50 ASCII digits")
        # Independent clocks keep traffic generation from altering arrival times.
        self.rng = random.Random(config.seed)
        self.clock_rng = random.Random(f"{config.seed}:clock")
        self.sequence = 0
        self.campaign_number = 0
        self.campaigns: dict[str, dict[str, Any]] = {}
        self.customers = [
            self.actor(f"customer:{i}", household=f"household:{i // 3}")
            for i in range(config.customers)
        ]
        self.prices = [
            self.rng.choice((4500, 6500, 9500, 12500, 18500)) for _ in range(config.events)
        ]

    def token(self, prefix: str, identity: str) -> str:
        """Create a stable token scoped to the scenario seed and run ID."""
        digest = hashlib.sha256(
            f"{self.config.seed}:{self.run_id}:{identity}".encode()
        ).hexdigest()[:20]
        return f"{prefix}_{digest}"

    def card(self, identity: str, new: bool = False) -> dict[str, str]:
        """Create card metadata, optionally restricting it to newly issued cards."""
        card_types = ("prepaid", "virtual") if new else ("credit", "debit", "prepaid", "virtual")
        return {
            "card_token": self.token("tok", identity),
            "bin": self.rng.choice(("900001", "900002", "900003", "900004")),
            "card_type": self.rng.choice(card_types),
            "card_issue_date": (
                self.start
                - timedelta(days=self.rng.randint(1, 14) if new else self.rng.randint(1, 1800))
            )
            .date()
            .isoformat(),
        }

    def actor(self, identity: str, household: str | None = None) -> dict[str, Any]:
        """Create a customer profile with stable household linkage tokens."""
        address = self.token("addr", household or identity)
        return {
            "actor_id": self.token("actor", identity),
            "card": self.card(identity),
            "ship_addr_hash": address,
            "bill_addr_hash": address,
            "email_hash": self.token("em", identity),
            "device_id": self.token("dev", household or identity),
            "ip_prefix": self.rng.choice(("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")),
            "channel": self.rng.choices(("web", "app", "api"), (65, 33, 2))[0],
            "home_merchant": self.rng.randrange(self.config.merchants),
            "bulk_buyer": hashlib.sha256(identity.encode()).digest()[0] % 6 == 0,
        }

    def in_drop(self, elapsed: float) -> bool:
        """Return whether an arrival falls inside the ticket release window."""
        return (
            self.config.drop_after <= elapsed < self.config.drop_after + self.config.drop_duration
        )

    def arrivals(self) -> Iterator[float]:
        """Piecewise Poisson arrivals, integrating rate across release boundaries."""
        elapsed = 0.0
        boundaries = sorted(
            {
                0.0,
                self.config.duration,
                min(self.config.duration, self.config.drop_after),
                min(self.config.duration, self.config.drop_after + self.config.drop_duration),
            }
        )
        while elapsed < self.config.duration:
            work = self.clock_rng.expovariate(1.0)
            while elapsed < self.config.duration:
                boundary = next(b for b in boundaries if b > elapsed)
                rate = self.config.rate * (
                    self.config.drop_multiplier if self.in_drop(elapsed) else 1
                )
                capacity = (boundary - elapsed) * rate
                if work < capacity:
                    elapsed += work / rate
                    yield elapsed
                    break
                work -= capacity
                elapsed = boundary

    def campaign(self, tactic: str) -> dict[str, Any]:
        """Continue a tactic's campaign or replace it when its order limit is met."""
        current = self.campaigns.get(tactic)
        if current is None or current["index"] >= current["size"]:
            self.campaign_number += 1
            identity = f"campaign:{self.campaign_number}"
            actor = self.actor(identity)
            current = {
                "actor": actor,
                "index": 0,
                "size": self.rng.randint(18, 30),
                "event": self.rng.randrange(self.config.events),
                "cards": [
                    self.card(f"{identity}:card:{i}", new=tactic == "card_testing")
                    for i in range(12)
                ],
            }
            self.campaigns[tactic] = current
        return current

    def purchase(self, elapsed: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (wire request, separate truth); each call creates one new order."""
        malicious = self.rng.random() < self.config.scalper_ratio
        tactic = self.rng.choice(TACTICS) if malicious else "normal"
        event = 0 if self.in_drop(elapsed) else self.rng.randrange(self.config.events)
        cents = None
        if malicious:
            campaign = self.campaign(tactic)
            actor, index = campaign["actor"], campaign["index"]
            campaign["index"] += 1
            card = actor["card"]
            merchant = actor["home_merchant"]
            event = 0 if self.in_drop(elapsed) else campaign["event"]
            quantity = self.rng.randint(1, 8)
            if tactic == "shared_address_ring":
                card = campaign["cards"][index % len(campaign["cards"])]
            elif tactic == "max_qty_bulk":
                quantity = 8 if self.rng.random() < 0.85 else self.rng.randint(1, 8)
            elif tactic == "card_testing":
                card = campaign["cards"][index % len(campaign["cards"])]
                if index < 12:
                    quantity, cents = 1, self.rng.choice((50, 99, 100, 199, 299))
            else:
                merchant = (merchant + index) % self.config.merchants
        else:
            actor = self.rng.choice(self.customers)
            card, merchant = actor["card"], actor["home_merchant"]
            # Stable corporate/family customers return with large legitimate orders.
            is_bulk = actor["bulk_buyer"]
            tactic = "legit_bulk" if is_bulk else "normal"
            quantity = (
                self.rng.randint(4, 6)
                if is_bulk
                else self.rng.choices((1, 2, 3, 4, 5, 6), (30, 35, 14, 14, 4, 3))[0]
            )
            if self.rng.random() < 0.15:
                merchant = self.rng.randrange(self.config.merchants)
        self.sequence += 1
        txn_id = f"T{self.run_id}{self.sequence:012d}"
        timestamp = self.start + timedelta(seconds=elapsed)
        request = {
            "txn_id": txn_id,
            "event_ts": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            **card,
            "merchant_id": f"M{merchant:04d}",
            "mcc": "7922",
            "event_id": f"EVT_{event:04d}",
            "quantity": quantity,
            "amount": (cents if cents is not None else self.prices[event] * quantity) / 100,
            "currency": "USD",
            **{
                key: actor[key]
                for key in (
                    "ship_addr_hash",
                    "bill_addr_hash",
                    "email_hash",
                    "device_id",
                    "ip_prefix",
                    "channel",
                )
            },
        }
        if self.rng.random() < 0.15:
            request["bill_addr_hash"] = self.token("addr", f"billing:{actor['actor_id']}")
        if malicious and self.rng.random() < 0.2:
            request["device_id"] = self.token("dev", f"rotate:{self.sequence}")
        # Missing merchant-supplied metadata is realistic in both populations.
        for key in ("device_id", "ip_prefix"):
            if self.rng.random() < 0.03:
                request.pop(key)
        truth = {
            "txn_id": txn_id,
            "is_scalper": int(malicious),
            "actor_id": actor["actor_id"],
            "tactic": tactic,
        }
        return request, truth
