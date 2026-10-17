"""Ticket-order validation, causal identity/event features, and card histories."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL = "is_scalper"
LINK_COLUMNS = ("shipping_hash", "billing_hash", "email_hash", "device_id")
REQUIRED_TEXT_COLUMNS = ("order_id", "card_id", "card_bin", "merchant_id", "category")
TEXT_COLUMNS = (*REQUIRED_TEXT_COLUMNS, *LINK_COLUMNS, "event_id", "card_type")
REQUIRED_COLUMNS = ("order_id", "timestamp", "card_id", "card_bin", "merchant_id", "category", "amount")
OPTIONAL_NUMERICS = {
    "quantity": (1.0, "quantity_available"),
    "card_age_days": (0.0, "card_age_known"),
    "seconds_since_release": (0.0, "release_time_known"),
    "ticket_limit": (0.0, "ticket_limit_known"),
}
AVAILABILITY_FLAGS = {column: flag for column, (_, flag) in OPTIONAL_NUMERICS.items()}
AVAILABILITY_FLAGS["event_id"] = "event_id_known"
NUMERIC_FEATURES = (
    "amount", "quantity", "quantity_available", "card_age_days", "card_age_known",
    "seconds_since_release", "release_time_known", "ticket_limit", "ticket_limit_known", "event_id_known",
    "card_orders_1m", "card_orders_5m", "card_orders_1h",
    "bin_orders_1m", "bin_orders_5m", "cluster_orders_1m", "cluster_orders_5m", "cluster_orders_1h",
    "card_merchants_5m", "cluster_merchants_5m", "cluster_cards_1h", "linked_cards_count",
    "shipping_linked_cards", "email_linked_cards", "device_linked_cards",
    "seconds_since_previous", "interval_regularity", "basket_uniformity_5m", "hour_sin", "hour_cos",
    "card_event_orders_5m", "cluster_event_orders_5m", "cluster_event_merchants_5m", "cluster_event_cards_5m",
    "card_event_tickets_1h", "cluster_event_tickets_1h", "cluster_event_tickets_24h",
    "cluster_event_quantity_coverage_1h", "cluster_event_limit_excess",
    "cluster_distinct_events_5m", "cluster_same_event_order_fraction_5m", "cluster_event_identity_coverage_5m",
)


def read_events(path: str, *, require_labels: bool = False) -> pd.DataFrame:
    """Read tokenized identifiers as strings, preserving leading zeros."""
    return validate_events(pd.read_csv(path, dtype={c: "string" for c in TEXT_COLUMNS}),
                           require_labels=require_labels)


def _numbers(values: pd.Series, name: str) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite, nonmissing numbers.")
    if (result < 0).any():
        raise ValueError(f"{name} cannot be negative.")
    return result


def _ticket_categories(values: pd.Series) -> pd.Series:
    canonical = values.astype(str).str.strip().str.lower()
    if not canonical.eq("tickets").all():
        raise ValueError("Only category='tickets' is supported; non-ticket orders cannot be scored.")
    return canonical


def validate_events(frame: pd.DataFrame, *, require_labels: bool = False) -> pd.DataFrame:
    """Validate ticket orders and sort globally in stable chronological order."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The payment-order dataset is empty.")
    df = frame.copy()
    for column in REQUIRED_TEXT_COLUMNS:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must contain nonempty tokenized values.")
        df[column] = df[column].astype(str).str.strip()
    df["category"] = _ticket_categories(df["category"])
    for column in (*LINK_COLUMNS, "event_id"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()
    if "event_id_known" in df:
        if df.event_id_known.isna().any() or not df.event_id_known.isin([0, 1]).all():
            raise ValueError("event_id_known must contain only 0 or 1.")
        if (df.event_id_known.eq(1) & df.event_id.eq("")).any():
            raise ValueError("event_id cannot be missing when event_id_known is 1.")
        df.loc[df.event_id_known.eq(0), "event_id"] = ""
    df["event_id_known"] = df.event_id.ne("").astype(int)
    if "card_type" not in df:
        df["card_type"] = "unknown"
    df["card_type"] = df["card_type"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    if df.order_id.duplicated().any():
        raise ValueError("Duplicate order_id values are not allowed.")
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("timestamp must contain valid ISO-8601 timestamps.") from exc
    if df.timestamp.isna().any():
        raise ValueError("timestamp cannot be missing.")
    df["amount"] = _numbers(df["amount"], "amount")
    if df.amount.le(0).any():
        raise ValueError("amount must be greater than zero.")
    population_columns = ("processor_bin_orders_1m", "processor_bin_orders_5m")
    for column in population_columns:
        if column not in df:
            df[column] = np.nan
        absent = df[column].isna() | df[column].astype(str).str.strip().eq("")
        present = ~absent
        values = _numbers(df.loc[present, column], column)
        if values.le(0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain positive whole counts including the current order.")
        df[column] = pd.to_numeric(df[column].mask(absent, np.nan), errors="raise").astype(float)
    if df[population_columns[0]].notna().ne(df[population_columns[1]].notna()).any():
        raise ValueError("Supply both processor_bin_orders_1m and processor_bin_orders_5m, or neither.")
    if df[population_columns[1]].lt(df[population_columns[0]]).any():
        raise ValueError("processor_bin_orders_5m cannot be less than processor_bin_orders_1m.")
    for column, (default, flag) in OPTIONAL_NUMERICS.items():
        if column not in df:
            df[column] = np.nan
        absent = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if flag in df:
            if df[flag].isna().any() or not df[flag].isin([0, 1]).all():
                raise ValueError(f"{flag} must contain only 0 or 1.")
            known = df[flag].astype(int)
            if (known.eq(1) & absent).any():
                raise ValueError(f"{column} cannot be missing when {flag} is 1.")
        else:
            known = (~absent).astype(int)
        df[column] = _numbers(df[column].mask(absent, default), column)
        df[flag] = known
        df.loc[known.eq(0), column] = default
    if df.quantity.le(0).any() or (df.quantity % 1 != 0).any():
        raise ValueError("quantity must contain positive whole units when available.")
    available_limits = df.loc[df.ticket_limit_known.eq(1), "ticket_limit"]
    if available_limits.le(0).any() or (available_limits % 1 != 0).any():
        raise ValueError("ticket_limit must contain positive whole units when known.")
    if require_labels and LABEL not in df:
        raise ValueError("Training requires an is_scalper column with independently verified per-order 0/1 labels.")
    if LABEL in df:
        if df[LABEL].isna().any() or not df[LABEL].isin([0, 1]).all():
            raise ValueError("is_scalper must contain only 0 or 1 per order.")
        df[LABEL] = df[LABEL].astype(int)
    canonical = [*REQUIRED_COLUMNS, *LINK_COLUMNS, "event_id", "event_id_known", "card_type"]
    for column, (_, flag) in OPTIONAL_NUMERICS.items():
        canonical.extend((column, flag))
    if LABEL in df:
        canonical.append(LABEL)
    canonical.extend(column for column in df.columns if column not in canonical)
    return df[canonical].sort_values("timestamp", kind="stable").reset_index(drop=True)


class _IdentityGraph:
    """Union-find over cards connected through observed nonempty identities."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.identities: dict[tuple[str, str], set[str]] = defaultdict(set)

    def find(self, card: str) -> str:
        if card not in self.parent:
            self.parent[card] = card
            self.members[card] = {card}
        root = card
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[card] != card:
            previous = self.parent[card]
            self.parent[card] = root
            card = previous
        return root

    @staticmethod
    def key(column: str, value: str) -> tuple[str, str]:
        # A hashed address can appear as shipping on one card and billing on another.
        return ("address" if column in {"shipping_hash", "billing_hash"} else column, value)

    def connect(self, order: dict) -> tuple[str, list[tuple[str, str]]]:
        card = order["card_id"]
        root = self.find(card)
        mergers = []
        for column in LINK_COLUMNS:
            value = order[column]
            if not value:
                continue
            key = self.key(column, value)
            for other in self.identities[key]:
                other_root = self.find(other)
                root = self.find(root)
                if root != other_root:
                    keep, remove = sorted((root, other_root))
                    self.parent[remove] = keep
                    self.members[keep].update(self.members.pop(remove))
                    mergers.append((keep, remove))
                    root = keep
            self.identities[key].add(card)
        return self.find(card), mergers

    def identity_count(self, column: str, value: str) -> int:
        return len(self.identities[self.key(column, value)]) if value else 0


class PaymentFeatureStore:
    """Online causal payment features. Counts include the current order.

    Temporal windows include both boundaries. BINs supply population velocity
    and never form graph links. card_id is an upstream token, not a raw PAN.
    """

    def __init__(self):
        self.graph = _IdentityGraph()
        self.card_events: dict[str, deque] = defaultdict(deque)
        self.bin_events: dict[str, deque] = defaultdict(deque)
        self.cluster_events: dict[str, deque] = defaultdict(deque)
        self.card_totals: dict[str, int] = defaultdict(int)
        self.card_last_time: dict[str, int] = {}
        self.seen_orders: set[str] = set()
        self.last_time: int | None = None

    @staticmethod
    def _trim(events: deque, cutoff: int) -> None:
        while events and events[0]["time"] < cutoff:
            events.popleft()

    def update(self, order: dict) -> dict:
        """Validate one order and reject duplicates or arrivals before the last timestamp."""
        return self._update_validated(validate_events(pd.DataFrame([order])).iloc[0].to_dict())

    def _update_validated(self, order: dict) -> dict:
        time = order["timestamp"].value
        if self.last_time is not None and time < self.last_time:
            raise ValueError("PaymentFeatureStore requires nondecreasing timestamp order.")
        if order["order_id"] in self.seen_orders:
            raise ValueError("Duplicate order_id values are not allowed.")
        self.last_time = time
        self.seen_orders.add(order["order_id"])
        root, mergers = self.graph.connect(order)
        for keep, remove in mergers:
            merged = [*self.cluster_events[keep], *self.cluster_events.pop(remove, ())]
            merged.sort(key=lambda event: event["time"])
            self.cluster_events[keep] = deque(merged)
        card = order["card_id"]
        previous = self.card_last_time.get(card)
        self.card_last_time[card] = time
        self.card_totals[card] += 1
        event = {"time": time, "merchant": order["merchant_id"], "card": card, "amount": order["amount"],
                 "event": order["event_id"], "quantity_available": bool(order["quantity_available"]),
                 "known_quantity": order["quantity"] if order["quantity_available"] else 0.0}
        card_history, bin_history = self.card_events[card], self.bin_events[order["card_bin"]]
        cluster_history = self.cluster_events[root]
        for history, span in ((card_history, 86400), (bin_history, 300), (cluster_history, 86400)):
            self._trim(history, time - span * 1_000_000_000)
            history.append(event)
        one_minute = time - 60 * 1_000_000_000
        five_minutes = time - 300 * 1_000_000_000
        one_hour = time - 3600 * 1_000_000_000
        card_recent = [item for item in card_history if item["time"] >= five_minutes]
        cluster_recent = [item for item in cluster_history if item["time"] >= five_minutes]
        card_hour = [item for item in card_history if item["time"] >= one_hour]
        cluster_hour = [item for item in cluster_history if item["time"] >= one_hour]
        intervals = np.diff([item["time"] for item in cluster_recent]) / 1e9
        if len(intervals) >= 2:
            center = float(intervals.mean())
            regularity = 1.0 if center == 0 else 1 / (1 + float(intervals.std()) / center)
        else:
            regularity = 0.0
        amounts = np.asarray([item["amount"] for item in cluster_recent])
        normalized_amounts = amounts / amounts.max()
        uniformity = (1 / (1 + float(normalized_amounts.std()) / float(normalized_amounts.mean()))
                      if len(amounts) >= 2 else 0.0)
        event_features = {name: 0.0 for name in (
            "card_event_orders_5m", "cluster_event_orders_5m", "cluster_event_merchants_5m",
            "cluster_event_cards_5m", "card_event_tickets_1h", "cluster_event_tickets_1h",
            "cluster_event_tickets_24h", "cluster_event_quantity_coverage_1h", "cluster_event_limit_excess",
        )}
        target = order["event_id"]
        if target:
            card_event_recent = [item for item in card_recent if item["event"] == target]
            cluster_event_recent = [item for item in cluster_recent if item["event"] == target]
            card_event_hour = [item for item in card_hour if item["event"] == target]
            cluster_event_hour = [item for item in cluster_hour if item["event"] == target]
            cluster_event_day = [item for item in cluster_history if item["event"] == target]
            tickets_day = sum(item["known_quantity"] for item in cluster_event_day)
            event_features.update({
                "card_event_orders_5m": len(card_event_recent),
                "cluster_event_orders_5m": len(cluster_event_recent),
                "cluster_event_merchants_5m": len({item["merchant"] for item in cluster_event_recent}),
                "cluster_event_cards_5m": len({item["card"] for item in cluster_event_recent}),
                "card_event_tickets_1h": sum(item["known_quantity"] for item in card_event_hour),
                "cluster_event_tickets_1h": sum(item["known_quantity"] for item in cluster_event_hour),
                "cluster_event_tickets_24h": tickets_day,
                "cluster_event_quantity_coverage_1h": (
                    sum(item["quantity_available"] for item in cluster_event_hour) / len(cluster_event_hour)),
                "cluster_event_limit_excess": max(0, tickets_day - order["ticket_limit"])
                    if order["ticket_limit_known"] else 0.0,
            })
        known_event_recent = [item for item in cluster_recent if item["event"]]
        event_features.update({
            "cluster_distinct_events_5m": len({item["event"] for item in known_event_recent}),
            "cluster_same_event_order_fraction_5m": (
                event_features["cluster_event_orders_5m"] / len(known_event_recent)
                if target and known_event_recent else 0.0),
            "cluster_event_identity_coverage_5m": len(known_event_recent) / len(cluster_recent),
        })
        hour = order["timestamp"].hour + order["timestamp"].minute / 60
        return {
            **order,
            "card_orders_1m": sum(item["time"] >= one_minute for item in card_history),
            "card_orders_5m": len(card_recent), "card_orders_1h": len(card_hour),
            "bin_orders_1m": (int(order["processor_bin_orders_1m"]) if pd.notna(order["processor_bin_orders_1m"])
                              else sum(item["time"] >= one_minute for item in bin_history)),
            "bin_orders_5m": (int(order["processor_bin_orders_5m"]) if pd.notna(order["processor_bin_orders_5m"])
                              else len(bin_history)),
            "cluster_orders_1m": sum(item["time"] >= one_minute for item in cluster_history),
            "cluster_orders_5m": len(cluster_recent), "cluster_orders_1h": len(cluster_hour),
            "card_merchants_5m": len({item["merchant"] for item in card_recent}),
            "cluster_merchants_5m": len({item["merchant"] for item in cluster_recent}),
            "cluster_cards_1h": len({item["card"] for item in cluster_hour}),
            "linked_cards_count": len(self.graph.members[root]),
            "shipping_linked_cards": self.graph.identity_count("shipping_hash", order["shipping_hash"]),
            "email_linked_cards": self.graph.identity_count("email_hash", order["email_hash"]),
            "device_linked_cards": self.graph.identity_count("device_id", order["device_id"]),
            "seconds_since_previous": 0.0 if previous is None else (time - previous) / 1e9,
            "interval_regularity": regularity, "basket_uniformity_5m": uniformity,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "history_count": self.card_totals[card], "cluster_id": root,
            **event_features,
        }


def engineer_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute causal features with fresh state, including a causal graph."""
    ordered = validate_events(frame)
    store = PaymentFeatureStore()
    return pd.DataFrame([store._update_validated(order) for order in ordered.to_dict("records")])


def linked_components(frame: pd.DataFrame) -> dict[str, str]:
    """Map cards to final components for retrospective reporting/splits only."""
    graph = _IdentityGraph()
    for order in validate_events(frame).to_dict("records"):
        graph.connect(order)
    return {card: graph.find(card) for card in graph.parent}


def split_accounts(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split linked components into train/validation/calibration/test (60/15/10/15)."""
    df = validate_events(frame, require_labels=True)
    df["split_group"] = df.card_id.map(linked_components(df))
    groups = df.groupby("split_group", sort=True)[LABEL].max()
    if set(groups.unique()) != {0, 1}:
        raise ValueError("Training requires both all-normal and positive linked-card components.")
    rng = np.random.default_rng(seed)
    ids: dict[str, list[str]] = {"train": [], "validation": [], "calibration": [], "test": []}
    for label in (0, 1):
        component_ids = groups.index[groups.eq(label)].to_numpy().copy()
        if len(component_ids) < 4:
            raise ValueError("Provide at least four independent linked-card components per class for train/validation/calibration/test.")
        rng.shuffle(component_ids)
        validation_count = max(1, int(round(len(component_ids) * 0.15)))
        calibration_count = max(1, int(round(len(component_ids) * 0.10)))
        test_count = max(1, int(round(len(component_ids) * 0.15)))
        validation_end = validation_count
        calibration_end = validation_end + calibration_count
        test_end = calibration_end + test_count
        ids["validation"].extend(component_ids[:validation_end])
        ids["calibration"].extend(component_ids[validation_end:calibration_end])
        ids["test"].extend(component_ids[calibration_end:test_end])
        ids["train"].extend(component_ids[test_end:])
    return {name: df[df.split_group.isin(values)].reset_index(drop=True) for name, values in ids.items()}


def numeric_features(frame: pd.DataFrame) -> np.ndarray:
    """Extract numeric fields from already causally engineered orders."""
    return frame.loc[:, list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)


@dataclass
class EventPreprocessor:
    """Training-only scaling and categorical vocabularies, saved as JSON."""

    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    vocabularies: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "EventPreprocessor":
        df = engineer_events(frame)
        values = numeric_features(df)
        # Normalize first so legitimate finite, large payment amounts cannot
        # overflow the squared differences used by standard deviation.
        magnitude = np.maximum(np.abs(values).max(axis=0), 1)
        normalized = values / magnitude
        self.mean = (normalized.mean(axis=0) * magnitude).tolist()
        self.scale = np.maximum(normalized.std(axis=0) * magnitude, 1e-6).tolist()
        self.vocabularies = {c: sorted(df[c].unique().tolist()) for c in ("category", "card_type")}
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(NUMERIC_FEATURES) + [f"{column}={value}" for column, values in self.vocabularies.items()
                                       for value in ["<unknown>", *values]]

    def transform_engineered(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.mean:
            raise ValueError("Fit the preprocessor on training data before transforming.")
        frame = frame.copy()
        frame["category"] = _ticket_categories(frame["category"])
        values = numeric_features(frame)
        if not np.isfinite(values).all():
            raise ValueError("Engineered payment features must be finite.")
        scaled = np.clip((values - np.asarray(self.mean)) / np.asarray(self.scale), -8, 8)
        features = [scaled]
        for column, vocabulary in self.vocabularies.items():
            mapping = {value: i + 1 for i, value in enumerate(vocabulary)}
            indices = frame[column].map(mapping).fillna(0).to_numpy(dtype=int)
            encoded = np.zeros((len(frame), len(vocabulary) + 1), dtype=np.float32)
            encoded[np.arange(len(frame)), indices] = 1
            features.append(encoded)
        return np.concatenate(features, axis=1).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df = engineer_events(frame)
        return df, self.transform_engineered(df)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "scale": self.scale, "vocabularies": self.vocabularies}

    @classmethod
    def from_dict(cls, values: dict) -> "EventPreprocessor":
        return cls(mean=values["mean"], scale=values["scale"], vocabularies=values["vocabularies"])


class ActivitySequences(Dataset):
    """Right-padded histories of prior orders from the same card, including now."""

    def __init__(self, frame: pd.DataFrame, features: np.ndarray, sequence_length: int = 7,
                 *, latest_only: bool = False):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        if features.ndim != 2 or len(frame) != len(features):
            raise ValueError("Feature rows must match the payment dataframe.")
        self.frame = frame.reset_index(drop=True)
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.sequence_length = sequence_length
        self.card_positions: dict[str, list[int]] = defaultdict(list)
        self.offsets = np.zeros(len(frame), dtype=np.int64)
        for position, card in enumerate(self.frame.card_id):
            self.offsets[position] = len(self.card_positions[card])
            self.card_positions[card].append(position)
        self.endpoints = np.asarray(
            sorted(positions[-1] for positions in self.card_positions.values()) if latest_only else range(len(frame)),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        end = int(self.endpoints[index])
        card = self.frame.iloc[end].card_id
        offset = int(self.offsets[end]) + 1
        positions = self.card_positions[card][max(0, offset - self.sequence_length):offset]
        length = len(positions)
        window = torch.zeros((self.sequence_length, self.features.shape[1]), dtype=torch.float32)
        window[:length] = self.features[positions]
        label = float(self.frame.iloc[end][LABEL]) if LABEL in self.frame else 0.0
        return window, length, torch.tensor(label, dtype=torch.float32)
