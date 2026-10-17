"""Stateful simulated authorization path; no orders are blocked or refunded here."""

from __future__ import annotations

from collections import defaultdict, deque
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .data import LABEL, PaymentFeatureStore
from .pipeline import _combine


class OrderScoringSession:
    """Maintain causal cross-merchant counters and recent histories in one process.

    Feed a single chronological, authorized payment stream into each session.
    Histories are in memory; a fresh session has no knowledge of previous calls
    or other workers. Shared tokens must use the same namespace across merchants.
    """

    def __init__(self, detector, *, policy=None, merchant_policies=None):
        self.detector = detector
        self.policy = policy
        self.merchant_policies = merchant_policies or {}
        self.feature_store = PaymentFeatureStore()
        self.sequence_length = detector.metadata["training_config"]["sequence_length"]
        self.histories = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self.last_neural_ms = 0.0

    def observe(self, order: dict) -> tuple[dict, np.ndarray]:
        """Warm context using an earlier order, without producing a risk decision."""
        clean = {name: value for name, value in order.items() if name != LABEL}
        features = self.feature_store.update(clean)
        values = self.detector.preprocessor.transform_engineered(pd.DataFrame([features]))[0]
        self.histories[features["card_id"]].append(values)
        return features, values

    def score_order(self, order: dict) -> dict:
        features, values = self.observe(order)
        history = self.histories[features["card_id"]]
        tensor = torch.zeros((1, self.sequence_length, len(values)), dtype=torch.float32)
        tensor[0, :len(history)] = torch.from_numpy(np.stack(history))
        lengths = torch.tensor([len(history)], dtype=torch.int64)
        start = perf_counter()
        with torch.inference_mode():
            logits = self.detector.model(tensor, lengths).numpy()
        self.last_neural_ms = (perf_counter() - start) * 1000
        raw_tree = self.detector.forest.predict_proba(values[None, :])
        neural, tree, probabilities = _combine(logits, raw_tree, self.detector.metadata)
        policy = self.merchant_policies.get(features["merchant_id"], self.policy)
        return self.detector._response(features, float(neural[0]), float(tree[0]), float(probabilities[0]),
                                       float(self.detector.metadata["decision_threshold"]), policy)
