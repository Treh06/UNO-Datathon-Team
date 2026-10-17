"""Train a tabular forest and persist its inference state as portable JSON."""

from __future__ import annotations

import numpy as np


class PaymentForest:
    def __init__(self, trees: list[dict]):
        self.trees = trees

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, *, seed: int = 42,
            threads: int = 2) -> "PaymentForest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=48, max_depth=8, min_samples_leaf=8,
                                       n_jobs=threads, random_state=seed)
        model.fit(features, labels)
        trees = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            values = tree.value[:, 0, :]
            positive = np.divide(values[:, 1], values.sum(axis=1),
                                 out=np.zeros(len(values)), where=values.sum(axis=1) > 0)
            trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                          "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                          "probability": positive.tolist()})
        result = cls(trees)
        # The JSON inference implementation must match the trained forest.
        np.testing.assert_allclose(result.predict_proba(features), model.predict_proba(features)[:, 1], atol=1e-12)
        return result

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        scores = np.zeros(len(values), dtype=float)
        for tree in self.trees:
            left = np.asarray(tree["left"])
            right = np.asarray(tree["right"])
            columns = np.asarray(tree["feature"])
            thresholds = np.asarray(tree["threshold"])
            probabilities = np.asarray(tree["probability"])
            nodes = np.zeros(len(values), dtype=int)
            active = left[nodes] != -1
            while active.any():
                rows = np.flatnonzero(active)
                current = nodes[rows]
                nodes[rows] = np.where(values[rows, columns[current]] <= thresholds[current],
                                       left[current], right[current])
                active = left[nodes] != -1
            scores += probabilities[nodes]
        return scores / len(self.trees)

    def to_dict(self) -> dict:
        return {"format": "payment_forest_v1", "trees": self.trees}

    @classmethod
    def from_dict(cls, state: dict) -> "PaymentForest":
        if state.get("format") != "payment_forest_v1" or not state.get("trees"):
            raise ValueError("Unsupported or empty tree ensemble state.")
        return cls(state["trees"])
