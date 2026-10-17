"""Ticket-only payment risk with independent calibration and event-level evidence."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ActivitySequences, EventPreprocessor, LABEL, engineer_events, linked_components, split_accounts, validate_events
from .decision import MerchantPolicy, decision_fields
from .metrics import binary_metrics, fit_temperature, select_threshold, sigmoid
from .model import ScalperLSTM
from .tree import PaymentForest


@dataclass
class TrainConfig:
    sequence_length: int = 7
    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 0.001
    neural_weight: float = 0.5
    target_precision: float | None = None
    seed: int = 42
    threads: int = 2

    def validate(self) -> None:
        for name in ("sequence_length", "hidden_size", "num_layers", "batch_size", "epochs", "patience", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in ("learning_rate", "dropout", "neural_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= self.dropout < 1 or not 0 <= self.neural_weight <= 1:
            raise ValueError("dropout must be in [0,1) and neural_weight in [0,1].")
        if self.target_precision is not None:
            if (isinstance(self.target_precision, bool) or not isinstance(self.target_precision, Real)
                    or not math.isfinite(self.target_precision) or not 0 < self.target_precision <= 1):
                raise ValueError("target_precision must be in (0,1].")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _logits(model: ScalperLSTM, dataset: ActivitySequences, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for features, lengths, target in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            scores.append(model(features, lengths).numpy())
            labels.append(target.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def _combine(neural_logits: np.ndarray, tree_probabilities: np.ndarray, metadata: dict):
    neural = sigmoid(neural_logits.astype(float) / metadata["neural_temperature"])
    tree = sigmoid(_logit(tree_probabilities) / metadata["tree_temperature"])
    weight = metadata["training_config"]["neural_weight"]
    mixed = weight * neural + (1 - weight) * tree
    return neural, tree, sigmoid(_logit(mixed) / metadata["ensemble_temperature"])


def _false_positive_analysis(frame: pd.DataFrame, labels, probabilities, threshold) -> dict:
    result = {}
    groups = {
        "all_legitimate_orders": np.asarray(labels) == 0,
        "legitimate_bulk_orders": (np.asarray(labels) == 0) & (frame.quantity.to_numpy() >= 4)
                                   & (frame.quantity_available.to_numpy() == 1),
        "legitimate_shared_identity_orders": (np.asarray(labels) == 0) & (frame.linked_cards_count.to_numpy() >= 3),
        "legitimate_family_sized_bookings": (np.asarray(labels) == 0) & frame.quantity.between(2, 6).to_numpy()
                                           & frame.quantity_available.eq(1).to_numpy() & frame.event_id_known.eq(1).to_numpy()
                                           & frame.ticket_limit_known.eq(1).to_numpy()
                                           & frame.cluster_event_limit_excess.eq(0).to_numpy(),
        "legitimate_on_sale_rush_orders": (np.asarray(labels) == 0) & frame.release_time_known.eq(1).to_numpy()
                                         & frame.seconds_since_release.le(120).to_numpy(),
        "legitimate_multi_event_fan_orders": (np.asarray(labels) == 0) & frame.event_id_known.eq(1).to_numpy()
                                             & frame.cluster_distinct_events_5m.ge(3).to_numpy(),
    }
    for name, mask in groups.items():
        count = int(mask.sum())
        flagged = int(((probabilities >= threshold) & mask).sum())
        result[name] = {"orders": count, "flagged": flagged,
                        "false_positive_rate": flagged / count if count else None}
    return result


def train_model(frame: pd.DataFrame, output_dir: str | Path, config: TrainConfig | None = None,
                *, data_source: str = "user_supplied", verbose: bool = True) -> dict:
    """Keep linked cards separate across fit, model selection, calibration, and test."""
    config = config or TrainConfig()
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    population = validate_events(frame, require_labels=True)
    # BINs describe population activity rather than buyer identities. Compute
    # their causal unlabeled background before splitting linked buyers, so a
    # smaller calibration partition does not imply a quieter payment network.
    background = engineer_events(population.drop(columns=[LABEL]))
    population["processor_bin_orders_1m"] = background.bin_orders_1m.to_numpy()
    population["processor_bin_orders_5m"] = background.bin_orders_5m.to_numpy()
    partitions = split_accounts(population, config.seed)
    preprocessor = EventPreprocessor().fit(partitions["train"])
    datasets, features, ordered = {}, {}, {}
    for name, part in partitions.items():
        ordered[name], features[name] = preprocessor.transform(part)
        datasets[name] = ActivitySequences(ordered[name], features[name], config.sequence_length)
    model_config = {"input_size": len(preprocessor.feature_names), "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers, "dropout": config.dropout}
    model = ScalperLSTM(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for values, lengths, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        val_logits, val_labels = _logits(model, datasets["validation"], config.batch_size)
        val_loss = float(np.logaddexp(0, (1 - 2 * val_labels) * val_logits).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(datasets["train"]),
                        "validation_log_loss": val_loss})
        if verbose:
            print(f"Epoch {epoch:02d}: train loss {history[-1]['train_loss']:.4f}, validation loss {val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    train_labels = ordered["train"][LABEL].to_numpy()
    forest = PaymentForest.fit(features["train"], train_labels, seed=config.seed, threads=config.threads)
    cal_logits, cal_labels = _logits(model, datasets["calibration"], config.batch_size)
    cal_tree = forest.predict_proba(features["calibration"])
    metadata = {"format_version": 3, "task": "ticket_scalping", "model_config": model_config, "training_config": asdict(config),
                "data_source": data_source, "neural_temperature": fit_temperature(cal_logits, cal_labels),
                "tree_temperature": fit_temperature(_logit(cal_tree), cal_labels), "ensemble_temperature": 1.0,
                "feature_names": preprocessor.feature_names, "preprocessor": preprocessor.to_dict(),
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__}}
    _, _, cal_probabilities = _combine(cal_logits, cal_tree, metadata)
    metadata["ensemble_temperature"] = fit_temperature(_logit(cal_probabilities), cal_labels)
    _, _, cal_probabilities = _combine(cal_logits, cal_tree, metadata)
    threshold = select_threshold(cal_labels, cal_probabilities, target_precision=config.target_precision)
    metadata["decision_threshold"] = threshold
    val_logits, val_labels = _logits(model, datasets["validation"], config.batch_size)
    _, _, val_probabilities = _combine(val_logits, forest.predict_proba(features["validation"]), metadata)
    test_logits, test_labels = _logits(model, datasets["test"], config.batch_size)
    neural, tree, test_probabilities = _combine(test_logits, forest.predict_proba(features["test"]), metadata)
    test_frame = ordered["test"].iloc[datasets["test"].endpoints].reset_index(drop=True)
    threshold_candidates = sorted(set([0.2, 0.35, 0.5, 0.65, 0.8, threshold]))
    report = {
        "data_source": data_source,
        "task": "ticket_scalping",
        "interpretation": ("Synthetic ticket-payment demo; likelihoods and confidence tiers are not validated on real ticket scalpers."
                           if data_source == "synthetic" else
                           "Ticket-scalper labels and sampling require external validation on representative future ticket orders."),
        "evaluation_unit": "Every held-out order, with only current and earlier payment information",
        "split_strategy": "Approximately 60/15/10/15 train/validation/calibration/test by linked-card component; BIN never links buyers",
        "bin_background": "Causal unlabeled payment-population counters computed before buyer splits; supplied processor counters are honored",
        "split_counts": {name: {"cards": int(part.card_id.nunique()), "orders": len(part),
                                  "linked_components": int(part.split_group.nunique()),
                                  "positive_orders": int(part[LABEL].sum())} for name, part in partitions.items()},
        "best_epoch": best_epoch, "decision_threshold": threshold,
        "calibration_temperatures": {name: metadata[name] for name in ("neural_temperature", "tree_temperature", "ensemble_temperature")},
        "calibration": binary_metrics(cal_labels, cal_probabilities, threshold),
        "threshold_selection": {"source": "calibration", "target_precision": config.target_precision,
                                "target_met": (binary_metrics(cal_labels, cal_probabilities, threshold)["precision"] >= config.target_precision
                                               if config.target_precision is not None else None),
                                "fallback": "Highest attainable calibration precision, then recall, when target cannot be reached",
                                "test_labels_used": False},
        "validation": binary_metrics(val_labels, val_probabilities, threshold),
        "calibration_threshold_tradeoff": [dict(threshold=value, **binary_metrics(cal_labels, cal_probabilities, value))
                                          for value in threshold_candidates],
        "test": binary_metrics(test_labels, test_probabilities, threshold),
        "test_neural": binary_metrics(test_labels, neural, threshold),
        "test_tree": binary_metrics(test_labels, tree, threshold),
        "test_false_positive_analysis": _false_positive_analysis(test_frame, test_labels, test_probabilities, threshold),
        "constant_probability_baseline": binary_metrics(test_labels, np.full(len(test_labels), train_labels.mean())),
        "hardware": {"training": "CPU", "telum_measured": False}, "history": history,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    _write_json(output / "tree.json", forest.to_dict())
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "metrics.json", report)
    pd.concat([part[["order_id", "card_id", "split_group", LABEL]].assign(split=name)
               for name, part in partitions.items()], ignore_index=True).to_csv(output / "order_splits.csv", index=False)
    partitions["test"].to_csv(output / "test_orders.csv", index=False)
    return report


class ScalperDetector:
    """Score ticket orders and return same-event evidence and merchant recommendations."""

    def __init__(self, model: ScalperLSTM, preprocessor: EventPreprocessor, metadata: dict,
                 forest: PaymentForest):
        self.model, self.preprocessor, self.metadata, self.forest = model.eval(), preprocessor, metadata, forest

    @classmethod
    def load(cls, model_dir: str | Path) -> "ScalperDetector":
        directory = Path(model_dir)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 3 or metadata.get("task") != "ticket_scalping":
            raise ValueError("A ticket-only format 3 model is required. Retrain with scalper-detector demo.")
        preprocessor = EventPreprocessor.from_dict(metadata["preprocessor"])
        if (metadata.get("feature_names") != preprocessor.feature_names
                or metadata["model_config"]["input_size"] != len(preprocessor.feature_names)):
            raise ValueError("The saved model's ticket feature schema differs from this code. Retrain with scalper-detector demo.")
        torch.set_num_threads(metadata["training_config"]["threads"])
        model = ScalperLSTM(**metadata["model_config"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        forest = PaymentForest.from_dict(json.loads((directory / "tree.json").read_text(encoding="utf-8")))
        return cls(model, preprocessor, metadata, forest)

    def _response(self, order: dict, neural: float, tree: float, probability: float,
                  cutoff: float, policy: MerchantPolicy | None = None) -> dict:
        length = self.metadata["training_config"]["sequence_length"]
        count = int(order["history_count"])
        response = {name: order[name] for name in ("order_id", "timestamp", "card_id", "merchant_id", "category", "event_id")}
        response.update({"scalper_probability": float(probability), "scalper_likelihood_percent": float(probability * 100),
                         "detection_scope": "ticket_scalping",
                         "neural_probability": float(neural), "tree_probability": float(tree),
                         "flagged_for_review": bool(probability >= cutoff), "decision_threshold": cutoff,
                         "events_observed": count, "events_in_window": min(count, length),
                         "limited_history": count < length, "training_data_source": self.metadata["data_source"],
                         "same_event_sites_5m": int(order["cluster_event_merchants_5m"]),
                         "distinct_events_5m": int(order["cluster_distinct_events_5m"]),
                         "same_event_order_fraction_5m": float(order["cluster_same_event_order_fraction_5m"]),
                         "observed_event_tickets_24h": int(order["cluster_event_tickets_24h"]),
                         "event_quantity_coverage_1h": float(order["cluster_event_quantity_coverage_1h"]),
                         "event_ticket_limit": int(order["ticket_limit"]) if order["ticket_limit_known"] else None})
        response.update(decision_fields(probability, neural, tree, order, policy))
        return response

    def predict(self, frame: pd.DataFrame, *, threshold: float | None = None, batch_size: int = 256,
                policy: MerchantPolicy | None = None, merchant_policies: dict | None = None) -> pd.DataFrame:
        cutoff = self.metadata["decision_threshold"] if threshold is None else threshold
        if not isinstance(cutoff, Real) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("threshold must be in [0,1].")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        df, values = self.preprocessor.transform(frame.drop(columns=[LABEL], errors="ignore"))
        dataset = ActivitySequences(df, values, self.metadata["training_config"]["sequence_length"])
        logits, _ = _logits(self.model, dataset, batch_size)
        neural, tree, probabilities = _combine(logits, self.forest.predict_proba(values[dataset.endpoints]), self.metadata)
        rows = df.iloc[dataset.endpoints].to_dict("records")
        return pd.DataFrame([self._response(row, n, t, p, float(cutoff),
                                           (merchant_policies or {}).get(row["merchant_id"], policy))
                             for row, n, t, p in zip(rows, neural, tree, probabilities)])

    def evaluate(self, frame: pd.DataFrame) -> dict:
        df = validate_events(frame, require_labels=True)
        predictions = self.predict(df)
        labels = df.set_index("order_id")[LABEL].reindex(predictions.order_id).to_numpy()
        return {"data_source": self.metadata["data_source"], "decision_threshold": self.metadata["decision_threshold"],
                "metrics": binary_metrics(labels, predictions.scalper_probability.to_numpy(), self.metadata["decision_threshold"])}

    def ranked_report(self, frame: pd.DataFrame, *, threshold: float | None = None,
                      policy: MerchantPolicy | None = None, merchant_policies: dict | None = None) -> pd.DataFrame:
        """Prioritize the same event within linked cards; preserve order-time scores."""
        scores = self.predict(frame, threshold=threshold, policy=policy, merchant_policies=merchant_policies)
        components = linked_components(validate_events(frame.drop(columns=[LABEL], errors="ignore")))
        scores["review_cluster"] = scores.card_id.map(components)
        scores["review_group"] = [json.dumps([cluster, event or None, None if event else order_id])
                                  for cluster, event, order_id in zip(scores.review_cluster, scores.event_id, scores.order_id)]
        scores["review_priority_score"] = scores.groupby("review_group").scalper_probability.transform("max") * 100
        triggers = scores.loc[scores.groupby("review_group").scalper_probability.idxmax()].set_index("review_group")
        scores["review_trigger_order_id"] = scores.review_group.map(triggers.order_id)
        scores["review_trigger_timestamp"] = scores.review_group.map(triggers.timestamp)
        scores["review_cluster_reason_codes"] = scores.review_group.map(triggers.reason_codes)
        scores["flagged_by_cluster_review"] = (~scores.flagged_for_review) & (
            scores.review_priority_score >= scores.decision_threshold * 100)
        scores["flagged_by_later_cluster_activity"] = scores.flagged_by_cluster_review & (
            scores.review_trigger_timestamp > scores.timestamp)
        scores["review_evidence_as_of"] = scores.timestamp.max()
        selected = scores.review_priority_score >= scores.decision_threshold * 100
        report = scores[selected].sort_values(["review_priority_score", "risk_score"], ascending=False, kind="stable")
        report.insert(0, "review_rank", np.arange(1, len(report) + 1))
        return report.reset_index(drop=True)

    def new_session(self, *, policy: MerchantPolicy | None = None, merchant_policies: dict | None = None):
        from .service import OrderScoringSession
        return OrderScoringSession(self, policy=policy, merchant_policies=merchant_policies)
