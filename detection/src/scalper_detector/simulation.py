"""Offline integration for the simulator's chronological, entity-isolated splits.

Ground truth is read only by these training/evaluation helpers. Authorization
requests remain label-free, and actor/tactic identifiers never become features.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .authorizations import read_authorizations, read_ground_truth
from .data import LABEL, engineer_events, linked_components
from .metrics import binary_metrics
from .pipeline import ScalperDetector, TrainConfig, train_model

SOURCE_SPLITS = ("train", "validation", "test")
SCALPER_TACTICS = ("shared_address_ring", "max_qty_bulk", "card_testing", "cross_merchant_sweep")
TACTICS = (*SCALPER_TACTICS, "normal", "legit_bulk")
SPLIT_STRATEGY = (
    "Preserve simulator chronological train/test populations and held-out tactics; "
    "split only source validation 50/50 by linked-card component for early stopping/calibration; "
    "reset causal BIN background between original source populations"
)


def _timestamp(value, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


def _validate_manifest(manifest: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Simulation manifest requires schema_version 1.")
    config, splits = manifest.get("config"), manifest.get("splits")
    if not isinstance(config, dict) or not isinstance(splits, dict) or set(splits) != set(SOURCE_SPLITS):
        raise ValueError("Simulation manifest must contain config and train/validation/test splits.")
    start = _timestamp(config.get("start"), "config.start")
    end = _timestamp(config.get("end"), "config.end")
    if start >= end:
        raise ValueError("Simulation config.start must precede config.end.")
    fractions = config.get("split_fractions")
    if (not isinstance(fractions, dict) or set(fractions) != set(SOURCE_SPLITS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in fractions.values())
            or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9)):
        raise ValueError("Manifest split_fractions must specify positive train/validation/test fractions summing to one.")
    held_out = config.get("held_out_tactic")
    if held_out is not None and held_out not in SCALPER_TACTICS:
        raise ValueError("Manifest held_out_tactic is unknown.")
    boundaries = {}
    previous_end, expected_start = None, start
    # The generator serializes millisecond timestamps after calculating its
    # boundaries, so permit at most one millisecond of serialization rounding.
    tolerance = timedelta(milliseconds=1)
    for name in SOURCE_SPLITS:
        summary = splits[name]
        if not isinstance(summary, dict):
            raise ValueError(f"Manifest {name} summary must be an object.")
        left = _timestamp(summary.get("start"), f"{name}.start")
        right = _timestamp(summary.get("end_exclusive"), f"{name}.end_exclusive")
        expected_end = end if name == "test" else expected_start + (end - start) * fractions[name]
        if (left >= right or (previous_end is not None and left != previous_end)
                or abs(left - expected_start) > tolerance or abs(right - expected_end) > tolerance):
            raise ValueError(f"Manifest {name} time boundaries do not match contiguous configured splits.")
        orders = summary.get("orders")
        tactics = summary.get("tactics")
        if type(orders) is not int or orders < 1:
            raise ValueError(f"Manifest {name}.orders must be a positive integer.")
        if (not isinstance(tactics, dict) or set(tactics) - set(TACTICS)
                or any(type(count) is not int or count < 0 for count in tactics.values())
                or sum(tactics.values()) != orders):
            raise ValueError(f"Manifest {name} tactic counts must be nonnegative integers summing to orders.")
        boundaries[name] = (left, right)
        previous_end, expected_start = right, expected_end
    return boundaries


def _verify_request_order(path: Path, split: str) -> None:
    previous = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = _timestamp(record.get("event_ts"), f"{split} request event_ts")
            if previous is not None and timestamp < previous:
                raise ValueError(f"Simulation {split} requests must be in chronological order.")
            previous = timestamp


def load_simulation_splits(simulation_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Validate the original three populations and attach causal BIN background.

    Returns canonical labeled orders, separate offline ground truth, and the
    source manifest. No order is moved between these original populations.
    """
    directory = Path(simulation_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    boundaries = _validate_manifest(manifest)
    frames, truths = {}, {}
    seen_orders, seen_cards, seen_actors = set(), set(), set()
    for split in SOURCE_SPLITS:
        requests = directory / split / "requests.jsonl"
        ground_truth = directory / split / "ground_truth.jsonl"
        frame = read_authorizations(requests, ground_truth_path=ground_truth)
        truth = read_ground_truth(ground_truth)
        _verify_request_order(requests, split)
        summary = manifest["splits"][split]
        if len(frame) != summary["orders"] or len(truth) != summary["orders"]:
            raise ValueError(f"Simulation {split} row counts disagree with its manifest.")
        actual_tactics = Counter(truth.tactic)
        expected_tactics = {name: count for name, count in summary["tactics"].items() if count}
        if dict(actual_tactics) != expected_tactics:
            raise ValueError(f"Simulation {split} tactic counts disagree with its manifest.")
        if not truth.tactic.isin(TACTICS).all() or not truth[LABEL].eq(truth.tactic.isin(SCALPER_TACTICS).astype(int)).all():
            raise ValueError(f"Simulation {split} tactic/label definitions disagree.")
        held_out = manifest["config"].get("held_out_tactic")
        if split != "test" and held_out is not None and truth.tactic.eq(held_out).any():
            raise ValueError(f"Held-out tactic {held_out} appears in simulation {split}.")
        start, end = boundaries[split]
        if not (frame.timestamp.ge(start) & frame.timestamp.lt(end)).all():
            raise ValueError(f"Simulation {split} order timestamps fall outside its declared boundaries.")
        orders, cards, actors = set(frame.order_id), set(frame.card_id), set(truth.actor_id)
        if seen_orders & orders:
            raise ValueError("Simulation transaction IDs must be globally unique across splits.")
        if seen_cards & cards:
            raise ValueError("Simulation card identities must be isolated across splits.")
        if seen_actors & actors:
            raise ValueError("Simulation actors must be isolated across splits.")
        seen_orders.update(orders)
        seen_cards.update(cards)
        seen_actors.update(actors)
        # Reset state for each independently generated source population. These
        # background features use requests, never actor/tactic/target labels.
        background = engineer_events(frame.drop(columns=[LABEL]))
        frame["processor_bin_orders_1m"] = background.bin_orders_1m.to_numpy()
        frame["processor_bin_orders_5m"] = background.bin_orders_5m.to_numpy()
        frames[split], truths[split] = frame, truth
    combined = pd.concat([frame.assign(source_split=split) for split, frame in frames.items()], ignore_index=True)
    components = linked_components(combined.drop(columns=[LABEL]))
    combined["component"] = combined.card_id.map(components)
    if combined.groupby("component").source_split.nunique().gt(1).any():
        raise ValueError("Simulation shared-identity linked components must be isolated across source splits.")
    return frames, truths, manifest


def _validation_partitions(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    components = linked_components(frame)
    grouped = frame.assign(split_group=frame.card_id.map(components))
    labels = grouped.groupby("split_group", sort=True)[LABEL].max()
    rng = np.random.default_rng(seed)
    selected = {"validation": [], "calibration": []}
    for label in (0, 1):
        ids = labels.index[labels.eq(label)].to_numpy().copy()
        if len(ids) < 2:
            raise ValueError("Source validation requires at least two independent linked-card components per class for validation/calibration.")
        rng.shuffle(ids)
        middle = len(ids) // 2
        selected["validation"].extend(ids[:middle])
        selected["calibration"].extend(ids[middle:])
    return {name: grouped[grouped.split_group.isin(ids)].reset_index(drop=True) for name, ids in selected.items()}


def _counts(frame: pd.DataFrame) -> dict:
    return {"orders": len(frame), "cards": int(frame.card_id.nunique()),
            "linked_components": len(set(linked_components(frame).values())),
            "positive_orders": int(frame[LABEL].sum())}


def train_simulation(simulation_dir: str | Path, output_dir: str | Path,
                     config: TrainConfig | None = None, *, verbose: bool = True) -> dict:
    """Train without redistributing the simulator's chronological test/tactics."""
    config = config or TrainConfig()
    config.validate()
    source, truths, manifest = load_simulation_splits(simulation_dir)
    divided = _validation_partitions(source["validation"], config.seed)
    partitions = {"train": source["train"], "validation": divided["validation"],
                  "calibration": divided["calibration"], "test": source["test"]}
    report = train_model(pd.concat(list(source.values()), ignore_index=True), output_dir, config,
                         data_source="synthetic", verbose=verbose, partitions=partitions,
                         split_strategy=SPLIT_STRATEGY)
    detector = ScalperDetector.load(output_dir)
    predictions = detector.predict(partitions["test"].drop(columns=[LABEL]))
    truth = truths["test"].set_index("txn_id").reindex(predictions.order_id)
    flagged = predictions.flagged_for_review.to_numpy(dtype=bool)
    per_tactic = {}
    for tactic in SCALPER_TACTICS:
        mask = truth.tactic.eq(tactic).to_numpy()
        count = int(mask.sum())
        count_flagged = int((flagged & mask).sum())
        per_tactic[tactic] = {"orders": count, "flagged": count_flagged,
                              "recall": count_flagged / count if count else None}
    bulk = truth.tactic.eq("legit_bulk").to_numpy()
    bulk_count, bulk_flagged = int(bulk.sum()), int((flagged & bulk).sum())
    diagnostics = {
        "offline_only": True, "generator_manifest": manifest,
        "split_strategy": SPLIT_STRATEGY,
        "source_split_counts": {name: dict(_counts(frame), actors=int(truths[name].actor_id.nunique()),
                                          tactics=dict(Counter(truths[name].tactic))) for name, frame in source.items()},
        "prepared_partition_counts": {name: _counts(frame) for name, frame in partitions.items()},
        "per_tactic_test": per_tactic,
        "legit_bulk_test": {"orders": bulk_count, "flagged": bulk_flagged,
                            "false_positive_rate": bulk_flagged / bulk_count if bulk_count else None},
        "test_overall": binary_metrics(truth[LABEL].to_numpy(), predictions.scalper_probability.to_numpy(),
                                       detector.metadata["decision_threshold"]),
        "interpretation": "Offline simulated campaign labels; per-tactic recall and legitimate-bulk false positives do not establish real-world accuracy.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "simulation_evaluation.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(report, simulation_evaluation=diagnostics)
