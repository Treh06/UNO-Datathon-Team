"""Use the actual generator while preserving chronology and held-out tactics."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scalper_detector import simulation
from scalper_detector.data import LABEL, engineer_events, read_events
from scalper_detector.pipeline import ScalperDetector, TrainConfig

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "authorization_simulator", ROOT / "generation_and_testing/simulator/generate.py"
)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR)
generate = GENERATOR.generate


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    config = json.loads((ROOT / "generation_and_testing/configs/default.json").read_text())
    config.update(
        {
            "seed": 19,
            "start": "2026-11-05T00:00:00Z",
            "end": "2026-11-05T06:00:00Z",
            "baseline_orders_per_hour": 120,
            "customers_per_split": 120,
            "scalper_ratio": 0.35,
            "split_fractions": {"train": 0.5, "validation": 0.25, "test": 0.25},
            "held_out_tactic": "card_testing",
            "tactic_weights": {
                "shared_address_ring": 0,
                "max_qty_bulk": 0,
                "card_testing": 100,
                "cross_merchant_sweep": 1,
            },
            "drops": [
                {
                    "timestamp": f"2026-11-05T{clock}Z",
                    "orders": 120,
                    "window_seconds": 60,
                    "event_index": event,
                }
                for clock, event in [("01:00:00", 0), ("03:30:00", 1), ("05:00:00", 2)]
            ],
        }
    )
    directory = tmp_path_factory.mktemp("generated") / "simulation"
    manifest = generate(config, directory)
    return directory, manifest


@pytest.fixture
def editable(generated, tmp_path):
    destination = tmp_path / "simulation"
    shutil.copytree(generated[0], destination)
    return destination


def update_manifest(directory, change):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_original_splits_and_unknown_allocation_release_are_preserved(generated):
    directory, original = generated
    frames, truth, manifest = simulation.load_simulation_splits(directory)
    assert manifest == original
    assert set(frames) == set(truth) == {"train", "validation", "test"}
    for split, frame in frames.items():
        assert len(frame) == manifest["splits"][split]["orders"]
        assert set(frame.order_id) == set(truth[split].txn_id)
        assert {"actor_id", "tactic", "txn_id"}.isdisjoint(frame.columns)
        assert frame[LABEL].isin([0, 1]).all()
        assert frame.ticket_limit_known.eq(0).all() and frame.ticket_limit.eq(0).all()
        assert frame.release_time_known.eq(0).all()
        assert frame.timestamp.ge(pd.Timestamp(manifest["splits"][split]["start"])).all()
        assert frame.timestamp.lt(pd.Timestamp(manifest["splits"][split]["end_exclusive"])).all()
        local = frame.drop(columns=[LABEL, "processor_bin_orders_1m", "processor_bin_orders_5m"])
        recomputed = engineer_events(local)
        np.testing.assert_array_equal(frame.processor_bin_orders_1m, recomputed.bin_orders_1m)
        np.testing.assert_array_equal(frame.processor_bin_orders_5m, recomputed.bin_orders_5m)
        assert frame.processor_bin_orders_1m.iloc[0] == 1
    assert not truth["train"].tactic.eq("card_testing").any()
    assert not truth["validation"].tactic.eq("card_testing").any()
    assert truth["test"].tactic.eq("card_testing").any()
    assert set(frames["train"].card_bin) & set(frames["test"].card_bin)


@pytest.mark.parametrize("value", [2, True, None])
def test_manifest_schema_version_is_checked(editable, value):
    update_manifest(editable, lambda manifest: manifest.update(schema_version=value))
    with pytest.raises(ValueError, match="schema_version"):
        simulation.load_simulation_splits(editable)


def test_manifest_actual_row_and_tactic_counts_are_checked(editable):
    def change(manifest):
        manifest["splits"]["train"]["orders"] += 1
        tactic = next(iter(manifest["splits"]["train"]["tactics"]))
        manifest["splits"]["train"]["tactics"][tactic] += 1

    update_manifest(editable, change)
    with pytest.raises(ValueError, match="row counts"):
        simulation.load_simulation_splits(editable)


def test_actual_tactic_counts_cannot_disagree_with_manifest(editable):
    path = editable / "train/ground_truth.jsonl"
    rows = read_jsonl(path)
    normal = next(row for row in rows if row["tactic"] == "normal")
    normal["tactic"] = "legit_bulk"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="tactic counts"):
        simulation.load_simulation_splits(editable)


def test_manifest_time_boundaries_are_contiguous_and_match_config(editable):
    update_manifest(
        editable,
        lambda manifest: manifest["splits"]["validation"].update(start="2026-11-05T03:00:10Z"),
    )
    with pytest.raises(ValueError, match="time boundaries"):
        simulation.load_simulation_splits(editable)


def test_requests_must_stay_within_declared_source_boundaries(editable):
    path = editable / "train/requests.jsonl"
    rows = read_jsonl(path)
    rows[0]["event_ts"] = "2026-11-04T23:59:59.999Z"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="declared boundaries"):
        simulation.load_simulation_splits(editable)


def test_source_request_stream_must_be_chronological(editable):
    path = editable / "validation/requests.jsonl"
    write_jsonl(path, list(reversed(read_jsonl(path))))
    with pytest.raises(ValueError, match="chronological"):
        simulation.load_simulation_splits(editable)


def test_transaction_ids_are_globally_unique_across_source_splits(editable):
    train = read_jsonl(editable / "train/requests.jsonl")
    request_path, truth_path = (
        editable / "test/requests.jsonl",
        editable / "test/ground_truth.jsonl",
    )
    requests, truth = read_jsonl(request_path), read_jsonl(truth_path)
    requests[0]["txn_id"] = truth[0]["txn_id"] = train[0]["txn_id"]
    write_jsonl(request_path, requests)
    write_jsonl(truth_path, truth)
    with pytest.raises(ValueError, match="globally unique"):
        simulation.load_simulation_splits(editable)


def test_actor_isolation_uses_offline_truth_without_creating_features(editable):
    train = read_jsonl(editable / "train/ground_truth.jsonl")
    path = editable / "test/ground_truth.jsonl"
    rows = read_jsonl(path)
    rows[0]["actor_id"] = train[0]["actor_id"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="actors must be isolated"):
        simulation.load_simulation_splits(editable)


def test_card_identity_isolation_is_checked(editable):
    train = read_jsonl(editable / "train/requests.jsonl")
    path = editable / "test/requests.jsonl"
    rows = read_jsonl(path)
    rows[0]["card_token"] = train[0]["card_token"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="card identities must be isolated"):
        simulation.load_simulation_splits(editable)


def test_full_linked_identity_graph_catches_cross_field_address_leakage(editable):
    train = read_jsonl(editable / "train/requests.jsonl")
    path = editable / "test/requests.jsonl"
    rows = read_jsonl(path)
    rows[0]["bill_addr_hash"] = train[0]["ship_addr_hash"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="shared-identity linked components"):
        simulation.load_simulation_splits(editable)


def test_declared_held_out_tactic_cannot_appear_before_test(editable):
    update_manifest(
        editable, lambda manifest: manifest["config"].update(held_out_tactic="cross_merchant_sweep")
    )
    with pytest.raises(ValueError, match="Held-out tactic"):
        simulation.load_simulation_splits(editable)


def test_validation_needs_two_linked_components_per_class():
    frame = pd.DataFrame(
        [
            {
                "order_id": str(index),
                "timestamp": f"2026-01-01T00:00:0{index}Z",
                "card_id": f"card-{index}",
                "card_bin": "900001",
                "merchant_id": "site",
                "category": "tickets",
                "amount": 100,
                LABEL: int(index == 2),
            }
            for index in range(3)
        ]
    )
    with pytest.raises(ValueError, match="at least two"):
        simulation._validation_partitions(frame, 42)


def test_train_only_divides_source_validation_and_preserves_population_context(
    generated, tmp_path, monkeypatch
):
    source, truth, _ = simulation.load_simulation_splits(generated[0])
    captured = {}

    def fake_train(frame, output, config, **kwargs):
        captured.update(kwargs)
        return {"test": {"n_samples": len(kwargs["partitions"]["test"])}}

    class FakeDetector:
        metadata = {"decision_threshold": 0.5}

        def predict(self, frame):
            assert LABEL not in frame and "actor_id" not in frame and "tactic" not in frame
            return pd.DataFrame(
                {
                    "order_id": frame.order_id,
                    "scalper_probability": 0.2,
                    "flagged_for_review": False,
                }
            )

    monkeypatch.setattr(simulation, "train_model", fake_train)
    monkeypatch.setattr(simulation.ScalperDetector, "load", lambda _: FakeDetector())
    report = simulation.train_simulation(generated[0], tmp_path / "model", verbose=False)
    partitions = captured["partitions"]
    assert set(partitions) == {"train", "validation", "calibration", "test"}
    assert captured["data_source"] == "synthetic"
    assert "Preserve simulator" in captured["split_strategy"]
    assert set(partitions["train"].order_id) == set(source["train"].order_id)
    assert set(partitions["test"].order_id) == set(source["test"].order_id)
    assert set(partitions["validation"].order_id) | set(partitions["calibration"].order_id) == set(
        source["validation"].order_id
    )
    assert set(partitions["validation"].split_group).isdisjoint(
        partitions["calibration"].split_group
    )
    component_counts = [
        partitions[name].split_group.nunique() for name in ("validation", "calibration")
    ]
    assert abs(component_counts[0] - component_counts[1]) <= 2
    original_bins = source["validation"].set_index("order_id")[
        ["processor_bin_orders_1m", "processor_bin_orders_5m"]
    ]
    for name in ("validation", "calibration"):
        actual = partitions[name].set_index("order_id")
        pd.testing.assert_frame_equal(
            actual[original_bins.columns], original_bins.reindex(actual.index)
        )
        tactics = truth["validation"].set_index("txn_id").reindex(actual.index).tactic
        assert not tactics.eq("card_testing").any()
    diagnostics = report["simulation_evaluation"]
    assert diagnostics["offline_only"] is True
    assert diagnostics["per_tactic_test"]["card_testing"]["orders"] > 0
    assert diagnostics["per_tactic_test"]["shared_address_ring"] == {
        "orders": 0,
        "flagged": 0,
        "recall": None,
    }
    assert diagnostics["legit_bulk_test"]["orders"] > 0
    assert diagnostics["legit_bulk_test"]["false_positive_rate"] == 0


def test_real_training_saves_offline_tactic_diagnostics_without_model_truth_features(
    generated, tmp_path
):
    output = tmp_path / "model"
    config = TrainConfig(
        epochs=2,
        hidden_size=6,
        num_layers=1,
        batch_size=256,
        threads=1,
        patience=2,
        sequence_length=3,
    )
    report = simulation.train_simulation(generated[0], output, config, verbose=False)
    diagnostics = json.loads((output / "simulation_evaluation.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    assert diagnostics == report["simulation_evaluation"]
    assert (
        diagnostics["source_split_counts"]["test"]["orders"]
        == generated[1]["splits"]["test"]["orders"]
    )
    assert (
        diagnostics["prepared_partition_counts"]["validation"]["orders"]
        + diagnostics["prepared_partition_counts"]["calibration"]["orders"]
        == generated[1]["splits"]["validation"]["orders"]
    )
    assert not any(
        token in name
        for name in metadata["feature_names"]
        for token in ("actor_id", "tactic", "txn_id", LABEL)
    )
    assert diagnostics["per_tactic_test"]["max_qty_bulk"]["recall"] is None
    assert diagnostics["per_tactic_test"]["card_testing"]["orders"] > 0
    assert diagnostics["test_overall"]["precision"] == report["test"]["precision"]
    split_rows = pd.read_csv(output / "order_splits.csv")
    assert split_rows.groupby("split_group").split.nunique().eq(1).all()
    source, truth, _ = simulation.load_simulation_splits(generated[0])
    assert set(split_rows[split_rows.split.eq("train")].order_id) == set(source["train"].order_id)
    assert set(split_rows[split_rows.split.eq("test")].order_id) == set(source["test"].order_id)
    truth_rows = pd.concat(list(truth.values())).set_index("txn_id")
    split_rows["tactic"] = split_rows.order_id.map(truth_rows.tactic)
    assert split_rows[split_rows.tactic.eq("card_testing")].split.eq("test").all()
    repeated = ScalperDetector.load(output).evaluate(
        read_events(output / "test_orders.csv", require_labels=True)
    )
    assert repeated["metrics"]["precision"] == report["test"]["precision"]
