"""Ticket-only integration checks for the ensemble, calibration, and merchant API."""

import json

import numpy as np
import pandas as pd
import pytest

from scalper_detector.cli import main
from scalper_detector.data import LABEL, linked_components, read_events, validate_events
from scalper_detector.decision import MerchantPolicy, decision_fields
from scalper_detector.pipeline import ScalperDetector, TrainConfig, train_model
from scalper_detector.synthetic import generate_synthetic_data
from scalper_detector.tree import PaymentForest


@pytest.fixture(scope="module")
def trained_example(tmp_path_factory):
    orders = generate_synthetic_data(60, seed=19)
    directory = tmp_path_factory.mktemp("trained_payment_detector")
    config = TrainConfig(
        epochs=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        sequence_length=5,
        batch_size=128,
        threads=1,
        seed=19,
    )
    report = train_model(orders, directory, config, data_source="synthetic", verbose=False)
    return orders, directory, report


def test_training_artifacts_and_exact_reload_predictions(trained_example):
    orders, directory, report = trained_example
    expected = {
        "model.pt",
        "tree.json",
        "metadata.json",
        "metrics.json",
        "order_splits.csv",
        "test_orders.csv",
    }
    assert expected <= {path.name for path in directory.iterdir()}
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["format_version"] == 3
    assert metadata["task"] == report["task"] == "ticket_scalping"
    assert json.loads((directory / "metrics.json").read_text()) == report
    for name, temperature in report["calibration_temperatures"].items():
        assert metadata[name] == temperature
        assert np.exp(-3) <= temperature <= np.exp(3)
    assert metadata["decision_threshold"] == report["decision_threshold"]
    assert 1 <= report["best_epoch"] <= 2
    assert len(report["history"]) == 2
    predictions = ScalperDetector.load(directory).predict(orders)
    reloaded = ScalperDetector.load(directory).predict(orders)
    pd.testing.assert_frame_equal(predictions, reloaded, check_exact=True)
    assert len(predictions) == len(orders)
    assert predictions.order_id.is_unique
    assert set(predictions.order_id) == set(orders.order_id)
    for name in ("scalper_probability", "neural_probability", "tree_probability"):
        assert predictions[name].between(0, 1).all()
    np.testing.assert_array_equal(
        predictions.flagged_for_review,
        predictions.scalper_probability >= metadata["decision_threshold"],
    )
    assert predictions.merchant_decision_required.all()
    assert predictions.detection_scope.eq("ticket_scalping").all()
    assert predictions.category.eq("tickets").all()


def test_prediction_ignores_target_annotations(trained_example):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    expected = detector.predict(orders.drop(columns=LABEL))
    for annotation in ("unknown", np.nan, 17):
        annotated = orders.copy()
        annotated[LABEL] = annotation
        pd.testing.assert_frame_equal(detector.predict(annotated), expected, check_exact=True)


def test_evaluation_counts_every_order_and_matches_saved_test(trained_example):
    orders, directory, report = trained_example
    detector = ScalperDetector.load(directory)
    assert detector.evaluate(orders)["metrics"]["n_samples"] == len(orders)
    test_orders = read_events(directory / "test_orders.csv", require_labels=True)
    metrics = detector.evaluate(test_orders)["metrics"]
    assert metrics == report["test"]
    assert metrics["n_samples"] == len(test_orders)
    assert metrics["n_samples"] > test_orders.card_id.nunique()
    predictions = detector.predict(test_orders)
    ordered = validate_events(test_orders)
    counts = (ordered.groupby("card_id").cumcount() + 1).to_numpy()
    np.testing.assert_array_equal(predictions.events_observed, counts)
    np.testing.assert_array_equal(predictions.events_in_window, np.minimum(counts, 5))


def test_evaluation_reuses_saved_inference_batch_size(trained_example):
    _, directory, report = trained_example
    detector = ScalperDetector.load(directory)
    test_orders = read_events(directory / "test_orders.csv", require_labels=True)
    batch_size = detector.metadata["training_config"]["batch_size"]
    assert len(test_orders) > batch_size
    observed_batches = []

    def record_batch(_model, args):
        observed_batches.append(len(args[0]))

    hook = detector.model.register_forward_pre_hook(record_batch)
    try:
        metrics = detector.evaluate(test_orders)["metrics"]
    finally:
        hook.remove()

    expected_batches = [
        min(batch_size, len(test_orders) - start)
        for start in range(0, len(test_orders), batch_size)
    ]
    assert observed_batches == expected_batches
    assert metrics == report["test"]


def test_split_csv_is_order_complete_and_keeps_linked_components_together(trained_example):
    orders, directory, report = trained_example
    splits = pd.read_csv(directory / "order_splits.csv")
    assert splits.order_id.is_unique
    assert set(splits.order_id) == set(orders.order_id)
    assert set(splits.split) == {"train", "validation", "calibration", "test"}
    assert splits.groupby("card_id").split.nunique().eq(1).all()
    assert splits.groupby("split_group").split.nunique().eq(1).all()
    components = linked_components(validate_events(orders))
    assert (
        splits.assign(component=splits.card_id.map(components))
        .groupby("component")
        .split.nunique()
        .eq(1)
        .all()
    )
    for name, part in splits.groupby("split"):
        assert report["split_counts"][name] == {
            "cards": int(part.card_id.nunique()),
            "orders": len(part),
            "linked_components": int(part.split_group.nunique()),
            "positive_orders": int(part[LABEL].sum()),
        }
    test_orders = read_events(directory / "test_orders.csv", require_labels=True)
    assert set(test_orders.order_id) == set(splits.loc[splits.split == "test", "order_id"])
    assert len(test_orders) == report["split_counts"]["test"]["orders"]
    for name in ("validation", "calibration", "test"):
        assert report[name]["n_samples"] == report["split_counts"][name]["orders"]


def test_calibration_is_separate_from_model_selection_and_test(tmp_path, monkeypatch):
    import scalper_detector.pipeline as pipeline

    orders = generate_synthetic_data(60, seed=11)
    logit_partitions, temperature_partitions, threshold_partitions = [], [], []
    original_logits = pipeline._logits
    original_temperature = pipeline.fit_temperature
    original_threshold = pipeline.select_threshold

    def traced_logits(model, dataset, batch_size):
        logit_partitions.append(frozenset(dataset.frame.order_id))
        return original_logits(model, dataset, batch_size)

    def traced_temperature(logits, labels):
        temperature_partitions.append(logit_partitions[-1])
        return original_temperature(logits, labels)

    def traced_threshold(labels, probabilities, **kwargs):
        threshold_partitions.append(logit_partitions[-1])
        assert kwargs["target_precision"] == 0.9
        return original_threshold(labels, probabilities, **kwargs)

    monkeypatch.setattr(pipeline, "_logits", traced_logits)
    monkeypatch.setattr(pipeline, "fit_temperature", traced_temperature)
    monkeypatch.setattr(pipeline, "select_threshold", traced_threshold)
    config = TrainConfig(
        epochs=1, hidden_size=4, num_layers=1, threads=1, target_precision=0.9, seed=11
    )
    report = pipeline.train_model(orders, tmp_path, config, data_source="synthetic", verbose=False)
    splits = pd.read_csv(tmp_path / "order_splits.csv")
    ids = {name: frozenset(part.order_id) for name, part in splits.groupby("split")}
    assert temperature_partitions == [ids["calibration"]] * 3
    assert threshold_partitions == [ids["calibration"]]
    assert logit_partitions == [
        ids["validation"],
        ids["calibration"],
        ids["validation"],
        ids["test"],
    ]
    selection = report["threshold_selection"]
    assert selection["source"] == "calibration"
    assert selection["test_labels_used"] is False
    assert selection["target_precision"] == 0.9
    assert selection["target_met"] == (report["calibration"]["precision"] >= 0.9)


@pytest.mark.parametrize("category", ["sneakers", "electronics", "collectibles"])
def test_non_ticket_orders_rejected_by_batch_and_online_scoring(trained_example, category):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    wrong_scope = orders.iloc[:3].copy()
    wrong_scope.loc[wrong_scope.index[0], "category"] = category
    with pytest.raises(ValueError, match="Only category='tickets'"):
        detector.predict(wrong_scope)
    with pytest.raises(ValueError, match="Only category='tickets'"):
        detector.new_session().score_order(wrong_scope.iloc[0].to_dict())


@pytest.mark.parametrize("override", [{"format_version": 2}, {"task": "marketplace_scalping"}])
def test_legacy_or_wrong_scope_model_rejected_before_weights_load(
    trained_example, tmp_path, override
):
    _, directory, _ = trained_example
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata.update(override)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="ticket-only format 3"):
        ScalperDetector.load(tmp_path)


@pytest.mark.parametrize("mismatch", ["feature_name", "feature_order", "input_size"])
def test_incompatible_feature_contract_rejected_before_weights_load(
    trained_example, tmp_path, mismatch
):
    _, directory, _ = trained_example
    metadata = json.loads((directory / "metadata.json").read_text())
    if mismatch == "feature_name":
        metadata["feature_names"][0] = "retired_feature"
    elif mismatch == "feature_order":
        metadata["feature_names"][:2] = metadata["feature_names"][:2][::-1]
    else:
        metadata["model_config"]["input_size"] += 1
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature schema differs"):
        ScalperDetector.load(tmp_path)


def test_saved_test_retains_unlabeled_population_bin_context(trained_example):
    from scalper_detector.data import engineer_events

    orders, directory, _ = trained_example
    population = engineer_events(orders.drop(columns=LABEL)).set_index("order_id")
    held_out = read_events(directory / "test_orders.csv", require_labels=True).set_index("order_id")
    for window in ("1m", "5m"):
        np.testing.assert_array_equal(
            held_out[f"processor_bin_orders_{window}"],
            population.loc[held_out.index, f"bin_orders_{window}"],
        )


def test_json_forest_preserves_saved_inference_state(trained_example):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    state = json.loads((directory / "tree.json").read_text())
    assert detector.forest.to_dict() == state
    restored = PaymentForest.from_dict(state)
    _, features = detector.preprocessor.transform(orders)
    np.testing.assert_array_equal(
        detector.forest.predict_proba(features), restored.predict_proba(features)
    )
    assert len(state["trees"]) == 48


def test_future_orders_cannot_change_past_order_scores(trained_example):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    prefix = orders.iloc[:100].copy()
    prefix_predictions = detector.predict(prefix).set_index("order_id")
    full_predictions = detector.predict(orders).set_index("order_id").loc[prefix_predictions.index]
    for name in ("scalper_probability", "neural_probability", "tree_probability", "risk_score"):
        np.testing.assert_allclose(
            prefix_predictions[name], full_predictions[name], atol=1e-7, rtol=1e-6
        )
    for name in (
        "confidence_tier",
        "reason_codes",
        "reason_details",
        "recommended_action",
        "events_observed",
    ):
        assert prefix_predictions[name].tolist() == full_predictions[name].tolist()


def test_stateful_order_scoring_matches_batch_prefixes(trained_example):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    prefix = orders.iloc[:100].drop(columns=LABEL)
    expected = detector.predict(prefix).set_index("order_id")
    session = detector.new_session()
    actual = pd.DataFrame(
        [session.score_order(order) for order in prefix.to_dict("records")]
    ).set_index("order_id")
    assert actual.index.tolist() == expected.index.tolist()
    for name in ("scalper_probability", "neural_probability", "tree_probability", "risk_score"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-7, rtol=1e-6)
    for name in (
        "events_observed",
        "events_in_window",
        "confidence_tier",
        "reason_codes",
        "reason_details",
        "recommended_action",
    ):
        assert actual[name].tolist() == expected[name].tolist()


def test_merchant_overrides_change_advice_and_preserve_model_scores(trained_example):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    prefix = orders.iloc[:100]
    baseline = detector.predict(prefix)
    merchant = baseline.merchant_id.iloc[0]
    custom = MerchantPolicy(allow_below=0.01, step_up_below=0.02, cap_below=0.03)
    overridden = detector.predict(prefix, merchant_policies={merchant: custom})
    np.testing.assert_array_equal(baseline.scalper_probability, overridden.scalper_probability)
    for row in overridden.to_dict("records"):
        policy = custom if row["merchant_id"] == merchant else MerchantPolicy()
        expected = decision_fields(
            row["scalper_probability"],
            row["neural_probability"],
            row["tree_probability"],
            {},
            policy,
        )
        assert row["recommended_action"] == expected["recommended_action"]
        assert row["risk_tier"] == expected["risk_tier"]
        assert row["merchant_decision_required"] is True
        assert row["policy_allow_below"] == policy.allow_below
        assert row["policy_step_up_below"] == policy.step_up_below
        assert row["policy_cap_below"] == policy.cap_below


def test_ranked_report_uses_later_cluster_evidence_without_rewriting_scores(
    trained_example, monkeypatch
):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    source = orders.iloc[:3].copy().drop(columns=LABEL)
    source["order_id"] = ["first", "second", "bridge"]
    source["card_id"] = ["card_a", "card_b", "card_b"]
    source["timestamp"] = ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z", "2025-01-01T00:02:00Z"]
    for name in ("shipping_hash", "billing_hash", "email_hash", "device_id"):
        source[name] = ""
    source["shipping_hash"] = ["shipping_a", "shipping_b", "shipping_a"]
    source["event_id"] = "shared_event"
    scores = source[["order_id", "timestamp", "card_id", "event_id"]].copy()
    scores["scalper_probability"] = [0.1, 0.2, 0.9]
    scores["risk_score"] = [10.0, 20.0, 90.0]
    scores["decision_threshold"] = 0.5
    scores["flagged_for_review"] = [False, False, True]
    scores["reason_codes"] = [[], [], ["linked_cluster_order_burst"]]
    monkeypatch.setattr(detector, "predict", lambda *args, **kwargs: scores.copy())
    report = detector.ranked_report(source).set_index("order_id")
    assert set(report.index) == {"first", "second", "bridge"}
    assert report.review_cluster.nunique() == 1
    assert report.review_priority_score.eq(90.0).all()
    assert bool(report.loc["first", "flagged_by_later_cluster_activity"])
    assert bool(report.loc["second", "flagged_by_later_cluster_activity"])
    assert not bool(report.loc["bridge", "flagged_by_later_cluster_activity"])
    assert report.review_trigger_order_id.eq("bridge").all()
    np.testing.assert_array_equal(
        report.loc[scores.order_id, "scalper_probability"], scores.scalper_probability
    )
    np.testing.assert_array_equal(report.loc[scores.order_id, "risk_score"], scores.risk_score)


@pytest.mark.parametrize("earlier_event", ["different_event", ""])
def test_review_does_not_propagate_across_different_or_missing_events(
    trained_example, monkeypatch, earlier_event
):
    orders, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    source = orders.iloc[:2].copy().drop(columns=LABEL)
    source["order_id"] = ["earlier", "trigger"]
    source["card_id"] = "same_card"
    source["event_id"] = [earlier_event, "trigger_event"]
    source["timestamp"] = ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z"]
    scores = source[["order_id", "timestamp", "card_id", "event_id"]].copy()
    scores["scalper_probability"] = [0.1, 0.9]
    scores["risk_score"] = [10.0, 90.0]
    scores["decision_threshold"] = 0.5
    scores["flagged_for_review"] = [False, True]
    scores["reason_codes"] = [[], ["same_event_cross_site_buying"]]
    monkeypatch.setattr(detector, "predict", lambda *args, **kwargs: scores.copy())
    report = detector.ranked_report(source)
    assert report.order_id.tolist() == ["trigger"]
    assert not report.flagged_by_later_cluster_activity.any()


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 0},
        {"sequence_length": -1},
        {"batch_size": 0},
        {"hidden_size": True},
        {"num_layers": 0},
        {"patience": 0},
        {"threads": 0},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": 0.0},
        {"learning_rate": True},
        {"learning_rate": "bad"},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"neural_weight": -0.1},
        {"neural_weight": 1.1},
        {"target_precision": 0},
        {"target_precision": 1.1},
        {"target_precision": True},
        {"target_precision": "0.9"},
    ],
)
def test_invalid_training_config_fails(values):
    with pytest.raises(ValueError):
        TrainConfig(**values).validate()


def test_cli_payment_workflow(tmp_path, capsys):
    training_path = tmp_path / "input" / "orders.csv"
    model_dir = tmp_path / "model"
    predictions_path = tmp_path / "predictions.csv"
    report_path = tmp_path / "review.csv"
    evaluation_path = tmp_path / "evaluation.json"
    main(["generate", "--cards", "60", "--seed", "31", "--output", str(training_path)])
    assert read_events(training_path, require_labels=True).card_id.nunique() == 60
    main(
        [
            "train",
            "--data",
            str(training_path),
            "--output",
            str(model_dir),
            "--epochs",
            "1",
            "--hidden-size",
            "8",
            "--threads",
            "1",
            "--data-source",
            "synthetic",
            "--target-precision",
            "0.9",
        ]
    )
    source = pd.read_csv(training_path)
    source[LABEL] = "stale target annotation"
    source.to_csv(training_path, index=False)
    main(
        [
            "predict",
            "--data",
            str(training_path),
            "--model",
            str(model_dir),
            "--output",
            str(predictions_path),
            "--threshold",
            "0.4",
        ]
    )
    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == len(source)
    assert predictions.scalper_probability.between(0, 1).all()
    assert predictions.decision_threshold.eq(0.4).all()
    assert isinstance(json.loads(predictions.reason_codes.iloc[0]), list)
    main(
        [
            "report",
            "--data",
            str(training_path),
            "--model",
            str(model_dir),
            "--output",
            str(report_path),
            "--threshold",
            "0",
        ]
    )
    assert len(pd.read_csv(report_path)) == len(source)
    main(
        [
            "evaluate",
            "--data",
            str(model_dir / "test_orders.csv"),
            "--model",
            str(model_dir),
            "--output",
            str(evaluation_path),
        ]
    )
    evaluation = json.loads(evaluation_path.read_text())
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert evaluation["metrics"] == metrics["test"]
    assert metrics["threshold_selection"]["target_precision"] == 0.9
    assert metrics["threshold_selection"]["target_met"] == (
        metrics["calibration"]["precision"] >= 0.9
    )
    history_path = tmp_path / "history.csv"
    order_path = tmp_path / "one_order.json"
    inline_path = tmp_path / "inline_score.json"
    source.iloc[:20].to_csv(history_path, index=False)
    one_order = json.loads(source.iloc[20:21].to_json(orient="records"))[0]
    order_path.write_text(json.dumps(one_order, allow_nan=False))
    main(
        [
            "score-order",
            "--model",
            str(model_dir),
            "--order",
            str(order_path),
            "--history",
            str(history_path),
            "--output",
            str(inline_path),
        ]
    )
    inline = json.loads(inline_path.read_text())
    expected = predictions.set_index("order_id").loc[one_order["order_id"]]
    assert inline["scalper_probability"] == pytest.approx(expected.scalper_probability, abs=1e-7)
    assert inline["merchant_decision_required"] is True
    benchmark_path = tmp_path / "benchmark.json"
    main(
        [
            "benchmark",
            "--model",
            str(model_dir),
            "--data",
            str(training_path),
            "--orders",
            "5",
            "--warmup",
            "1",
            "--output",
            str(benchmark_path),
        ]
    )
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark["normal"]["orders_measured"] == 5
    assert benchmark["normal"]["end_to_end_ms"]["p95"] > 0
    burst = benchmark["simulated_drop_day_burst"]
    assert burst["arrival_time_compression"] >= 10.0
    # Timestamp quantization can slightly perturb this five-order trace's rate.
    assert burst["trace_arrival_orders_per_second"] >= 1000.0 * (1 - 1e-3)
    assert benchmark["telum_measured"] is False
    assert "synthetic" in capsys.readouterr().out.lower()


def test_cli_ticket_scenarios_preserve_evidence_and_scope(trained_example, tmp_path, capsys):
    from scalper_detector.synthetic import generate_ticket_scenarios

    _, directory, _ = trained_example
    destination = tmp_path / "ticket_scenarios.csv"
    main(["scenarios", "--model", str(directory), "--output", str(destination)])
    scores = pd.read_csv(destination)
    cases = generate_ticket_scenarios()
    assert set(scores.order_id) == set(cases.order_id)
    assert set(scores.scenario_name) == set(cases.scenario_name)
    assert scores.scenario_name.nunique() >= 4
    assert scores.category.eq("tickets").all()
    assert scores.detection_scope.eq("ticket_scalping").all()
    assert scores.scalper_probability.between(0, 1).all()
    assert scores.merchant_decision_required.all()
    assert "Synthetic examples do not establish real-world accuracy" in capsys.readouterr().out


def test_different_event_fan_preserves_causal_diversity_and_online_scores(trained_example):
    from scalper_detector.data import engineer_events
    from scalper_detector.synthetic import generate_ticket_scenarios

    _, directory, _ = trained_example
    detector = ScalperDetector.load(directory)
    scenarios = generate_ticket_scenarios()
    fan = scenarios.loc[scenarios.scenario_name.eq("different_events_fan")].copy()
    assert len(fan) == 6
    evidence = engineer_events(fan)
    np.testing.assert_array_equal(evidence.cluster_distinct_events_5m, np.arange(1, 7))
    np.testing.assert_array_equal(evidence.cluster_event_orders_5m, np.ones(6))
    np.testing.assert_allclose(evidence.cluster_same_event_order_fraction_5m, 1 / np.arange(1, 7))
    expected = detector.predict(fan)
    early = detector.predict(fan.iloc[:3])
    np.testing.assert_allclose(
        early.scalper_probability, expected.scalper_probability.iloc[:3], atol=1e-7, rtol=1e-6
    )
    session = detector.new_session()
    online = pd.DataFrame([session.score_order(order) for order in fan.to_dict("records")])
    np.testing.assert_allclose(
        online.scalper_probability, expected.scalper_probability, atol=1e-7, rtol=1e-6
    )
    same_event_reasons = {
        "event_ticket_limit_exceeded",
        "same_event_cross_site_buying",
        "linked_cards_same_event",
        "rapid_ticket_repeats",
    }
    assert all(not same_event_reasons.intersection(codes) for codes in expected.reason_codes)


def test_cli_exports_real_saved_neural_model(trained_example, tmp_path, capsys):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    _, directory, _ = trained_example
    destination = tmp_path / "saved_neural.onnx"
    main(["export-onnx", "--model", str(directory), "--output", str(destination)])
    manifest = json.loads(destination.with_suffix(".manifest.json").read_text())
    assert manifest["cpu_numerical_parity"]["verified"] is True
    assert manifest["hardware_validated"] is False
    assert "CPU parity verified: True" in capsys.readouterr().out


def test_cli_invalid_config_writes_no_model(tmp_path, capsys):
    training_path = tmp_path / "orders.csv"
    generate_synthetic_data(20, seed=2).to_csv(training_path, index=False)
    model_dir = tmp_path / "invalid_model"
    with pytest.raises(SystemExit) as error:
        main(["train", "--data", str(training_path), "--output", str(model_dir), "--epochs", "0"])
    assert error.value.code == 2
    assert "epochs must be a positive integer" in capsys.readouterr().err
    assert not model_dir.exists()
