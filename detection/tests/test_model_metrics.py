"""Tests for metric edge cases and the LSTM's variable-length contract."""

import json

import numpy as np
import pytest
import torch

from scalper_detector.metrics import binary_metrics, fit_temperature, select_threshold, sigmoid
from scalper_detector.model import ScalperLSTM


def test_ranking_metrics_and_confusion_matrix():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert metrics["f1"] == pytest.approx(2 / 3)
    json.dumps(metrics, allow_nan=False)


def test_ranking_ties_are_independent_of_row_order():
    for labels in ([0, 1, 0, 1], [1, 1, 0, 0]):
        metrics = binary_metrics(labels, [0.7] * 4)
        assert metrics["roc_auc"] == pytest.approx(0.5)
        assert metrics["average_precision"] == pytest.approx(0.5)


def test_one_class_metrics_are_explicit_and_json_safe():
    negative_metrics = binary_metrics([0, 0], [0.0, 1.0])
    assert negative_metrics["roc_auc"] is None
    assert negative_metrics["average_precision"] is None
    assert negative_metrics["recall"] == 0.0
    positive_metrics = binary_metrics([1, 1], [0.0, 1.0])
    assert positive_metrics["roc_auc"] is None
    assert positive_metrics["average_precision"] == 1.0
    json.dumps([negative_metrics, positive_metrics], allow_nan=False)


def test_threshold_maximizes_f1_and_prefers_higher_ties():
    labels = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.3, 0.3, 0.6, 0.9])
    threshold = select_threshold(labels, probabilities)
    candidates = np.unique(np.r_[probabilities, 0.5])
    expected = max(
        candidates,
        key=lambda candidate: (binary_metrics(labels, probabilities, candidate)["f1"], candidate),
    )
    assert threshold == expected
    assert select_threshold([0, 1], [0.1, 0.9]) == 0.9


def test_precision_target_maximizes_recall_and_uses_higher_cutoff_ties():
    labels = [1, 0, 1, 1, 0]
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.1]
    assert select_threshold(labels, probabilities, target_precision=0.8) == 0.9
    assert select_threshold(labels, probabilities, target_precision=0.75) == 0.6
    assert select_threshold(labels, probabilities, target_precision=0.6) == 0.6
    assert select_threshold(labels, probabilities, target_precision=None) == select_threshold(
        labels, probabilities
    )


def test_unattainable_precision_target_falls_back_to_best_precision_then_recall():
    labels, probabilities = [0, 1, 1, 0], [0.9, 0.8, 0.7, 0.1]
    threshold = select_threshold(labels, probabilities, target_precision=0.95)
    assert threshold == 0.7
    result = binary_metrics(labels, probabilities, threshold)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["precision"] < 0.95


@pytest.mark.parametrize("target", [0, -0.1, 1.1, True, False, float("nan"), float("inf"), "0.9"])
def test_invalid_precision_target_fails(target):
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.1, 0.9], target_precision=target)


def test_sigmoid_extremes_and_temperature_improvement():
    probabilities = sigmoid([-1000.0, 0.0, 1000.0])
    np.testing.assert_array_equal(probabilities, [0.0, 0.5, 1.0])
    assert sigmoid(0.0).shape == ()
    logits = np.array([10.0, 10.0, -10.0, -10.0])
    labels = np.array([1, 0, 0, 1])
    temperature = fit_temperature(logits, labels)
    assert np.exp(-3) <= temperature <= np.exp(3)
    assert (
        binary_metrics(labels, sigmoid(logits / temperature))["log_loss"]
        < binary_metrics(labels, sigmoid(logits))["log_loss"]
    )
    assert fit_temperature(np.zeros(4), labels) == 1.0


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([], []),
        ([0, 1], [0.5]),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [-0.1, 0.8]),
        ([[0, 1]], [[0.2, 0.8]]),
    ],
)
def test_invalid_metric_inputs_fail(labels, probabilities):
    with pytest.raises(ValueError):
        binary_metrics(labels, probabilities)
    with pytest.raises(ValueError):
        select_threshold(labels, probabilities)


def test_lstm_ignores_right_padding_and_preserves_batch_order():
    torch.manual_seed(7)
    model = ScalperLSTM(input_size=3, hidden_size=8, num_layers=1).eval()
    features = torch.randn(3, 6, 3)
    lengths = torch.tensor([2, 6, 4])
    modified = features.clone()
    for index, length in enumerate(lengths):
        modified[index, length:] = 1000.0
    with torch.no_grad():
        logits = model(features, lengths)
        padded_logits = model(modified, lengths)
        individual = torch.cat(
            [
                model(features[index : index + 1, :length], lengths[index : index + 1])
                for index, length in enumerate(lengths)
            ]
        )
    assert logits.shape == (3,)
    torch.testing.assert_close(logits, padded_logits)
    torch.testing.assert_close(logits, individual)


def test_lstm_gradients_exclude_padding():
    model = ScalperLSTM(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    features = torch.randn(2, 4, 2, requires_grad=True)
    model(features, torch.tensor([1, 4])).sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad[0, 1:]).item() == 0
    assert torch.count_nonzero(features.grad[0, :1]).item() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": 0},
        {"input_size": True},
        {"input_size": 2, "num_layers": 0},
        {"input_size": 2, "dropout": np.nan},
        {"input_size": 2, "dropout": 1.0},
    ],
)
def test_invalid_model_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        ScalperLSTM(**kwargs)


def test_invalid_sequence_lengths_fail():
    model = ScalperLSTM(input_size=2)
    features = torch.zeros(2, 4, 2)
    for lengths in (
        torch.tensor([0, 4]),
        torch.tensor([1, 5]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1]),
    ):
        with pytest.raises(ValueError):
            model(features, lengths)
