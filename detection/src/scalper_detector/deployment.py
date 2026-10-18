"""Export the neural branch, with explicit CPU checks and deployment limits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _PaddedExportLSTM(nn.Module):
    """Equivalent unidirectional LSTM using fixed padding and a last-event gather."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.model.lstm_backend == "unfused":
            last_valid = self.model._unfused_hidden(features, lengths)
        else:
            outputs, _ = self.model.lstm(features)
            indices = (lengths - 1).reshape(-1, 1, 1).expand(-1, 1, outputs.shape[2])
            last_valid = outputs.gather(1, indices).squeeze(1)
        return self.model.classifier(self.model.dropout(last_valid)).squeeze(-1)


def export_onnx(model_dir: str | Path, output_path: str | Path) -> dict:
    """Export fixed-time, dynamic-batch neural logits and a deployment manifest.

    Preprocessing, tree inference, ensembling, calibration, explanations, and
    merchant policy remain in the CPU application. ONNX validity or CPU parity
    does not establish IBM zDLC operator support or accelerator compatibility.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ValueError(
            "ONNX export requires optional dependencies; install with pip install '.[deploy]'."
        ) from exc
    from .pipeline import ScalperDetector

    detector = ScalperDetector.load(model_dir)
    sequence_length = detector.metadata["training_config"]["sequence_length"]
    input_size = detector.metadata["model_config"]["input_size"]
    model = detector.model.cpu().eval()
    wrapper = _PaddedExportLSTM(model).eval()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    example_features = torch.zeros(2, sequence_length, input_size, dtype=torch.float32)
    example_lengths = torch.tensor(
        [sequence_length, max(1, sequence_length // 2)], dtype=torch.int64
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example_features, example_lengths),
            str(destination),
            input_names=["features", "lengths"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "lengths": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    onnx_model = onnx.load(str(destination))
    onnx.checker.check_model(onnx_model)
    parity: dict = {
        "verified": False,
        "runtime": "onnxruntime CPUExecutionProvider",
        "cases": [],
        "atol": 1e-5,
        "rtol": 1e-4,
    }
    try:
        import onnxruntime as ort
    except ImportError:
        parity["reason"] = "onnxruntime is not installed; CPU numerical parity was not tested."
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(destination), sess_options=options, providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(17)
        cases = (
            [1],
            [sequence_length, 1],
            [max(1, sequence_length // 2), sequence_length, 1, max(1, sequence_length - 1)],
        )
        for lengths in cases:
            lengths_array = np.asarray(lengths, dtype=np.int64)
            values = rng.normal(size=(len(lengths), sequence_length, input_size)).astype(np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(values), torch.from_numpy(lengths_array)).numpy()
            actual = session.run(["logits"], {"features": values, "lengths": lengths_array})[0]
            passed = bool(np.allclose(actual, expected, atol=parity["atol"], rtol=parity["rtol"]))
            parity["cases"].append(
                {
                    "batch_size": len(lengths),
                    "lengths": lengths,
                    "max_absolute_error": float(np.max(np.abs(actual - expected))),
                    "passed": passed,
                }
            )
        parity["verified"] = all(case["passed"] for case in parity["cases"])

    manifest = {
        "format_version": 1,
        "onnx_file": destination.name,
        "exported_branch": "neural_lstm_only",
        "lstm_backend": model.lstm_backend,
        "detection_scope": detector.metadata.get("task", "ticket_scalping"),
        "output_semantics": "Uncalibrated neural logits, not final ensemble probabilities or merchant decisions",
        "opset": 17,
        "feature_order": detector.metadata.get("feature_names", []),
        "preprocessing_contract": "Apply metadata.json from the same saved model before invoking the neural graph.",
        "inputs": {
            "features": {"dtype": "float32", "shape": ["batch", sequence_length, input_size]},
            "lengths": {"dtype": "int64", "shape": ["batch"], "valid_range": [1, sequence_length]},
        },
        "outputs": {"logits": {"dtype": "float32", "shape": ["batch"]}},
        "onnx_checker_passed": True,
        "cpu_numerical_parity": parity,
        "operators": sorted({node.op_type for node in onnx_model.graph.node}),
        "remains_on_cpu": [
            "token linkage and causal features",
            "preprocessing",
            "tree branch",
            "ensemble calibration",
            "reason generation",
            "merchant policy",
        ],
        "target_validation_required": (
            "Verify every operator, tensor shape, supported opset, compiler version, and target IBM Z hardware generation "
            "with IBM zDLC before deployment. ONNX export and CPU parity establish no accelerator compatibility or performance."
        ),
        "hardware_validated": False,
        "versions": {"torch": str(torch.__version__), "onnx": str(onnx.__version__)},
    }
    if parity["verified"]:
        manifest["versions"]["onnxruntime"] = str(ort.__version__)
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if parity["cases"] and not parity["verified"]:
        raise RuntimeError(
            f"Exported ONNX logits failed CPU parity checks; inspect {manifest_path}."
        )
    return manifest
