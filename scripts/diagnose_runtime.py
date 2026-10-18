#!/usr/bin/env python3
"""Isolate native dependency failures without importing them in this process.

Run this file with the same Python interpreter used for training. Every native
probe runs in a fresh subprocess, so an import crash cannot kill this reporter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_THREADING_LAYER",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "SCALPER_SAFE_MODE",
    "SCALPER_CPU_THREADS",
)
SAFE_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OPENBLAS_DEFAULT_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
PACKAGES = ("scalper-detector", "numpy", "pandas", "torch", "scikit-learn", "scipy")
PROBES = {
    "numpy": """
        print("Importing NumPy", flush=True)
        import numpy as np
        print("NumPy matrix multiplication", flush=True)
        values = np.arange(1024, dtype=np.float64).reshape(32, 32)
        result = values @ values.T
        assert np.isfinite(result).all()
        print("NumPy matrix multiplication passed", flush=True)
    """,
    "pandas": """
        print("Importing pandas", flush=True)
        import pandas as pd
        print("pandas numeric conversion and NumPy conversion", flush=True)
        frame = pd.DataFrame({"amount": ["1.25", "2.50"], "quantity": [1, 2]})
        frame["amount"] = pd.to_numeric(frame["amount"])
        values = frame.to_numpy(dtype="float32")
        assert values.shape == (2, 2)
        print("pandas conversion passed", flush=True)
    """,
    "torch": """
        print("Importing PyTorch", flush=True)
        import torch
        print("Reading PyTorch build configuration", flush=True)
        print(torch.__config__.show(), flush=True)
        configure_torch(torch)
        print("PyTorch CPU matrix multiplication", flush=True)
        values = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
        result = values @ values.T
        assert bool(torch.isfinite(result).all())
        print("PyTorch CPU matrix multiplication passed", flush=True)
    """,
    "lstm": """
        print("Importing PyTorch for the repository LSTM", flush=True)
        import torch
        configure_torch(torch)
        print("Importing the repository ScalperLSTM", flush=True)
        from scalper_detector.model import ScalperLSTM
        print("Constructing a two-layer ScalperLSTM and Adam optimizer", flush=True)
        torch.manual_seed(42)
        backend = "unfused" if DIAGNOSTIC_SAFE_MODE else "native"
        model = ScalperLSTM(input_size=8, hidden_size=16, num_layers=2, dropout=0.2,
                            lstm_backend=backend)
        print("ScalperLSTM backend: " + backend, flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        features = torch.randn(4, 6, 8)
        lengths = torch.tensor([6, 5, 3, 1], dtype=torch.long)
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        print("ScalperLSTM forward pass", flush=True)
        logits = model(features, lengths)
        print("ScalperLSTM loss calculation", flush=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        print("ScalperLSTM backward pass", flush=True)
        loss.backward()
        print("ScalperLSTM Adam optimizer step", flush=True)
        optimizer.step()
        assert bool(torch.isfinite(loss))
        print("ScalperLSTM forward, backward, and optimizer passed", flush=True)
    """,
    "sklearn": """
        print("Importing scikit-learn RandomForestClassifier", flush=True)
        from sklearn.ensemble import RandomForestClassifier
        features = [[float(i), float(i % 3)] for i in range(24)]
        targets = [i % 2 for i in range(24)]
        model = RandomForestClassifier(n_estimators=4, max_depth=3, n_jobs=1,
                                       random_state=42)
        print("RandomForestClassifier single-thread fit", flush=True)
        model.fit(features, targets)
        print("RandomForestClassifier prediction", flush=True)
        assert len(model.predict_proba(features)) == len(features)
        print("RandomForestClassifier fit and prediction passed", flush=True)
    """,
}


def installed_versions() -> dict[str, str | None]:
    """Read installed distribution metadata without importing native packages."""
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_information(environment: dict[str, str]) -> dict:
    """Collect platform details and an explicit allowlist of environment values."""
    os_release = {}
    if hasattr(platform, "freedesktop_os_release"):
        try:
            release = platform.freedesktop_os_release()
            os_release = {
                key: release[key]
                for key in ("NAME", "ID", "ID_LIKE", "VERSION", "VERSION_ID", "PRETTY_NAME")
                if key in release
            }
        except OSError:
            pass
    return {
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "os_release": os_release,
        "packages": installed_versions(),
        "thread_environment": {key: environment.get(key) for key in THREAD_ENVIRONMENT},
    }


def probe_source(name: str, *, safe: bool) -> str:
    """Build a child program that locates repository source independently of cwd."""
    setup = f"""
import sys
sys.path.insert(0, {str(REPOSITORY_ROOT / "detection" / "src")!r})
DIAGNOSTIC_SAFE_MODE = {safe!r}

def configure_torch(torch):
    if DIAGNOSTIC_SAFE_MODE:
        print("Safe mode: one PyTorch CPU thread; MKLDNN disabled", flush=True)
        from scalper_detector.runtime import configure_torch_runtime
        configure_torch_runtime(1)
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
"""
    return setup + "\n" + textwrap.dedent(PROBES[name])


def classify_returncode(returncode: int) -> dict:
    """Distinguish successful exits, native crashes, and ordinary Python errors."""
    if returncode == 0:
        return {"status": "passed", "returncode": returncode}
    windows_errors = {
        0xC0000005: "ACCESS_VIOLATION",
        0xC000001D: "ILLEGAL_INSTRUCTION",
        0xC0000094: "INTEGER_DIVIDE_BY_ZERO",
        0xC00000FD: "STACK_OVERFLOW",
    }
    windows_error = windows_errors.get(returncode & 0xFFFFFFFF)
    if windows_error:
        return {
            "status": "signal",
            "returncode": returncode,
            "signal": windows_error,
        }
    if returncode < 0:
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = f"SIGNAL_{-returncode}"
        return {"status": "signal", "returncode": returncode, "signal": signal_name}
    return {"status": "failed", "returncode": returncode}


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_probe(name: str, source: str, *, timeout: float, environment: dict[str, str]) -> dict:
    """Run one child, preserving output even when it crashes or times out."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "-c", source],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=environment,
            check=False,
        )
        result = {
            **classify_returncode(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        result = {
            "status": "timeout",
            "returncode": None,
            "timeout_seconds": timeout,
            "stdout": _as_text(error.stdout),
            "stderr": _as_text(error.stderr),
        }
    except OSError as error:
        result = {
            "status": "error",
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {"name": name, **result, "elapsed_seconds": round(time.monotonic() - started, 3)}


def positive_timeout(value: str) -> float:
    """Reject zero, negative, and nonfinite child timeouts."""
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return timeout


def main(argv: list[str] | None = None) -> int:
    """Print native probe results and optionally save a shareable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safe",
        action="store_true",
        help="use one native CPU thread, disable MKLDNN, and probe the unfused LSTM",
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=120.0,
        metavar="SECONDS",
        help="timeout per native probe (default: 120 seconds)",
    )
    parser.add_argument("--output", type=Path, help="write the complete report as JSON")
    args = parser.parse_args(argv)
    environment = os.environ.copy()
    environment["SCALPER_SAFE_MODE"] = "1" if args.safe else "0"
    if args.safe:
        environment.update(SAFE_ENVIRONMENT)
    report = {
        "runtime": runtime_information(environment),
        "safe_mode": args.safe,
        "lstm_backend": "unfused" if args.safe else "native",
        "timeout_seconds": args.timeout,
        "probes": [],
    }
    print(json.dumps(report["runtime"], indent=2), flush=True)
    print(f"Safe mode: {args.safe}; timeout per probe: {args.timeout:g}s", flush=True)
    for name in PROBES:
        print(f"\n=== {name}: starting isolated probe ===", flush=True)
        result = run_probe(
            name, probe_source(name, safe=args.safe), timeout=args.timeout, environment=environment
        )
        report["probes"].append(result)
        for stream in ("stdout", "stderr"):
            if result[stream]:
                print(f"[{stream}]\n{result[stream]}", end="", flush=True)
                if not result[stream].endswith("\n"):
                    print(flush=True)
        detail = result.get("signal", result.get("returncode"))
        print(f"=== {name}: {result['status']} ({detail}) ===", flush=True)
    success = all(result["status"] == "passed" for result in report["probes"])
    report["passed"] = success
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.output}", flush=True)
    print(
        "\nAll native probes passed. Full training may exercise additional shapes or libraries."
        if success
        else "\nNative probe failures detected; see the last stage printed for each failed probe.",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
