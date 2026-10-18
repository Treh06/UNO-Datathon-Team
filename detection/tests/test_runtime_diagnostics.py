"""The native diagnostic supervisor must survive failed child interpreters."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_runtime.py"
SPEC = importlib.util.spec_from_file_location("diagnose_runtime", SCRIPT)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


@pytest.mark.parametrize(
    "returncode,expected",
    [
        (0, {"status": "passed", "returncode": 0}),
        (1, {"status": "failed", "returncode": 1}),
        (-signal.SIGSEGV, {"status": "signal", "returncode": -signal.SIGSEGV, "signal": "SIGSEGV"}),
        (-signal.SIGILL, {"status": "signal", "returncode": -signal.SIGILL, "signal": "SIGILL"}),
        (
            0xC0000005,
            {"status": "signal", "returncode": 0xC0000005, "signal": "ACCESS_VIOLATION"},
        ),
        (
            -1073741819,
            {"status": "signal", "returncode": -1073741819, "signal": "ACCESS_VIOLATION"},
        ),
    ],
)
def test_exit_status_distinguishes_python_errors_and_native_crashes(returncode, expected):
    assert diagnostics.classify_returncode(returncode) == expected


def test_supervisor_imports_without_loading_native_dependencies():
    source = f"""
import importlib.abc
import importlib.util
import sys
class BlockNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'numpy', 'pandas', 'torch', 'sklearn'}}:
            raise RuntimeError('native import attempted in supervisor: ' + fullname)
sys.meta_path.insert(0, BlockNativeImports())
spec = importlib.util.spec_from_file_location('diagnostics', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.runtime_information({{}})
"""
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_python_child_failure_preserves_last_stage():
    result = diagnostics.run_probe(
        "failure",
        "print('before failure', flush=True)\nraise ValueError('broken stage')",
        timeout=5,
        environment=os.environ.copy(),
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert "before failure" in result["stdout"]
    assert "ValueError: broken stage" in result["stderr"]


def test_timeout_preserves_output_and_reports_timeout():
    result = diagnostics.run_probe(
        "slow",
        "import time\nprint('before sleep', flush=True)\ntime.sleep(10)",
        timeout=0.3,
        environment=os.environ.copy(),
    )
    assert result["status"] == "timeout"
    assert result["returncode"] is None
    assert result["timeout_seconds"] == 0.3
    assert "before sleep" in result["stdout"]


def test_child_signal_is_reported_without_signaling_the_test_process(monkeypatch):
    def failed_child(command, **kwargs):
        assert command[1:5] == ["-X", "faulthandler", "-u", "-c"]
        return SimpleNamespace(returncode=-signal.SIGSEGV, stdout="before matmul\n", stderr="")

    monkeypatch.setattr(diagnostics.subprocess, "run", failed_child)
    result = diagnostics.run_probe("crash", "pass", timeout=5, environment={})
    assert result["status"] == "signal"
    assert result["signal"] == "SIGSEGV"
    assert result["stdout"] == "before matmul\n"


def test_missing_interpreter_reports_launch_failure(monkeypatch):
    def missing_child(*args, **kwargs):
        raise FileNotFoundError("interpreter unavailable")

    monkeypatch.setattr(diagnostics.subprocess, "run", missing_child)
    result = diagnostics.run_probe("missing", "pass", timeout=5, environment={})
    assert result["status"] == "error"
    assert "interpreter unavailable" in result["stderr"]


@pytest.mark.parametrize("safe", [False, True])
def test_lstm_probe_explicitly_selects_backend_for_comparison(safe):
    source = diagnostics.probe_source("lstm", safe=safe)
    assert f"DIAGNOSTIC_SAFE_MODE = {safe!r}" in source
    assert 'backend = "unfused" if DIAGNOSTIC_SAFE_MODE else "native"' in source
    assert "lstm_backend=backend" in source


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(SystemExit) as error:
        diagnostics.main(["--timeout", value])
    assert error.value.code == 2


@pytest.mark.parametrize("safe", [False, True])
def test_report_continues_after_signal_and_explicitly_selects_runtime(monkeypatch, tmp_path, safe):
    calls = []
    monkeypatch.setenv("PRIVATE_TEST_TOKEN", "must not appear in the report")
    monkeypatch.setenv("SCALPER_SAFE_MODE", "0" if safe else "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setattr(diagnostics, "installed_versions", lambda: {"torch": "test"})

    def fake_probe(name, source, *, timeout, environment):
        calls.append((name, source, timeout, environment))
        return {
            "name": name,
            "status": "signal" if name == "numpy" else "passed",
            "returncode": -signal.SIGILL if name == "numpy" else 0,
            "signal": "SIGILL" if name == "numpy" else None,
            "stdout": "last stage\n",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(diagnostics, "run_probe", fake_probe)
    output = tmp_path / "reports" / "runtime.json"
    arguments = ["--timeout", "5", "--output", str(output)]
    if safe:
        arguments.append("--safe")
    assert diagnostics.main(arguments) == 1
    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["safe_mode"] is safe
    assert report["lstm_backend"] == ("unfused" if safe else "native")
    assert report["passed"] is False
    assert len(calls) == len(diagnostics.PROBES)
    assert "PRIVATE_TEST_TOKEN" not in report_text
    assert "must not appear" not in report_text
    for _, source, timeout, environment in calls:
        assert timeout == 5
        assert environment["SCALPER_SAFE_MODE"] == ("1" if safe else "0")
        if safe:
            for key, value in diagnostics.SAFE_ENVIRONMENT.items():
                assert environment[key] == value
        else:
            assert environment["OMP_NUM_THREADS"] == "7"
        assert "torch.backends.mkldnn.enabled = False" in source
        assert "torch.set_num_threads(1)" in source
        assert str(SCRIPT.parents[1] / "detection" / "src") in source
