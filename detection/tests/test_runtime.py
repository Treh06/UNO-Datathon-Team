"""Compatibility controls must run before native libraries initialize."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scalper_detector import runtime

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize(
    "system,machine,enabled",
    [
        ("Linux", "s390x", True),
        ("Linux", "s390", True),
        ("Linux", "S390X", True),
        ("Linux", "ppc64le", False),
        ("Linux", "x86_64", False),
        ("Darwin", "arm64", False),
        ("Windows", "AMD64", False),
    ],
)
def test_automatic_profile_is_specific_to_linux_ibm_z(monkeypatch, system, machine, enabled):
    monkeypatch.delenv("SCALPER_SAFE_MODE", raising=False)
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    assert runtime.safe_mode_enabled() is enabled


@pytest.mark.parametrize("setting,expected", [("0", False), ("1", True)])
@pytest.mark.parametrize("machine", ["s390x", "x86_64"])
def test_explicit_profile_overrides_architecture(monkeypatch, setting, expected, machine):
    monkeypatch.setenv("SCALPER_SAFE_MODE", setting)
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    assert runtime.safe_mode_enabled() is expected


def test_invalid_profile_is_rejected(monkeypatch):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "sometimes")
    with pytest.raises(ValueError, match="SCALPER_SAFE_MODE"):
        runtime.safe_mode_enabled()


@pytest.mark.parametrize("safe", [False, True])
def test_environment_limits_override_existing_values_only_in_safe_mode(monkeypatch, safe):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "1" if safe else "0")
    monkeypatch.setenv("PYTHONFAULTHANDLER", "0")
    for name in runtime.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "8")
    enable_traceback = Mock()
    monkeypatch.setattr(runtime, "enable_fault_traceback", enable_traceback)
    assert runtime.configure_environment() is safe
    for name in runtime.THREAD_ENVIRONMENT:
        assert os.environ[name] == ("1" if safe else "8")
    if safe:
        assert os.environ["PYTHONFAULTHANDLER"] == "1"
        enable_traceback.assert_called_once_with()
    else:
        enable_traceback.assert_not_called()


def _run_isolated(source):
    return subprocess.run(
        [sys.executable, "-S", "-c", source],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_package_import_configures_linuxone_before_any_native_import():
    source = f"""
import importlib.abc
import json
import os
import platform
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
native_packages = {{'numpy', 'pandas', 'torch', 'sklearn', 'scipy'}}
class BlockNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in native_packages:
            raise RuntimeError('native library imported before runtime setup: ' + fullname)
sys.meta_path.insert(0, BlockNativeImports())
os.environ['SCALPER_SAFE_MODE'] = 'auto'
platform.system = lambda: 'Linux'
platform.machine = lambda: 's390x'
for name in {runtime.THREAD_ENVIRONMENT!r}:
    os.environ[name] = '8'
import scalper_detector
assert not native_packages.intersection(sys.modules)
print(json.dumps({{name: os.environ[name] for name in {runtime.THREAD_ENVIRONMENT!r}}}))
"""
    completed = _run_isolated(source)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == dict.fromkeys(runtime.THREAD_ENVIRONMENT, "1")


@pytest.mark.parametrize("use_sys_argv", [False, True])
def test_bootstrap_removes_safe_flags_before_importing_cli_and_preserves_arguments(use_sys_argv):
    arguments = [
        "train-simulation",
        "--safe-mode",
        "--simulation",
        "input path",
        "--epochs",
        "20",
        "--safe-mode",
    ]
    expected = [value for value in arguments if value != "--safe-mode"]
    source = f"""
import importlib.abc
import importlib.util
import json
import os
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
os.environ['SCALPER_SAFE_MODE'] = '0'
for name in {runtime.THREAD_ENVIRONMENT!r}:
    os.environ[name] = '8'
native_packages = {{'numpy', 'pandas', 'torch', 'sklearn', 'scipy'}}
class FakeCli(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in native_packages:
            raise RuntimeError('unexpected native import: ' + fullname)
        if fullname == 'scalper_detector.cli':
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        assert os.environ['SCALPER_SAFE_MODE'] == '1'
        assert all(os.environ[name] == '1' for name in {runtime.THREAD_ENVIRONMENT!r})
        assert not native_packages.intersection(sys.modules)
        module.main = lambda args: print(json.dumps(args))
sys.meta_path.insert(0, FakeCli())
from scalper_detector.bootstrap import main
arguments = {arguments!r}
if {use_sys_argv!r}:
    sys.argv = ['scalper-detector'] + arguments
    main()
else:
    main(arguments)
"""
    completed = _run_isolated(source)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected
    assert "Compatibility mode" in completed.stderr


@pytest.fixture
def fake_torch(monkeypatch):
    module = ModuleType("torch")
    module.backends = SimpleNamespace(mkldnn=SimpleNamespace(enabled=True))
    module.get_num_interop_threads = Mock(return_value=4)
    module.set_num_interop_threads = Mock()
    module.set_num_threads = Mock()
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_torch_safe_mode_disables_mkldnn_and_forces_one_thread(monkeypatch, fake_torch):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "1")
    assert runtime.configure_torch_runtime(8) == 1
    assert fake_torch.backends.mkldnn.enabled is False
    fake_torch.set_num_threads.assert_called_once_with(1)
    fake_torch.set_num_interop_threads.assert_called_once_with(1)


def test_torch_preserves_requested_threads_outside_safe_mode(monkeypatch, fake_torch):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "0")
    assert runtime.configure_torch_runtime(6) == 6
    assert fake_torch.backends.mkldnn.enabled is True
    fake_torch.set_num_threads.assert_called_once_with(6)
    fake_torch.get_num_interop_threads.assert_not_called()
    fake_torch.set_num_interop_threads.assert_not_called()


def test_torch_does_not_reset_interop_threads_when_already_one(monkeypatch, fake_torch):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "1")
    fake_torch.get_num_interop_threads.return_value = 1
    assert runtime.configure_torch_runtime(4) == 1
    fake_torch.set_num_interop_threads.assert_not_called()
    fake_torch.set_num_threads.assert_called_once_with(1)


def test_late_interop_configuration_warns_and_still_applies_other_controls(monkeypatch, fake_torch):
    monkeypatch.setenv("SCALPER_SAFE_MODE", "1")
    fake_torch.set_num_interop_threads.side_effect = RuntimeError("parallel work started")
    with pytest.warns(RuntimeWarning, match="parallel work already started"):
        assert runtime.configure_torch_runtime(8) == 1
    assert fake_torch.backends.mkldnn.enabled is False
    fake_torch.set_num_threads.assert_called_once_with(1)


@pytest.mark.parametrize("error", [OSError("no descriptor"), RuntimeError("no stderr")])
def test_fault_traceback_handles_embedded_interpreters(monkeypatch, error):
    monkeypatch.setattr(runtime.faulthandler, "enable", Mock(side_effect=error))
    runtime.enable_fault_traceback()
