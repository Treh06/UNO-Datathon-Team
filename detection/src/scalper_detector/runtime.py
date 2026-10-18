"""Configure conservative native execution before numerical libraries load.

LinuxONE uses s390x, which needs its own native dependency builds. This profile
reduces threading and fused-kernel exposure; it cannot repair a broken binary.
"""

from __future__ import annotations

import faulthandler
import os
import platform
import warnings

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def safe_mode_enabled() -> bool:
    """Resolve an explicit override, defaulting to safe execution on LinuxONE."""
    setting = os.environ.get("SCALPER_SAFE_MODE", "auto").strip().lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting != "auto":
        raise ValueError("SCALPER_SAFE_MODE must be auto, 1, or 0.")
    return platform.system() == "Linux" and platform.machine().lower() in {"s390", "s390x"}


def enable_fault_traceback() -> None:
    """Keep Python stack traces available when native code crashes."""
    try:
        faulthandler.enable()
    except (OSError, RuntimeError):
        # Some embedded interpreters do not provide a writable stderr descriptor.
        pass


def configure_environment() -> bool:
    """Set native thread limits before imports initialize BLAS/OpenMP runtimes."""
    enabled = safe_mode_enabled()
    if enabled:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        os.environ["PYTHONFAULTHANDLER"] = "1"
        enable_fault_traceback()
    return enabled


def effective_thread_count(requested: int) -> int:
    """Honor the configured worker count outside the compatibility profile."""
    return 1 if safe_mode_enabled() else requested


def configure_torch_runtime(requested_threads: int) -> int:
    """Apply PyTorch settings before tensor work and return the actual thread count."""
    import torch

    threads = effective_thread_count(requested_threads)
    if safe_mode_enabled():
        torch.backends.mkldnn.enabled = False
        if torch.get_num_interop_threads() != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                warnings.warn(
                    "PyTorch parallel work already started; inter-op threads could not be reduced. "
                    "Start a new process with SCALPER_SAFE_MODE=1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    torch.set_num_threads(threads)
    return threads
