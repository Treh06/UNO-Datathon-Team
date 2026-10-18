"""Initialize native runtime controls before importing the detector CLI."""

from __future__ import annotations

import os
import sys

from .runtime import configure_environment, enable_fault_traceback


def main(argv: list[str] | None = None) -> None:
    """Accept compatibility mode anywhere in a command before heavy imports."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--safe-mode" in arguments:
        os.environ["SCALPER_SAFE_MODE"] = "1"
        arguments = [value for value in arguments if value != "--safe-mode"]
    enabled = configure_environment()
    enable_fault_traceback()
    if enabled:
        print(
            "Compatibility mode: one CPU thread, unfused LSTM, accelerated CPU kernels disabled.",
            file=sys.stderr,
            flush=True,
        )
    from .cli import main as run_command

    run_command(arguments)
