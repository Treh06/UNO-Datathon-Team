# macOS setup

Use Terminal (zsh or Bash), from the repository root, on an **Apple silicon Mac** with native arm64 Python. Local inference uses CPU; no GPU is required.

PyTorch 2.2 was the final macOS x86-64 release; this project requires PyTorch >=2.6. An Intel Mac therefore needs a compatible remote/Linux environment for the full detector rather than a native installation using these requirements. See the [PyTorch announcement](https://pytorch.org/blog/pytorch2-2/). The standard-library generator can still be used independently.

## Before installing

Install Git (the Command Line Tools or your package manager) and native Python 3.13 using your chosen Python distributor, such as [python.org](https://www.python.org/downloads/macos/). Check both terminal and interpreter architecture:

```bash
pwd
uname -m
python3.13 -c "import sys, platform; print(sys.version); print(sys.executable); print(platform.machine())"
```

Expect arm64 from the interpreter. An Apple silicon machine can run an x86-64 interpreter through Rosetta; that does not supply the native arm64 dependencies needed here.

If `.venv` exists, check it before continuing. A working environment can be reused; a different Python version, architecture, or checkout location requires [recovery](../environment.md#rebuild-a-broken-environment). Do not create a new interpreter over its existing packages.

## Install

For a new/cleared environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: -e './detection[dev,api]'
.venv/bin/python -m pip check
```

Use the normal PyPI packages on macOS, rather than the Linux/Windows PyTorch CPU index. Binary-only dependencies fail clearly if an architecture/version lacks a wheel. Activation is optional; select this checkout's `.venv/bin/python` in the IDE.

## Verify

```bash
.venv/bin/python -B -X faulthandler -c "import numpy as np; print(np.__version__, np.__file__); print(np.ones((2, 2)) @ np.ones((2, 2)))"
.venv/bin/python -B -X faulthandler -c "import pandas as pd; print(pd.__version__, pd.__file__); print(pd.to_datetime(['2026-11-05T17:00:00Z'], utc=True))"
.venv/bin/python -B -X faulthandler -c "import torch; print(torch.__version__, torch.__file__); print(torch.ones(2) + 1)"
.venv/bin/python -m scalper_detector --help
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

Tests create temporary data/models; they do not require a running API or supply the presentation model. Optional ONNX checks can skip without deployment packages.

## Generate, train, and run

Follow the [root walkthrough](../../README.md#run-the-complete-demo), using the Linux/macOS commands. Both terminals must start at the root. Keep the API in terminal 1 and the live producer in terminal 2. Restart the API before each independent demo/replay.

Set `export SCALPER_MODEL_DIR=artifacts/another_model` in terminal 1 before startup to use a different complete model folder; `unset SCALPER_MODEL_DIR` removes an old override.

## Optional ONNX export

After training:

```bash
.venv/bin/python -m pip install --only-binary=:all: -e './detection[dev,api,deploy]'
.venv/bin/python -m scalper_detector export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

This checks neural CPU parity. It does not establish Telum compatibility.

## Reuse and recovery

After pulling dependency changes, reinstall the project in the current valid environment and run `pip check`. Rebuild after changing Python/architecture or moving the checkout. Native import errors require checking interpreter/package architecture before changing application code.

[Environment diagnostics/reset](../environment.md) · [Machine handoff](../../detection/docs/git_workflow.md)
