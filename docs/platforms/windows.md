# Windows setup

Use PowerShell on native 64-bit Windows x86. Run from the repository root, not `detection/`. This guide installs the shared environment; the [root walkthrough](../../README.md#run-the-complete-demo) then covers generation, training, API startup, and the live demo with complete PowerShell commands.

## Before installing

Install [Git](https://git-scm.com/downloads) and [native 64-bit CPython 3.13](https://www.python.org/downloads/windows/) if needed. Check the launcher:

```powershell
py -0p
py -3.13 -c "import sys, platform, struct; print(sys.version); print(platform.machine()); print(struct.calcsize('P') * 8)"
Get-Location
```

Expect Python 3.13, AMD64, and 64 bits. If `py` is unavailable, use the full path to the intended installed `python.exe` in the environment-creation command. Do not assume the `python` or `pip` on PATH is the one you selected.

**If `.venv` already exists, stop before creating it again.** A working environment can be reused with its original interpreter. A Python-version change or incompatible wheels require [recovery or clearing](../environment.md#rebuild-a-broken-environment). In particular, Python 3.13 cannot use `cp312-cp312-win_amd64` packages left by a 3.12 install.

## Install

For a new/cleared environment:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --only-binary=:all: 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install --only-binary=:all: -e './detection[dev,api]'
.venv\Scripts\python.exe -m pip check
```

The editable install points at this checkout's detector source; API/demo modules run from the root. Binary-only dependency selection makes a missing wheel fail clearly rather than starting an unintended native build. A missing-wheel error needs an interpreter/architecture/package-source check.

Activation is optional. No PowerShell execution-policy change is needed when using explicit executable paths. In VS Code select this checkout's `.venv\Scripts\python.exe`.

## Verify

Run each native check separately:

```powershell
.venv\Scripts\python.exe -B -X faulthandler -c "import numpy as np; print(np.__version__, np.__file__); print(np.ones((2, 2)) @ np.ones((2, 2)))"
.venv\Scripts\python.exe -B -X faulthandler -c "import pandas as pd; print(pd.__version__, pd.__file__); print(pd.to_datetime(['2026-11-05T17:00:00Z'], utc=True))"
.venv\Scripts\python.exe -B -X faulthandler -c "import torch; print(torch.__version__, torch.__file__); print(torch.ones(2) + 1)"
.venv\Scripts\python.exe -m scalper_detector --help
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m pytest -q
```

Tests train temporary models; they need no saved dataset or running API. They do not create the presentation model. ONNX tests can skip without deployment packages.

## Run and repeat

Follow [generate, train, start API, start demo](../../README.md#run-the-complete-demo). Start both PowerShell terminals in the repository root. Before a second independent demo/replay, Ctrl+C the API and start it again.

For a different model, set `$env:SCALPER_MODEL_DIR='artifacts/another_model'` in the API terminal before launch. This variable is terminal-specific. Remove a stale override with `Remove-Item Env:SCALPER_MODEL_DIR -ErrorAction SilentlyContinue` when returning to the default.

After pulling changes to `detection/pyproject.toml`, reinstall the project in the existing valid environment and rerun `pip check`. Rebuild instead if Python changed.

## Optional ONNX export

After a trained model exists:

```powershell
.venv\Scripts\python.exe -m pip install --only-binary=:all: -e './detection[dev,api,deploy]'
.venv\Scripts\python.exe -m scalper_detector export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

This checks the neural graph on CPU. It does not compile or validate Telum execution.

## Problems and moving machines

- “Not supported on this platform”: inspect interpreter and wheel tags, then rebuild. Upgrading pip alone does not repair old native packages.
- Missing `scalper-detector.exe`: use `python -m scalper_detector` with the environment Python and confirm the editable install completed.
- DLL load error: record the failing module and check its required native runtime after confirming version/architecture. Do not treat every DLL issue as a model-code error.
- Moving the project or changing Python: recreate the environment. Keep datasets/models separately; never transfer `.venv`.

[Environment diagnostics and reset](../environment.md) · [Team machine handoff](../../detection/docs/git_workflow.md)
