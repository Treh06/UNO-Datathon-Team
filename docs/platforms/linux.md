# Linux x86-64 setup

Use Bash from the repository root. This guide covers ordinary **x86-64 Linux**. On IBM LinuxONE/Linux on IBM Z, use the [s390x guide](linuxone.md); its native packages and installation route differ.

## Before installing

Install Git, an appropriate Python interpreter, and its matching `venv` support. Python 3.13 matches the main CI configuration; the project declares >=3.10, and CI also exercises 3.10. Verify the actual interpreter:

```bash
pwd
uname -m
python3 --version
python3 -c "import sys, platform; print(sys.executable); print(platform.machine())"
```

Expect `x86_64`. The commands below use the `python3` you just checked. If choosing a separately installed Python 3.13 to match CI, use `python3.13` in the environment-creation command instead. Ubuntu 24.04 normally selects 3.12 with `python3`; that is within the project's declared range but is not a separate current CI job. Record and verify it rather than assuming it is 3.13. Do not replace the system Python just to follow this guide.

If venv creation reports missing `ensurepip`/venv support, install the distribution package matching that interpreter. The package name/repository depends on how Python was installed.

On Ubuntu 24.04, the distribution's default interpreter and venv support can be installed with the following administrator commands. These do not install Python 3.13:

```bash
sudo apt update
sudo apt install git python3 python3-venv
```

**If `.venv` exists, inspect it first.** Reuse a valid environment with its original interpreter, or follow [recovery](../environment.md#rebuild-a-broken-environment). Do not create a new Python version over existing installed packages.

## Install

For a new/cleared environment with the selected `python3`:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --only-binary=:all: -e './detection[dev,api]'
.venv/bin/python -m pip check
```

Use the normal glibc-based distribution/runtime compatible with the chosen wheels. Other architectures or libc variants need their own validation. Binary-only dependency selection fails clearly when no compatible wheel is available; it avoids an accidental native source build.

Activation is optional. Do not use `sudo pip`, copy system `site-packages`, or mix conda and pip packages without a validated environment plan.

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

Run native checks separately. If one crashes, stop and collect its trace rather than running the whole pipeline. Tests use temporary data/models and require no API or saved dataset. Optional ONNX tests can skip.

## Generate, train, and run

Follow the [root walkthrough](../../README.md#run-the-complete-demo), using its Linux/macOS blocks. Keep the API running in terminal 1 and the live producer in terminal 2, both at the root. Ctrl+C stops each process; restart the API before an independent demo/replay.

The API defaults to `artifacts/ticket_model`. To override, use `export SCALPER_MODEL_DIR=artifacts/another_model` before launching in terminal 1. Use `unset SCALPER_MODEL_DIR` to remove a stale override.

After dependency declarations change, reinstall the project in the current valid environment. After an interpreter or checkout-location change, rebuild the environment.

## Optional ONNX export

After training:

```bash
.venv/bin/python -m pip install --only-binary=:all: -e './detection[dev,api,deploy]'
.venv/bin/python -m scalper_detector export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

This verifies a neural graph on CPU; see [IBM deployment](../../detection/docs/telum_deployment.md) for target compilation.

## Segmentation faults

A segfault is a native crash, not a normal Python exception. Start with the [separate native checks and fault-handler procedure](../environment.md#check-native-packages-before-the-full-pipeline). Record the failing command, Python/package versions, package locations, architecture, and OS. A fresh environment can isolate mixed binaries, but reinstalling the same unvalidated stack is not a guaranteed fix.

On s390x, switch to the [LinuxONE procedure](linuxone.md) rather than repeatedly installing desktop CPU wheels. Do not assume that a traceback inside pandas proves pandas is the underlying defect.

[Environment recovery](../environment.md) · [Machine handoff](../../detection/docs/git_workflow.md)
