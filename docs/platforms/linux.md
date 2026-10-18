# Linux setup and run guide

Run these Bash commands from the repository root. Use Python 3.13 for the same
version as the main CI job; the package requires Python 3.10 or later. The CI
workflow exercises Ubuntu with Python 3.10 and 3.13. Other distributions and CPU
architectures need compatible dependency wheels and their own validation.

[The verified CI run](https://github.com/Treh06/UNO-Datathon-Team/actions/runs/39079987870)
passed 439 tests with 2 optional ONNX skips on Python 3.10, and all 441 tests with
the ONNX dependencies installed on Python 3.13.

All platforms use the same source code. Create `.venv` on the Linux machine; do
not reuse a Windows or macOS environment. No GPU is needed for the CPU workflow.

## 1. Install

Check that your Python version is supported:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e './detection[dev,api]'
.venv/bin/python -m pip check
```

If `python3` is not the Python version you intend to use, substitute that
interpreter in the `venv` command, for example `python3.13 -m venv .venv`. If
Python cannot create a virtual environment, install your distribution's matching
Python `venv` package. Activation is optional because commands use explicit paths.

## 2. Check the installation

```bash
.venv/bin/scalper-detector --help
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

Tests create their own temporary data and models. A running API is unnecessary.
Optional ONNX checks skip until the deployment dependencies are installed.

## 3. Generate data and train

```bash
.venv/bin/python -m generation_and_testing.simulator.generate \
  --config generation_and_testing/configs/default.json \
  --output generation_and_testing/data/simulation

.venv/bin/scalper-detector train-simulation \
  --simulation generation_and_testing/data/simulation \
  --output artifacts/ticket_model \
  --epochs 20
```

The generator requires a new or empty output directory. Skip generation when
reusing an existing dataset. Training creates the model loaded by the API.

## 4. Start the API in terminal 1

Open terminal 1 at the repository root:

```bash
.venv/bin/python -m uvicorn datathon_authorization.authorization.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

The default model is `artifacts/ticket_model`. To use another directory, set the
environment variable in this terminal before starting the API:

```bash
export SCALPER_MODEL_DIR=artifacts/another_model
```

Open the [API docs](http://127.0.0.1:8000/docs) or
[readiness endpoint](http://127.0.0.1:8000/health). Use one worker: history is in
memory and must advance in global timestamp order.

## 5. Run a demo or replay in terminal 2

Open terminal 2 at the repository root and run the two-minute demo:

```bash
.venv/bin/python -m live_demo
```

For held-out replay instead, stop terminal 1 with Ctrl+C and restart the API so
its history is empty. Then run in terminal 2:

```bash
.venv/bin/python -m datathon_authorization.replay \
  --input generation_and_testing/data/simulation/test/requests.jsonl \
  --output artifacts/api_responses.jsonl \
  --ground-truth generation_and_testing/data/simulation/test/ground_truth.jsonl \
  --evaluation-output artifacts/api_evaluation.json
```

Restart the API between independent demos and replays. Run only one producer
against it at a time. Live output is saved under `artifacts/live_demo/`.

## Optional ONNX export

```bash
.venv/bin/python -m pip install -e './detection[dev,api,deploy]'
.venv/bin/scalper-detector export-onnx \
  --model artifacts/ticket_model \
  --output artifacts/ticket_model/scalper_lstm.onnx
```

This exports and checks the neural branch on CPU. It does not validate IBM Telum
execution. See the [deployment guide](../../detection/docs/telum_deployment.md).

## Training segmentation fault

If training exits with `Segmentation fault (core dumped)` before printing an
epoch, first try a separate environment with freshly installed Linux CPU wheels.
This avoids changing or deleting the existing `.venv` and helps rule out mixed
system/pip packages or a damaged native wheel:

```bash
python3 -m venv .venv-clean
.venv-clean/bin/python -m pip install --upgrade pip
.venv-clean/bin/python -m pip install --no-cache-dir 'torch>=2.6,<3' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-clean/bin/python -m pip install --no-cache-dir -e './detection[dev,api]'
.venv-clean/bin/python -m pip check
.venv-clean/bin/python -c 'import sys, numpy, pandas, torch; print(sys.version); print("numpy", numpy.__version__, "pandas", pandas.__version__, "torch", torch.__version__, "torch threads", torch.get_num_threads())'
PYTHONFAULTHANDLER=1 .venv-clean/bin/scalper-detector train-simulation \
  --simulation generation_and_testing/data/simulation \
  --output artifacts/ticket_model_clean \
  --epochs 20
```

Use the exact same Python interpreter to install and run the command. If the
clean environment still crashes, retain the complete fault-handler output and
the printed package versions; the traceback location distinguishes an input
reader failure from a native model-training failure.

Return to the [project overview](../../README.md) for data contracts, metrics, and
component documentation.
