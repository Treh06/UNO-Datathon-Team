# macOS setup and run guide

Run these commands in Terminal using zsh or Bash, from the repository root, on an
**Apple silicon (arm64) Mac**. Use Python 3.13 for the CI configuration; the package
requires Python 3.10 or later. The full suite passed locally on Apple silicon with
Python 3.13.2, including ONNX checks. The newly added macOS CI job exercises Python
3.13 with API and ONNX packages and awaits its first remote run after the changes
are pushed.

All platforms use the same source code. Create `.venv` on the Mac; do not copy a
Linux or Windows environment. Install native Python and native dependencies for
your Mac's architecture. Local inference uses CPU and does not require a GPU.

**Intel Mac limitation:** this project requires `torch>=2.6,<3`, while PyTorch
2.2.x was the last release for macOS x86-64, according to the
[PyTorch release announcement](https://pytorch.org/blog/pytorch2-2/). Native Intel
macOS is therefore outside the supported dependency configuration. Run the
pipeline in a compatible [Linux environment](linux.md) or on a
[Windows computer](windows.md), or connect to an API running there. Creating a
separate Mac source branch would not provide the missing PyTorch wheels.

## 1. Install

Check that your Python version is supported:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './detection[dev,api]'
.venv/bin/python -m pip check
```

If `python3` is not the Python version you intend to use, substitute that
interpreter when creating the environment, for example `python3.13 -m venv .venv`.
Activation is optional because commands use explicit paths.

Use the normal PyPI installation above for macOS. The PyTorch CPU wheel index
used in the Linux and Windows guides is not the macOS installation path. A
missing-wheel error requires checking the Python version and architecture of the
interpreter used to create `.venv`; a different Git branch does not supply a
missing native dependency.

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

Return to the [project overview](../../README.md) for data contracts, metrics, and
component documentation.
