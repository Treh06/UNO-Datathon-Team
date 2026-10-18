# Windows setup and run guide

Run these commands in **PowerShell**, from the repository root. Use Python 3.13
for the same version as Windows CI; the package requires Python 3.10 or later.
The CI workflow exercises Windows with Python 3.13. Use a native 64-bit Python
installation with compatible dependency wheels.

[The verified CI run](https://github.com/Treh06/UNO-Datathon-Team/actions/runs/39079987870)
passed 439 tests with 2 optional ONNX checks skipped on Windows Server 2025 x64
with Python 3.13. The optional deployment packages were not installed in that job.

All platforms use the same source code. Create `.venv` on Windows; do not copy a
Linux or macOS environment. No GPU is needed for the CPU workflow.

## 1. Install

Check that the Python launcher can find Python 3.13:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e './detection[dev,api]'
.venv\Scripts\python.exe -m pip check
```

If you use another supported Python version, change the launcher selector in the
first two commands. If `py` is unavailable but `python` selects the intended
interpreter, use `python -m venv .venv` instead. Activation is optional because
commands use explicit executable paths; no PowerShell execution-policy change is
needed.

## 2. Check the installation

```powershell
.venv\Scripts\scalper-detector.exe --help
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m pytest -q
```

Tests create their own temporary data and models. A running API is unnecessary.
Optional ONNX checks skip until the deployment dependencies are installed.

## 3. Generate data and train

Each command is written on one line so it can be pasted directly into PowerShell:

```powershell
.venv\Scripts\python.exe -m generation_and_testing.simulator.generate --config generation_and_testing/configs/default.json --output generation_and_testing/data/simulation
.venv\Scripts\scalper-detector.exe train-simulation --simulation generation_and_testing/data/simulation --output artifacts/ticket_model --epochs 20
```

The generator requires a new or empty output directory. Skip generation when
reusing an existing dataset. Training creates the model loaded by the API.

## 4. Start the API in terminal 1

Open PowerShell terminal 1 at the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn datathon_authorization.authorization.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The default model is `artifacts/ticket_model`. To use another directory, set the
environment variable in this terminal before starting the API:

```powershell
$env:SCALPER_MODEL_DIR = 'artifacts/another_model'
```

Open the [API docs](http://127.0.0.1:8000/docs) or
[readiness endpoint](http://127.0.0.1:8000/health). Use one worker: history is in
memory and must advance in global timestamp order.

## 5. Run a demo or replay in terminal 2

Open PowerShell terminal 2 at the repository root and run the two-minute demo:

```powershell
.venv\Scripts\python.exe -m live_demo
```

For held-out replay instead, stop terminal 1 with Ctrl+C and restart the API so
its history is empty. Then run in terminal 2:

```powershell
.venv\Scripts\python.exe -m datathon_authorization.replay --input generation_and_testing/data/simulation/test/requests.jsonl --output artifacts/api_responses.jsonl --ground-truth generation_and_testing/data/simulation/test/ground_truth.jsonl --evaluation-output artifacts/api_evaluation.json
```

Restart the API between independent demos and replays. Run only one producer
against it at a time. Live output is saved under `artifacts/live_demo/`.

## Optional ONNX export

```powershell
.venv\Scripts\python.exe -m pip install -e './detection[dev,api,deploy]'
.venv\Scripts\scalper-detector.exe export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

This exports and checks the neural branch on CPU. It does not validate IBM Telum
execution. See the [deployment guide](../../detection/docs/telum_deployment.md).

Return to the [project overview](../../README.md) for data contracts, metrics, and
component documentation.
