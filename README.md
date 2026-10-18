# Ticket scalper detection

A datathon prototype that detects suspicious ticket buying across participating merchants. It combines an LSTM (recent card activity) with a random forest (transaction patterns) and returns a score, observed reasons, and simulated APPROVE/REVIEW/DECLINE advice. It never moves money or executes merchant actions.

## Start here

Run commands from **the repository root**: the directory containing `detection/`, `generation_and_testing/`, `live_demo/`, and this README. Install once on each machine, generate and train once per scenario, then restart the API for each independent demonstration.

| Your machine | Setup guide |
| --- | --- |
| Windows, 64-bit x86 | [Windows / PowerShell](docs/platforms/windows.md) |
| Linux, x86-64 | [Linux / Bash](docs/platforms/linux.md) |
| macOS, Apple silicon | [macOS / Terminal](docs/platforms/macos.md) |
| IBM LinuxONE / Linux on IBM Z, s390x | [LinuxONE / Bash](docs/platforms/linuxone.md) |

For desktop development, Python 3.13 matches the main CI configuration. The package declares Python >=3.10, but native dependency availability also depends on architecture and Python version. On IBM, use the interpreter and native packages validated for the selected IBM environment.

**Already have a broken environment or changed Python versions?** Follow [environment recovery](docs/environment.md#rebuild-a-broken-environment) before installing. Running `venv` over an existing directory does not clear old packages.

## What each part does

| Component | Input and result |
| --- | --- |
| [Generator](generation_and_testing/README.md) | Creates synthetic requests and separate labels in train/validation/test splits. |
| [Detector](detection/README.md) | Trains, calibrates, saves, scores, and evaluates the two-model ensemble. |
| [Authorization API](datathon_authorization/README.md) | Receives a purchase, uses causal transaction history, and returns the saved model's decision. |
| [Live demo](live_demo/README.md) | Creates new purchases over time and displays actual API responses, timing, and evaluation. |
| Replay client | Sends an existing request file to the API and optionally evaluates responses against separate truth. |

Only current and earlier purchases contribute to an order's features. Labels, actor IDs, and tactic names stay out of online inputs. Synthetic accuracy describes the simulated scenario; it does not establish real-world scalper accuracy.

## Run the complete demo

Complete your platform setup and installation checks first. Commands below are single lines to simplify copying. Choose the block for your shell. Activation is unnecessary because commands name the environment explicitly.

Run one command at a time and stop if it fails. Fix installation or training errors before starting the API.

### 1. Generate training and test purchases

Linux/macOS:

```bash
.venv/bin/python -m generation_and_testing.simulator.generate --config generation_and_testing/configs/default.json --output generation_and_testing/data/simulation
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m generation_and_testing.simulator.generate --config generation_and_testing/configs/default.json --output generation_and_testing/data/simulation
```

The destination must be new or empty. If the dataset already exists, reuse it and skip generation, or choose another output path and update subsequent commands. Defaults create 11,640 requests: 7,284 train, 1,728 validation, and 2,628 test. Each split has `requests.jsonl` and separate `ground_truth.jsonl`; `manifest.json` records the actual counts and settings.

### 2. Train and save the detector

Linux/macOS:

```bash
.venv/bin/python -m scalper_detector train-simulation --simulation generation_and_testing/data/simulation --output artifacts/ticket_model --epochs 20
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m scalper_detector train-simulation --simulation generation_and_testing/data/simulation --output artifacts/ticket_model --epochs 20
```

Training joins labels offline. It preserves the generator's train/test groups and splits source validation into model-selection and calibration groups. It writes `model.pt`, `tree.json`, `metadata.json`, evaluation reports, and held-out orders. Choose a different output directory to preserve an earlier model; training can replace artifacts at an existing destination.

Tests train temporary models; passing tests does **not** create `artifacts/ticket_model`. The API needs the model from this step or a complete model directory supplied by the team.

### 3. Start the API in terminal 1

Linux/macOS:

```bash
.venv/bin/python -m uvicorn datathon_authorization.authorization.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m uvicorn datathon_authorization.authorization.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Leave this terminal running. Open [health](http://127.0.0.1:8000/health) and confirm `ready: true`. [API docs](http://127.0.0.1:8000/docs) show the contract; [stats](http://127.0.0.1:8000/stats) show counts.

The default model is `artifacts/ticket_model`. To select another model, set `SCALPER_MODEL_DIR` **in terminal 1 before launching**: `export SCALPER_MODEL_DIR=artifacts/another_model` in Bash/zsh or `$env:SCALPER_MODEL_DIR='artifacts/another_model'` in PowerShell.

### 4. Start live purchases in terminal 2

Open a second terminal at the same repository root.

Linux/macOS:

```bash
.venv/bin/python -m live_demo
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m live_demo
```

The two-minute default mixes legitimate buyers, bulk-buying decoys, and four scalper tactics. Traffic rises fivefold from seconds 30 to 50. The terminal shows scores, decisions, reasons, latency, and schedule delay; results go to a new folder under `artifacts/live_demo/`. Ctrl+C saves a partial summary. See [demo controls and outputs](live_demo/README.md).

For each new independent demo, stop the API with Ctrl+C and start it again before running the producer. Use one API worker and one producer: its in-memory history requires global chronological order. Simulated declines still enter attempt history.

### Replay a fixed test file instead

Start a fresh API. Replay is useful for comparing models on identical requests; the live demo creates new traffic.

Linux/macOS:

```bash
.venv/bin/python -m datathon_authorization.replay --input generation_and_testing/data/simulation/test/requests.jsonl --output artifacts/api_responses.jsonl --ground-truth generation_and_testing/data/simulation/test/ground_truth.jsonl --evaluation-output artifacts/api_evaluation.json
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m datathon_authorization.replay --input generation_and_testing/data/simulation/test/requests.jsonl --output artifacts/api_responses.jsonl --ground-truth generation_and_testing/data/simulation/test/ground_truth.jsonl --evaluation-output artifacts/api_evaluation.json
```

Truth is joined locally after scoring. Replay response/report paths may be overwritten; use new filenames when preserving earlier runs.

## Moving machines and sharing team work

Copy or pull **source code**, then recreate the environment on the destination. Never copy `.venv`, installed packages, or compiled model libraries between platforms. Datasets and model artifacts are ignored by Git, so a new checkout will not contain them. Transfer complete model folders separately or regenerate and train.

Use the [machine handoff and team workflow](detection/docs/git_workflow.md) for source updates, artifact transfer, checks, and remote demonstrations. Use [environment diagnostics](docs/environment.md) for interpreter mismatches, native crashes, and version records.

## Check a team change

After platform setup, run these from the root. Windows uses `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

```bash
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

These commands check dependencies, formatting, and software behavior. They do not change source formatting. Tests create temporary data/models and need no running API. Optional ONNX tests can skip without deployment extras. The [CI workflow](.github/workflows/ci.yml) specifies OS/Python combinations; it does not test s390x or Telum execution. Use the completed run for the relevant commit as evidence, rather than a historical test count.

## Common problems

| Symptom | Next step |
| --- | --- |
| Packages “not supported on this platform” | Check interpreter and wheel tags, then [rebuild the environment](docs/environment.md#rebuild-a-broken-environment). |
| Missing Python, import, or command | Follow the matching platform guide, run from the root, and select the same interpreter in the IDE. |
| LinuxONE installation or native crash | Follow the [s390x guide](docs/platforms/linuxone.md); desktop Linux wheels are not interchangeable with IBM binaries. |
| API cannot load the model | Complete training or set the correct model path; keep all model files together. |
| HTTP 409 or demo refuses used history | Restart the API; preserve timestamp order and run only one producer. |
| HTTP 422 | Read the response details: tickets, USD, MCC 7922, valid fields, no truth labels. |
| Generator refuses output | Reuse the existing dataset or choose a new/empty destination. |
| Port 8000 is occupied | Stop the earlier server, or use another port and the same `--url` in the client. |
| Live traffic falls behind | Inspect schedule delay; lower the offered rate. This serial demo is not a concurrent load benchmark. |

## Technical reference

[Input/features and model behavior](detection/docs/implementation.md) · [IBM deployment and measurement](detection/docs/telum_deployment.md)

The API is a local single-worker prototype. Persistent state, coordinated workers, authentication, real processor integration, and hardware performance validation remain separate work. Merchant-policy decisions and calibrated review flags use different cutoffs and can differ.
