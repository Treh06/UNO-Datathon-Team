# Contributing

Run commands from the repository root. The [README](README.md) covers installation,
training, and the live demonstration. All components share one virtual environment.

## Development setup

Use Python 3.10 or later. On Linux and macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e './detection[dev,api]'
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e './detection[dev,api]'
```

For the commands below, Windows users should replace `.venv/bin/python` with
`.venv\Scripts\python.exe`. Add `deploy` to the installed extras to run the optional
ONNX export and runtime parity checks.

## Code quality

Ruff provides shared formatting and import sorting through [ruff.toml](ruff.toml).
Use descriptive names, document public interfaces and state assumptions, and keep
validation close to the boundary where inputs enter the system.

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
```

To apply formatting and safe lint fixes:

```bash
.venv/bin/python -m ruff check --fix .
.venv/bin/python -m ruff format .
```

CI runs lint and formatting checks, tests on Linux with Python 3.10 and 3.13,
and tests on Windows with Python 3.13. The Linux Python 3.13 job also checks ONNX
export and CPU parity. IBM accelerator compilation and timing require separate
target hardware validation.

## Project layout

| Directory | Responsibility |
| --- | --- |
| `generation_and_testing/` | Seeded synthetic authorization requests and separate labels |
| `detection/src/scalper_detector/` | Validation, causal features, training, calibration, scoring, and export |
| `datathon_authorization/` | Stateful FastAPI authorization service and offline replay client |
| `live_demo/` | Paced online traffic, terminal output, and run summaries |

Tests live beside their component. The root `pytest.ini` collects the complete
suite and makes repository modules available without custom `PYTHONPATH` settings.

## Changes and pull requests

Use a descriptive branch name and keep each change focused. Include the reason
for the change, the resulting behavior, and the checks you ran in the pull request.
Add a regression test when fixing behavior; keep formatting changes separate from
new features where practical.

Keep generated datasets in `data/` directories and model/demo output under
`artifacts/`. These directories and `.venv/` are ignored. Do not commit payment
data, credentials, or machine-specific runtime paths.

Preserve these contracts when changing the pipeline:

- Ground-truth labels stay outside live requests and model features.
- Feature history advances in global chronological order, using only observed
  attempts. Each authorization worker owns independent in-memory state.
- Model artifacts, feature ordering, and preprocessing metadata must agree.
- Scores and merchant actions remain simulated; measured CPU results and
  accelerator deployment plans must be described separately.
