# Python environments and recovery

Use this guide when installing on another machine, changing Python versions, or diagnosing dependency errors. Run commands from the repository root. This guide changes no model or dataset files; environment-reset commands affect installed packages only.

## What must match

A native package must match the operating system, CPU architecture, Python implementation/version, and binary interface. For example, `cp312-cp312-win_amd64` means a CPython 3.12 Windows x64 binary. It is not a CPython 3.13 binary. Some packages use a compatible stable ABI, but these version-specific wheels do not.

An existing environment is reusable only with its original compatible interpreter and location. Changing branches does not update installed dependencies. After a dependency declaration changes, reinstall the project in the current valid environment and rerun checks.

[Python documents that existing venv directories are reused](https://docs.python.org/3.13/library/venv.html). Running `python -m venv .venv` again is not a clean reinstall.

## Inspect before changing anything

Windows PowerShell:

```powershell
Get-Location
Get-Content .venv\pyvenv.cfg
.venv\Scripts\python.exe -B -c "import sys, platform, struct; print(sys.executable); print(sys.version); print(platform.machine()); print(struct.calcsize('P') * 8)"
.venv\Scripts\python.exe -B -m pip --version
.venv\Scripts\python.exe -B -m pip check
.venv\Scripts\python.exe -B -m pip debug --verbose
```

Linux/macOS:

```bash
pwd
uname -m
cat .venv/pyvenv.cfg
.venv/bin/python -B -c "import sys, platform, struct; print(sys.executable); print(sys.version); print(platform.machine()); print(struct.calcsize('P') * 8)"
.venv/bin/python -B -m pip --version
.venv/bin/python -B -m pip check
.venv/bin/python -B -m pip debug --verbose
```

The interpreter path should be in this checkout. The IDE's selected interpreter must be the same one. On Windows use “Python: Select Interpreter” in VS Code and choose this checkout's `.venv\Scripts\python.exe`; on Linux/macOS choose `.venv/bin/python`. Open a new terminal after changing IDE interpreter settings.

`pip check` checks dependency metadata and wheel compatibility. It does not run native numerical operations. Passing it is necessary but does not establish runtime stability.

## Rebuild a broken environment

Use this after changing Python minor versions, moving the checkout, copying an environment, or finding incompatible wheels. Do not try to fix a 3.12/3.13 binary mismatch by editing `pyvenv.cfg`, renaming wheel files, or merely upgrading pip.

1. Stop the API, notebooks, tests, and IDE debugging sessions using the environment. If activated, run `deactivate` or open a fresh terminal.
2. Confirm the repository root and inspect `.venv/pyvenv.cfg`. Check that `.venv` is an ordinary directory inside this checkout, containing only the environment, with no source/data stored there.
3. Choose preservation or reset below. Then follow the platform guide's package-install steps again.

### Preserve the old files first

This moves the old environment under ignored `artifacts/` for inspection. A moved environment is **not runnable**; the backup preserves package/version evidence, not a working installation. If the backup destination exists, choose another suffix.

Windows PowerShell:

```powershell
if (Test-Path -LiteralPath artifacts/environment-backups/venv-before-rebuild) { throw 'Backup already exists; choose another backup name.' }
New-Item -ItemType Directory -Force -Path artifacts/environment-backups | Out-Null
Move-Item -LiteralPath .venv -Destination artifacts/environment-backups/venv-before-rebuild -ErrorAction Stop
py -3.13 -m venv .venv
```

Linux/macOS, using the intended installed interpreter:

```bash
mkdir -p artifacts/environment-backups
if [ -e artifacts/environment-backups/venv-before-rebuild ]; then
  printf '%s\n' 'Backup already exists; choose another backup name.'
else
  mv .venv artifacts/environment-backups/venv-before-rebuild && python3.13 -m venv .venv
fi
```

Stop if the move failed; do not run `venv` over the old directory. For IBM, use the interpreter selected for the validated s390x stack instead of assuming Python 3.13.

### Clear the old environment instead

Only use these commands after the directory check above and when you do not need its installed packages. **`--clear` deletes the contents of `.venv`.** It does not delete the project's separate `data/` or `artifacts/` directories. Do not change the target to the repository root.

Windows PowerShell:

```powershell
py -3.13 -m venv --clear .venv
```

Linux/macOS:

```bash
python3.13 -m venv --clear .venv
```

Now reinstall using [Windows](platforms/windows.md#install), [Linux x86-64](platforms/linux.md#install), or [macOS](platforms/macos.md#install). IBM users follow [LinuxONE](platforms/linuxone.md). Do not restore old `site-packages`.

Keep the canonical environment at `.venv` so the run commands work unchanged. To test in another environment instead, place it under ignored `artifacts/envs/` and substitute its full Python path in every command.

## Check native packages before the full pipeline

Run each command separately so a crash identifies a smaller area. Linux/macOS examples follow; Windows replaces the interpreter with `.venv\Scripts\python.exe`.

```bash
.venv/bin/python -B -X faulthandler -c "import numpy as np; print(np.__version__, np.__file__, flush=True); print(np.ones((4, 4)) @ np.ones((4, 4)), flush=True)"
.venv/bin/python -B -X faulthandler -c "import pandas as pd; print(pd.__version__, pd.__file__, flush=True); print(pd.to_datetime(pd.Series(['2026-11-05T17:00:00.412Z']), utc=True, format='mixed'), flush=True)"
.venv/bin/python -B -X faulthandler -c "import torch; print(torch.__version__, torch.__file__, flush=True); print(torch.randn(2, 3) @ torch.randn(3, 2), flush=True)"
.venv/bin/python -B -X faulthandler -c "import sklearn, scipy; print(sklearn.__version__, scipy.__version__, flush=True)"
```

Then run `python -m scalper_detector --help` with the same environment interpreter, followed by the root guide's tests and training. That CLI imports the numerical stack even for help.

If training segfaults, rerun its exact command with the environment's Python and `-B -X faulthandler -m scalper_detector`. For example:

```bash
.venv/bin/python -B -X faulthandler -m scalper_detector train-simulation --simulation generation_and_testing/data/simulation --output artifacts/ticket_model_diagnostic --epochs 1
```

Choose a new diagnostic output path to preserve earlier models. Record the command, last output, fault-handler trace, package versions/locations, `uname -m`, OS release, and whether packages came from pip, system packages, conda, a source build, or an IBM image. A pandas frame in the trace narrows the stage; it does not prove pandas caused memory corruption.

When imports pass but math crashes, inspect `numpy.show_runtime()` and native BLAS/OpenMP libraries. Thread limits can be a controlled diagnostic experiment, not an assumed permanent fix. Follow [NumPy troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html); do not apply x86 SIMD settings to s390x. Do not blindly downgrade to NumPy 1.x: this project declares `numpy>=2`.

## Pytest temporary-folder permissions

If tests report `PermissionError: [WinError 5]` on `Temp/pytest-of-*`, the current account cannot access pytest's shared scratch directory. Many tests can then fail during setup before model code runs. Reinstalling numerical packages does not repair this folder's permissions.

The root `pytest.ini` uses `--basetemp=.pytest-tmp` and `cache_dir=.pytest-cache` so the normal root test command creates fixture/cache files in this checkout. These directories are ignored by Git and their parent (the repository root) already exists. Run tests using the same account as your terminal. **Pytest clears the exact fixture scratch directory before a run**; never store source, datasets, or saved presentation models there. The project's separate model and dataset folders are unaffected.

If that project scratch folder was also created by another account, or if running concurrent sessions, override it with a fresh dedicated path. From the repository root in PowerShell:

```powershell
$pytestTemp = '.pytest-tmp-' + [guid]::NewGuid().ToString('N')
.venv\Scripts\python.exe -m pytest -c pytest.ini -q --basetemp $pytestTemp
```

An explicit command-line `--basetemp` overrides the configured default. Pytest requires its parent directory to exist; the example uses a flat name directly under the root to avoid missing-parent errors. Do not point it at the repository root, `artifacts` itself, or an existing model/data folder. Keep test sessions sequential when using the shared default. There is no need to run the whole project as Administrator or modify global temp-folder permissions for this workaround.

Use `-c pytest.ini` for focused tests as well as the full suite. Selecting a file under `detection/tests` can otherwise make pytest use the package's separate `pyproject.toml`, bypassing this root configuration.

## Keep a reproducible team record

After a successful run, save these next to your reports, under ignored `artifacts/`:

- Git commit, OS/architecture, Python version, and exact run commands/configuration.
- `python -m pip freeze` output from the successful environment, plus package source/index.
- For IBM containers, the image tag/digest and configured framework version.
- Dataset manifest, complete model directory, and evaluation outputs.

A freeze file is an environment record, not a cross-platform lockfile. This repository currently uses version ranges; it has no fully pinned platform-specific dependency lock. Revalidate resolution on each platform rather than copying native packages or asserting that a successful x86 run proves s390x compatibility.

See [team handoff](../detection/docs/git_workflow.md) for what to transfer.
