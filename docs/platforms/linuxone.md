# IBM LinuxONE setup and crash recovery

IBM LinuxONE uses the `s390x` CPU architecture. Check the server with `uname -m`.
The Ubuntu `x86_64`, Windows, and Apple silicon results in the project overview
do not establish compatibility with `s390x`. These recovery changes have not
been verified on a LinuxONE server; the exact cause of the reported segmentation
fault remains unconfirmed without a traceback from that server.

Run all commands from the repository root using the updated source code.

## Update NumPy and check the environment

After the updated source has been pushed to GitHub, pull it on the server and
install the project's pinned NumPy version:

```bash
git pull --ff-only
.venv/bin/python -m pip install 'numpy==2.2.6'
.venv/bin/python -m pip install -e './detection[dev,api]'
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy; print(numpy.__version__)'
```

The last command must print `2.2.6`. Keep native thread limits in place for
diagnostics and tests, including libraries that tests import before the detector:

```bash
export SCALPER_SAFE_MODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
```

## Retry training with the conservative CPU runtime

Use the existing dataset and environment first:

```bash
.venv/bin/python -X faulthandler -m scalper_detector --safe-mode train-simulation \
  --simulation generation_and_testing/data/simulation \
  --output artifacts/ticket_model \
  --epochs 20 \
  --threads 1 \
  --batch-size 32
```

Safe mode configures native thread limits before numerical imports, uses one
CPU thread, disables the optional MKLDNN backend, and evaluates the LSTM through
ordinary tensor operations. The LSTM retains the same parameters and saved
weight names. The forest also uses one thread. This reduces exposure to native
threading and fused recurrent kernels; it does not repair an incompatible or
broken dependency installation.

Safe mode is enabled automatically on Linux `s390x`. The explicit flag makes
the choice visible and also works on other architectures. For processes such as
the API, enable the same profile before starting Python:

```bash
export SCALPER_SAFE_MODE=1
```

`SCALPER_SAFE_MODE=0` opts out of the automatic profile for a new process.
Keep the conservative profile enabled while investigating this crash.

## Identify the failing native operation

If training still crashes, run the diagnostics with the same interpreter:

```bash
.venv/bin/python scripts/diagnose_runtime.py \
  --safe \
  --output artifacts/linuxone_runtime.json
```

The diagnostic launcher uses only the standard library and runs probes in
separate child processes, so a native crash can be reported with its failing
stage, exit status or signal, and captured output. The report also records the
platform and installed dependency information. Keep the final training output
and `artifacts/linuxone_runtime.json` together when reporting a failure.

An import or basic matrix multiplication crash points to the installed native
runtime and must be resolved before full training can work. A successful
diagnostic run checks the exercised operations; the complete training command
above still needs to finish on the server to establish that the reported
failure is resolved. `-X faulthandler` makes fatal Python stack output available
if a native library terminates training.

## Run the regression tests and recheck the saved model

```bash
.venv/bin/python -X faulthandler -m pytest -q
.venv/bin/python -X faulthandler -m scalper_detector --safe-mode evaluate \
  --data artifacts/ticket_model/test_orders.csv \
  --model artifacts/ticket_model \
  --output artifacts/linuxone_evaluation.json
```

Tests that explicitly compare against native recurrent kernels skip in compatibility
mode; those comparisons still run in ordinary workstation CI. ONNX checks also skip
when optional deployment dependencies are absent. For a sustained
20-epoch training check, add `--patience 20` to the training command; otherwise
early stopping may finish before epoch 20.

## Optional: create a separate environment

If the diagnostics indicate a dependency problem, create a new environment to
compare with the existing installation. Use a Python 3.11 or 3.12 interpreter
installed for this server. Do not copy a virtual environment from another
computer.

The official [PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
lists `torch-2.8.0+cpu` wheels for Linux `s390x`, including Python 3.11 and 3.12.
This version satisfies the project's `torch>=2.6,<3` requirement. Availability
of a wheel does not establish compatibility with every distribution or prove
that replacing PyTorch resolves the crash.

```bash
python3.12 -m venv .venv-linuxone
.venv-linuxone/bin/python -m pip install --upgrade pip
.venv-linuxone/bin/python -m pip install 'torch==2.8.0' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-linuxone/bin/python -m pip install -e './detection[dev,api]'
.venv-linuxone/bin/python -m pip check
.venv-linuxone/bin/python scripts/diagnose_runtime.py \
  --safe \
  --output artifacts/linuxone_fresh_runtime.json
```

Substitute `python3.11` if that is the available interpreter. If a dependency
has no matching wheel, installation may require distribution packages and
compilers or a maintained environment supplied by the server administrator.
Keep the installation output if it fails. After the probes pass, retry training
with `.venv-linuxone/bin/python` in place of `.venv/bin/python`.

If working inside an IBM supplied PyTorch container, preserve its installed
framework and follow that image's version requirements. IBM warns against
replacing the framework in its accelerated container. The separate environment
above is an alternative installation, not an instruction to modify that image.
[IBM container guidance](https://github.com/IBM/ibmz-accelerated-for-pytorch#q-can-i-install-a-newer-or-older-version-of-pytorch-in-the-container).

## Start the API after training completes

Use the environment in which training succeeded:

```bash
SCALPER_SAFE_MODE=1 .venv/bin/python -X faulthandler \
  -m uvicorn datathon_authorization.authorization.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

The API loads `artifacts/ticket_model` by default. Follow the
[Linux guide](linux.md#5-run-a-demo-or-replay-in-terminal-2) for replay and the
live demo. Use one API worker and restart between independent request streams.

## CPU training and Telum inference

LinuxONE CPU training and Telum/NNPA inference are separate execution paths.
IBM's accelerated PyTorch documentation says NNPA training is unsupported and
recommends training on CPU. [IBM training workload guidance](https://github.com/IBM/ibmz-accelerated-for-pytorch#training-workloads).

This project trains on CPU. Its normal API also uses the CPU model; it does not
automatically load a compiled Telum model. After training succeeds, follow the
[Telum deployment guide](../../detection/docs/telum_deployment.md) for optional
export, compilation, and target measurement.
