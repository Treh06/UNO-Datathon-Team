# IBM LinuxONE / Linux on IBM Z setup

Use this guide for Ubuntu 24.04 on LinuxONE 5 and other approved **s390x** Linux environments. LinuxONE is the server family; Ubuntu is the OS. Telum II provides both CPU execution and on-chip AI acceleration, which require separate validation.

The repository has no recorded s390x end-to-end test result or Telum performance measurement. These instructions establish a controlled setup and verification path; they do not assert that an arbitrary package combination or image is compatible.

## 1. Identify the machine and package source

On the IBM host:

```bash
uname -m
cat /etc/os-release
python3 -c "import sys, platform; print(sys.version); print(sys.executable); print(platform.machine()); print(sys.byteorder)"
```

Expect s390x. Record the installed interpreter rather than assuming the desktop CI's Python 3.13 is the correct IBM choice. Confirm the host OS/kernel/service levels with the machine administrator and [IBM's platform support matrix](https://www.ibm.com/support/pages/linux-ibm-zibm-linuxone-tested-platforms).

Do not copy a Windows/x86 environment or install x86 wheels. The [standard PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) is not a documented s390x installation route for this project. The desktop Linux commands and CI results do not establish IBM compatibility.

Choose one approved native stack:

- An IBM CPU PyTorch container plus compatible native NumPy/pandas/scikit-learn dependencies.
- An administrator-provided, validated s390x environment/wheel source.

Start with CPU execution. [IBM's PyTorch documentation](https://github.com/IBM/ibmz-accelerated-for-pytorch) distinguishes the `ibmz/pytorch` CPU image from `ibmz/ibmz-accelerated-for-pytorch`. The accelerated image has additional inference restrictions and is not the starting point for ordinary training.

## 2. Select an image with the administrator

Confirm the image's Python, PyTorch, NumPy, and other framework versions against [pyproject.toml](../../detection/pyproject.toml). The project requires torch >=2.6,<3, NumPy >=2,<3, pandas >=2.2,<3, and scikit-learn >=1.6,<2. A container name alone does not guarantee these versions. Obtain registry access and an approved tag/digest through the [IBM AI Toolkit](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

The example below uses Podman. Replace the placeholder with an approved **CPU** tag before running. Run from the checkout root on the IBM host; ensure host port 8000 is available and no old container named `payment-demo` is running. Docker users can substitute `docker` for `podman`.

```bash
export PAYMENT_CPU_IMAGE='icr.io/ibmz/pytorch:REPLACE_WITH_APPROVED_TAG'
podman pull "$PAYMENT_CPU_IMAGE"
podman run --rm -it --name payment-demo -p 127.0.0.1:8000:8000 -v "$PWD:/workspace" -w /workspace "$PAYMENT_CPU_IMAGE" /bin/bash
```

The source, datasets, and model folders remain on the host through the mount. Container changes under `/tmp` disappear when it exits. The host's existing `.venv` may belong to another platform; do not use it inside the container.

## 3. Preserve the container's native framework versions

Inside the container, inspect the supplied stack before installing project extras:

```bash
python3 -B -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.__file__)"
python3 -m pip check
```

Create constraints from installed native packages so installing the project cannot silently replace them:

```bash
python3 -c "import importlib.metadata as m; names={'torch','numpy','pandas','scipy','scikit-learn'}; print('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'].lower() in names)))" > /tmp/payment-native-constraints.txt
cat /tmp/payment-native-constraints.txt
python3 -m venv --system-site-packages /tmp/payment-env
/tmp/payment-env/bin/python -m pip install --only-binary=:all: -c /tmp/payment-native-constraints.txt -e './detection[api]'
/tmp/payment-env/bin/python -m pip check
```

This container-only environment intentionally sees the approved image's packages. A desktop `.venv` normally remains isolated. If the image lacks venv support, ask the administrator for a matching environment method/image rather than modifying its framework blindly.

These are first-setup commands in a new container. If `/tmp/payment-env` already exists, inspect/reuse it or start a fresh container; do not recreate it over old packages. After stopping/removing the old named container, repeat the approved setup in the new one.

If constraints conflict or a native s390x wheel is unavailable, **stop**. Select a compatible image or approved package source with the administrator. Do not remove the constraints to make pip replace the framework, use `--no-deps` to conceal incompatibilities, or downgrade NumPy below the project's declared minimum. Optional Ruff/ONNX packages may have separate s390x availability limits.

## 4. Validate before training

Use the [native checks](../environment.md#check-native-packages-before-the-full-pipeline), replacing `.venv/bin/python` with `/tmp/payment-env/bin/python`. Run them one at a time. Include pandas datetime conversion and basic NumPy/PyTorch arithmetic, not just imports or `pip check`.

If runtime checks pass, install compatible test tools without replacing the pinned native stack:

```bash
/tmp/payment-env/bin/python -m pip install --only-binary=:all: -c /tmp/payment-native-constraints.txt 'pytest>=8,<10' 'httpx>=0.27,<1'
/tmp/payment-env/bin/python -B -X faulthandler -m pytest -q
```

Tests build temporary models. Capture the actual result on this hardware. Desktop lint checks can be run separately; Ruff is not needed to serve the trained model.

For a segfault, retain the failing command, full fault-handler output, package locations/versions, architecture, image digest, and package source. Check data loading, feature construction, LSTM operations, and forest training separately. A crash inside pandas is an investigation lead, not a confirmed root cause. Do not add x86-specific SIMD workarounds on s390x.

## 5. Generate/train and run the demo

Inside the container at `/workspace`, follow the [root walkthrough](../../README.md#run-the-complete-demo), replacing each `.venv/bin/python` with `/tmp/payment-env/bin/python`. Generate/train there, or copy a complete compatible model from another machine and verify it with the same ordered data before presenting.

For the published host port, start the API **inside the container** with:

```bash
/tmp/payment-env/bin/python -m uvicorn datathon_authorization.authorization.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The container listens on its interfaces, while the port mapping binds only the IBM host's loopback address. Do not expose this unauthenticated demo directly to the network.

In terminal 2 on the IBM host, enter the same running container:

```bash
podman exec -it -w /workspace payment-demo /bin/bash
/tmp/payment-env/bin/python -m live_demo
```

The client defaults to `http://127.0.0.1:8000` inside that container. Restart the API process before each independent demo/replay. Keep one worker and one chronological producer. Exiting the container removes `/tmp/payment-env`; recreate it with the recorded setup next time, while source/data/model outputs under `/workspace` persist.

For laptop viewing or laptop-generated requests, see [SSH access and team handoff](../../detection/docs/git_workflow.md#use-the-ibm-api-from-a-laptop). Do not transfer the environment itself.

## 6. Treat acceleration as a separate step

CPU success establishes a baseline, not Telum use. The existing ONNX exporter supports a zDLC integration path, but the service does not load the compiled library yet. IBM's accelerated PyTorch is another possible inference route; exact operations and tensor shapes still need validation, and IBM warns against replacing the bundled PyTorch.

Keep training on the validated CPU path and compare accelerated inference with the baseline before making speed or accuracy claims. Follow [Telum deployment and measurement](../../detection/docs/telum_deployment.md).
