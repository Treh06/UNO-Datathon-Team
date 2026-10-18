# IBM deployment and Telum measurement

First establish reliable CPU execution using the [LinuxONE setup](../../docs/platforms/linuxone.md). This guide covers the next step: enabling and measuring inference acceleration. No compiled library, s390x test result, or Telum performance measurement is bundled with the repository.

IBM z17 and LinuxONE 5 use Telum II. LinuxONE is a hardware family, not an OS; run the project in an approved Linux/s390x environment. Copy source/models through the [team handoff](git_workflow.md), and recreate all native environments.

## Choose the execution path

| Path | Project status and next work |
| --- | --- |
| Validated s390x CPU PyTorch | Baseline to establish on the actual machine; desktop tests do not prove it. |
| ONNX compiled with IBM zDLC | Exporter implemented; target compilation, runtime integration, placement, and hardware measurement pending. |
| IBM Z Accelerated for PyTorch | Possible alternative inference route; exact model/operator behavior and placement remain unverified. |

The service currently calls PyTorch; it does not load a compiled zDLC library. Installing ordinary PyTorch on IBM hardware does not establish accelerator use. IBM's adapted runtime can target eligible operations, but that must be observed for this model.

The neural LSTM and supported arithmetic are acceleration candidates. Feature/history processing, the current NumPy forest, calibration, explanations, policy, and HTTP work remain outside the exported neural graph. End-to-end improvement depends on all these costs.

## Export the trained neural branch

On a machine with a valid environment and compatible optional ONNX wheels, install deployment extras through its platform guide. From the repository root:

```bash
.venv/bin/python -m scalper_detector export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

Windows uses `.venv\Scripts\python.exe`. The exporter writes the graph and `scalper_lstm.manifest.json`. Keep those with the complete source model directory. The graph returns uncalibrated neural logits, not the final ensemble probability or merchant decision.

Inputs are float32 `features[B,T,N]` and int64 `lengths[B]`. Read T/N from the manifest; do not assume a feature width. Histories are right padded and lengths identify the last valid order. The current schema has 42 numeric fields plus learned category/card-type encodings, so width depends on training.

Inspect `onnx_checker_passed` and `cpu_numerical_parity.verified`. The CPU checks cover several batch sizes/history lengths. They establish implementation agreement, not accelerator compatibility or real-world model accuracy.

## Compile with zDLC on the target

Arrange registry access, container tooling, and the approved compiler/runtime version with the IBM administrator. Follow [IBM zDLC instructions](https://github.com/IBM/zDLC), including supported ONNX operators and runtime libraries. The following is a template, not a recorded successful compile.

Set the image tag approved for your machine. Version 5.1.0 is an example; keep the version/digest and logs with the results. Use the actual complete model directory and its exported manifest:

```bash
export PAYMENT_ZDLC_IMAGE='icr.io/ibmz/zdlc:5.1.0'
export PAYMENT_MODEL_DIR="$(realpath artifacts/ticket_model)"
export PAYMENT_SHAPES="$(.venv/bin/python -c "import json; m=json.load(open('artifacts/ticket_model/scalper_lstm.manifest.json')); s=m['inputs']['features']['shape']; print('0:1x{}x{},1:1'.format(s[1],s[2]))")"
docker pull "$PAYMENT_ZDLC_IMAGE"
docker run --rm -v "$PAYMENT_MODEL_DIR:/workdir:z" "$PAYMENT_ZDLC_IMAGE" --EmitLib --O3 -march=z17 --mtriple=s390x-ibm-loz --maccel=NNPA --onnx-op-stats=JSON --shapeInformation="$PAYMENT_SHAPES" scalper_lstm.onnx
```

If exporting on a laptop and compiling on IBM, transfer the model folder first. Replace the `.venv/bin/python` above with the validated IBM interpreter when reading the manifest there. Podman can replace Docker where supported. Update the manifest-reading path together with `PAYMENT_MODEL_DIR` if using another model folder.

Check compiler help and the emitted signature. Inspect operator placement, particularly LSTM and last-valid-timestep indexing. Some operations can remain on CPU. Successful compilation alone does not prove NNPA use.

## Integrate and validate the compiled model

A compiled `.so` is architecture/runtime-specific; do not copy a Windows or x86 library to s390x. Use IBM's [Python runtime example](https://github.com/IBM/zDLC#running-the-python-example) and match the exported input/output contract.

Integration requires a code change to the neural call in `OrderScoringSession`; that change is not implemented by the export command. Preserve this order:

1. Validate the request and build causal card/BIN/linkage features.
2. Apply the exact saved preprocessing and form the history tensor/length.
3. Invoke the compiled neural branch and the existing forest.
4. Apply saved branch temperatures, ensemble weight, and final calibration.
5. Produce the existing score, reasons, confidence, and merchant advice.

Compare logits and final decisions with CPU across every valid history length, missing optional inputs, unseen categories, linked-card bridges, and busy releases. Account for accelerator precision differences and check orders near decision thresholds. Reject non-ticket input before inference. Keep token namespaces and state identical in both comparisons.

## Alternative: IBM's accelerated PyTorch

[IBM Z Accelerated for PyTorch](https://github.com/IBM/ibmz-accelerated-for-pytorch) provides an inference-oriented runtime with supported operations routed to the accelerator and CPU fallback for others. It may reduce custom integration work, but our packed-history LSTM and complete request path still need verification.

Use the supplied framework version; IBM advises against replacing PyTorch inside that image. IBM distinguishes CPU training from accelerated inference. Start with the [CPU setup](../../docs/platforms/linuxone.md), then assess this runtime with the administrator. Do not assume that a `torch>=2.6` pip package is equivalent to IBM's adapted build.

## Measure the actual benefit

Use matching CPU and accelerated runs on the same hardware, model, ordered inputs, and state. First compare numerical results, then neural invocation and full-request time.

Record hardware/OS, image/compiler/runtime versions, input shapes, operation placement, precision, warmup, sample count, p50/p95/p99 latency, offered/completed rates, errors, queue delay, and fallback. Separate cold startup from warm inference.

The local benchmark and live demo are serial. Their burst settings do not establish concurrent capacity or a service-level target. A true concurrency test needs globally ordered, shared-state handling; simply adding API workers breaks the current state model.

Report accelerator speedup only after both paths are measured. Until then, describe target compilation/integration, accelerator placement, and hardware benchmarking as pending.
