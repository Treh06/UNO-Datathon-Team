# Telum deployment and measurement

Start with a trained model from the [repository quick start](../../README.md) or your own labeled ticket orders. Run commands from the repository root. No transaction dataset, pretrained model, compiled shared library, or hardware measurement is bundled with the repository. The compiler command below is a template for target-environment work.

## Training on LinuxONE

For a LinuxONE server reporting `s390x`, follow the
[LinuxONE setup and crash recovery guide](../../docs/platforms/linuxone.md)
before exporting. It includes a conservative CPU training command and isolated
native runtime probes for segmentation faults. Existing Linux `x86_64`, Windows,
and Apple silicon validation does not validate LinuxONE; native `s390x`
training remains unverified.

CPU training is distinct from NNPA inference. IBM documents that training on
its NNPA device is unsupported and recommends CPU training. Preserve the
framework versions supplied with an IBM accelerated container rather than
replacing its PyTorch installation. [IBM training and container guidance](https://github.com/IBM/ibmz-accelerated-for-pytorch#training-workloads).

## Export the trained neural branch

```bash
.venv/bin/python -m pip install -e './detection[deploy]'
.venv/bin/scalper-detector export-onnx \
  --model artifacts/ticket_model \
  --output artifacts/ticket_model/scalper_lstm.onnx
```

For team-supplied data, use `train --data data/ticket_orders.csv --data-source user_supplied --output artifacts/ticket_model`, as described in the [detector guide](../README.md). Keep `model.pt`, the ONNX graph, its `.manifest.json`, `metadata.json`, `tree.json`, and model version together. The manifest records operator names, feature order, input shapes, and any numerical checks performed. The graph outputs a neural **logit**; calibration and merchant policy are applied separately.

The input contract is float32 `features[B,T,N]` and int64 `lengths[B]`. `T` defaults to seven history steps but follows the saved training configuration. The current numeric schema has 42 fields; learned category/card-type encodings determine the final width `N`. Read the exported manifest's `inputs.features.shape` and `feature_order`, and compare them with `metadata.json`. There is no included model with a fixed width. Histories are right padded; lengths identify the last real order. Chronological state and standardization remain outside the graph.

With ONNX Runtime installed, export checks CPU logit agreement at several valid history lengths and batch sizes. Inspect `onnx_checker_passed` and `cpu_numerical_parity.verified` in your own export manifest. These numerical checks, and automated tests using temporary synthetic tensors, verify implementation consistency; they do not validate the trained model's accuracy on real ticket orders or its compatibility with an accelerator.

## Compile on the target environment

IBM's instructions describe compiling ONNX to shared libraries, enabling NNPA using `--maccel=NNPA`, and inspecting operation placement. Unsupported operations or settings can run on CPU. Check the installed compiler's supported operators and target generation before treating a successful compile as acceleration. [Official IBM zDLC instructions](https://github.com/IBM/zDLC#ibm-z-integrated-accelerator-for-ai).

The following template applies **only if your exported manifest reports `T=7` and `N=48`, and your approved target is z17**. Otherwise replace `1x7x48` with the actual batch-one input shape and select the architecture/image for your target. For example, training on the companion simulator's default configuration currently learns 49 features because its card types include `virtual`; read the exported manifest instead of assuming 48. The `lengths` input remains a one-element vector for this batch-one example. Image access, supported operators, and target generation must be checked against the installed IBM compiler.

```bash
PAYMENT_ZDLC_IMAGE=icr.io/ibmz/zdlc:5.1.0
PAYMENT_MODEL_DIR=/absolute/path/to/artifacts/ticket_model

docker run --rm \
  -v "$PAYMENT_MODEL_DIR:/workdir:z" \
  "$PAYMENT_ZDLC_IMAGE" \
  --EmitLib --O3 -march=z17 --mtriple=s390x-ibm-loz \
  --maccel=NNPA --onnx-op-stats=JSON \
  --shapeInformation=0:1x7x48,1:1 \
  scalper_lstm.onnx
```

Check the compiler help and actual input signature before running. Save the compilation log and placement report with the model. Pay particular attention to the LSTM and the last-valid-timestep gather. A mixture of CPU and accelerator operations is possible; document it rather than attributing the whole request to Telum.

## Integrate the compiled branch

The current `OrderScoringSession` invokes PyTorch CPU. It does not load a compiled `.so`. Replace that one neural invocation with the target runtime after checking its input/output contract. IBM provides a Python runtime example as well as C/C++ and Java integration examples in the [zDLC repository](https://github.com/IBM/zDLC#running-the-python-example).

Retain the surrounding sequence:

1. Validate/tokenize an order and look up or update causal card, BIN, and linked-cluster state.
2. Apply the exact saved preprocessing and build the card's history tensor and length.
3. Invoke the compiled neural branch and CPU forest.
4. Apply saved branch temperatures, ensemble weight, and final temperature.
5. Return the score, confidence, observed evidence, and merchant's advisory policy result.

Compare compiled logits to the saved CPU model before comparing latency. Include histories from length 1 through your saved `T`, missing optional inputs, linked-card bridge events, unseen card types, and boundary-sized payment bursts. Confirm that non-ticket categories are rejected before neural invocation. Use the same token namespaces and graph history on both paths.

## Record a defensible hardware comparison

Create CPU-only and NNPA-enabled target builds from the same neural artifact. First measure isolated neural invocation, then the complete order path. Use matching input orders, feature state, batch size, precision, warmup, concurrency, and measurement boundaries.

Save hardware generation, compiler/image/runtime versions, operator placement, sample count, p50/p95/p99 latency, sustained offered/completed rates, errors, queueing, and fallback. Include cold starts separately from warm inference. Repeat a true concurrent drop-day load, with an agreed latency target and realistic feature-store/network costs.

The local `benchmark` command is a single-process CPU replay. It compresses event timestamps to a requested burst rate and estimates a serial queue from measured service times. It does not generate concurrent authorization traffic. Running that command on another host alone does not switch its neural backend to a compiled accelerator model.

Report a speedup only after both target paths have run under those conditions. Until then, record deployment as **target compilation, accelerator placement, and hardware benchmarking pending**. Describe CPU parity only when it is confirmed in the manifest produced from your trained model.
