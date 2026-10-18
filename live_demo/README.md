# Live purchase demo

Create synthetic purchases during a live run and submit them to the trained authorization API. The terminal displays actual API scores and simulated merchant decisions. Labels stay in separate local output files; no real payment processor is involved.

## Run

Complete [platform setup](../README.md#start-here) and train or transfer a complete model using the [root guide](../README.md#run-the-complete-demo). Tests do not create the saved presentation model. From the repository root, start a **fresh** API in terminal 1:

```bash
.venv/bin/python -m uvicorn datathon_authorization.authorization.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

In terminal 2:

```bash
.venv/bin/python -m live_demo
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\Scripts\python.exe` and put multiline commands on one line. Start both terminals in the repository root. If the model lives elsewhere, set `SCALPER_MODEL_DIR` in terminal 1 before starting the API.

Wait for `/health` to report `ready: true`; `/stats` should report zero processed requests. Leave terminal 1 running. After a run, stop/restart the API before starting another independent demo or replay. To stop the live producer, press Ctrl+C in terminal 2; stopping it does not reset API history.

The default is a **two-minute demonstration** with an average of two purchases per second, about 25% scalper purchases, and a fivefold release surge from seconds 30 through 50. Poisson arrivals create irregular gaps rather than a fixed metronome. These are expected rates and proportions, not exact counts. Ctrl+C stops and saves the partial summary.

```bash
.venv/bin/python -m live_demo \
  --duration 180 --rate 3 --scalper-ratio 0.3 \
  --drop-after 45 --drop-duration 30 --drop-multiplier 4 --seed 27
```

Use `--help` for all settings. Other controls are `--customers`, `--merchants` (at least two), `--events`, `--timeout`, `--url`, and `--output`. Set `--drop-multiplier 1` to disable the surge. The default output is a new timestamped folder under `artifacts/live_demo/`; an explicit output folder must not already exist.

The terminal displays scheduled elapsed time, release phase, simulated behavior, **actual API decision**, risk score, HTTP round-trip time, schedule delay, and top reasons. The truth column is a presenter diagnostic; it is never sent to the model. Look for changes in risk as repeated attempts build history. Actual scores depend on your saved model, thresholds, traffic, and warm-up; the demo does not guarantee every scalper is caught or every legitimate buyer approved.

## Traffic behavior

Returning legitimate customers reuse cards and identity tokens. Households share addresses/devices, and some persistent family/corporate customers buy four to six tickets. Negative examples overlap suspicious behavior.

Four interleaved campaigns continue across requests:

- Shared-address rings rotate multiple cards through one address.
- Bulk campaigns repeatedly buy near an eight-ticket quantity.
- Card-testing campaigns make small probes on recently issued prepaid/virtual cards, then larger purchases.
- Cross-merchant campaigns reuse a card at multiple ticket sites.

Each campaign finishes after 18–30 of its own orders and is replaced with a new synthetic actor. Card-testing probes precede its larger purchases even while other buyers are interleaved. During the release window, purchases focus on event zero. Low rates, short runs, or low scalper ratios may not complete every campaign or show every tactic.

The seed repeats the generated random sequence. Current UTC timestamps and a new run namespace intentionally differ between runs. The traffic module supports fixed `start` and numeric `run_id` for deterministic tests. All card IDs are token strings, BIN metadata is fabricated, and IP ranges are reserved documentation ranges.

## Outputs

| File | Contents |
| --- | --- |
| `requests.jsonl` | Label-free attempted requests in submission order |
| `ground_truth.jsonl` | Separate campaign labels joined by `txn_id` |
| `responses.jsonl` | Unmodified successful API responses |
| `timings.jsonl` | HTTP round-trip and schedule-delay measurements, or request error |
| `manifest.json` | Configuration, run namespace, start time, endpoint, and final status |
| `summary.json` | Decision/tactic counts, response-based precision/recall, per-tactic recall or false positives, confusion matrix, latency/delay percentiles, completion status |

Precision and recall use `flagged_for_review`, which can differ from APPROVE/REVIEW/DECLINE merchant-policy cutoffs. Metrics cover only saved successful responses. The attempted count can be larger if a request fails. Latency measures client HTTP round-trip, not individual feature/inference stages. Percentiles use the nearest observed rank.

## Ordering and limits

The current API requires **global chronological order**, not just ordering per card. This client therefore keeps one request in flight. Event timestamps represent scheduled purchase times anchored to the run's UTC start; monotonic time paces delivery and protects scheduling from system-clock changes. Request history advances with simulated purchase times, including when delivery is late.

If scoring cannot keep up, schedule delay increases. The demo stops dispatching at the configured duration and reports `capacity_limited` when pending arrivals remain; it does not silently claim the requested throughput was achieved. An in-flight request can run past that deadline up to its timeout. A normal run exits 0, capacity limitation exits 3, Ctrl+C exits 130, and failures exit 2.

Before sending purchases, the CLI checks readiness and requires empty API history. Restart the API before each independent demonstration, and do not run another producer against it simultaneously. API state is local to one worker. This is a demo of real-time transaction processing, not a concurrent load benchmark or confirmed-purchase accounting.

On HTTP/network failure, sending stops and partial logs remain. Requests are not automatically retried: after a timeout the server may already have observed the attempt. Restart before another run. Synthetic results do not establish accuracy on real payment traffic.

## Present from another machine

The producer itself uses standard-library Python; it does not load the detector or read training data. It can run on a laptop while the API/model runs on IBM. Use the [SSH-forwarding procedure](../detection/docs/git_workflow.md#use-the-ibm-api-from-a-laptop), then pass `--url http://127.0.0.1:8000` (or the forwarded port).

For a producer-only laptop with an appropriate Python already installed, `python3 -m live_demo` or `py -3 -m live_demo` from the repository root avoids installing the numerical stack there. Full training/API execution still requires the platform's environment setup. Keep clocks synchronized and do not copy the server's virtual environment to the laptop.

## Quick troubleshooting

| Problem | Resolution |
| --- | --- |
| Preflight reports used history | Restart the API process before this independent run. |
| API not ready/connection refused | Check its terminal, trained model, machine address, and port/SSH tunnel. |
| Output directory already exists | Choose a new `--output`, or omit it for an automatic timestamped folder. |
| `capacity_limited` / increasing delay | Lower `--rate` or surge multiplier; serial scoring has fallen behind. |
| Too few tactic examples | Increase duration/rate or scalper ratio; proportions and arrivals are random. |
| Native dependency crash on the server | Use [environment diagnostics](../docs/environment.md), especially the [s390x guide](../docs/platforms/linuxone.md) on IBM. |

## Tests

```bash
.venv/bin/python -m unittest live_demo.test_live_demo -v
.venv/bin/python -m pytest -q
```

Focused checks exercise arrival scheduling, bursts, continuing campaign behavior, label isolation, deterministic inputs, pacing, partial failures, preflight checks, and overload reporting without waiting in real time. The API integration test sends online traffic through a trained detector and compares its decisions to offline scoring of the same ordered stream.
