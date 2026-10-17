# Live purchase demo

Create new synthetic purchases online and send them, paced by a live clock, to the existing trained-model authorization API. No training files are read and no scores or decisions are invented by the simulator. Requests use the original authorization fields; all labels stay in a separate local file. No real payment processor is involved.

## Run

First install the project and train a model using the [root guide](../README.md). From the repository root, start a **fresh** API in terminal 1:

```powershell
.venv\Scripts\python.exe -m uvicorn datathon_authorization.authorization.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In terminal 2:

```powershell
.venv\Scripts\python.exe -m live_demo
```

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. Start both terminals in the repository root. If the model lives elsewhere, set `SCALPER_MODEL_DIR` in terminal 1 before starting the API.

The default is a **two-minute demonstration** with an average of two purchases per second, about 25% scalper purchases, and a fivefold release surge from seconds 30 through 50. Poisson arrivals create irregular gaps rather than a fixed metronome. These are expected rates and proportions, not exact counts. Ctrl+C stops and saves the partial summary.

```powershell
.venv\Scripts\python.exe -m live_demo --duration 180 --rate 3 --scalper-ratio 0.3 --drop-after 45 --drop-duration 30 --drop-multiplier 4 --seed 27
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

## Verify

```powershell
.venv\Scripts\python.exe -m unittest live_demo.test_live_demo -v
.venv\Scripts\python.exe -m pytest -q
```

Focused checks exercise arrival scheduling, bursts, continuing campaign behavior, label isolation, deterministic inputs, pacing, partial failures, preflight checks, and overload reporting without waiting in real time. The API integration test sends online traffic through a trained detector and compares its decisions to offline scoring of the same ordered stream.
