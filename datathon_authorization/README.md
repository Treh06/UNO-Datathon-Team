# Ticket authorization with the trained scalper detector

This FastAPI component connects the ticket-request generator to the PyTorch LSTM/random-forest detector. `POST /authorize` uses a saved model and a shared causal scoring session to return risk, evidence, and a simulated merchant decision. It does not contact a bank or execute a payment, cancellation, or refund.

## Start from the repository root

Install the shared environment and train a model using the [root quick start](../README.md). Then start the API from the folder containing `detection/`, `generation_and_testing/`, and `datathon_authorization/`:

```bash
.venv/bin/python -m uvicorn datathon_authorization.authorization.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\Scripts\python.exe` and put multiline commands on one line.

The API loads `artifacts/ticket_model` once at startup. To select another model, set `SCALPER_MODEL_DIR` before launching. Missing or incompatible artifacts stop startup with a load error.

Use one worker. State lives in memory and resets on restart. Interactive docs are at <http://127.0.0.1:8000/docs>.

## Request contract

Requests use the generator's `requests.jsonl` fields:

| Fields | Contract |
| --- | --- |
| `txn_id` | Unique transaction identifier; generator IDs and the original sample IDs are supported. |
| `event_ts` | Timezone-aware event time, normalized to UTC. Submit globally in nondecreasing time order. |
| `card_token`, `bin` | Required synthetic card token and six-digit BIN metadata. |
| `card_type`, `card_issue_date` | Optional metadata; missing values remain unknown, and issue date cannot be after the request time. |
| `merchant_id`, `mcc` | Ticket merchant and the simulator's ticket MCC `7922`. |
| `product_category` | `tickets`; defaults to tickets for generator files that omit it. Other categories are rejected. |
| `quantity`, `amount`, `currency` | Positive whole ticket quantity, finite positive USD amount, currency `USD`. Amounts accept JSON numbers or decimal strings and retain USD units. |
| `ship_addr_hash`, `bill_addr_hash`, `email_hash` | Optional identity tokens, consistently scoped across merchants; empty/missing tokens never link buyers. |
| `event_id`, `device_id`, `ip_prefix` | Optional source fields; missing events remain unknown and are not pooled into one event. |
| `channel` | Optional `web`, `app`, or `api` source metadata. |

Ground-truth fields (`is_scalper`, `actor_id`, `tactic`) and unknown extra fields are rejected. Fraud labels or retrospective campaign information never enter online features. The detector uses an explicit feature allowlist; tokens identify history and linkage but are not learned identity encodings.

## Response and decision policy

The response retains `txn_id`, `authorization_status`, `risk_score`, `reason_codes`, and `processed_ts`. It also exposes `scalper_probability`, `confidence_tier`, `recommended_action`, `flagged_for_review`, `decision_threshold`, and the complete detector response in `model_result`.

| Default model-policy recommendation | Simulated status |
| --- | --- |
| `allow` | APPROVE |
| `step_up_verification` or `quantity_cap_or_throttle` | REVIEW |
| `cancel_or_refund` | DECLINE |

The API uses the detector's default merchant action policy. The review flag uses the separately calibrated model threshold, so it can differ from the status. Status values are simulated decisions. The generator's `quantity_cap` describes synthetic buying behavior; it does not establish an event-specific buyer allocation.

## Endpoints and state

- `POST /authorize`: score one valid request and record its observed attempt context.
- `GET /health`: readiness and model provenance, including synthetic training status.
- `GET /stats`: processed request count and APPROVE/REVIEW/DECLINE counts.

One locked session shares card/BIN/linkage context across merchants. Duplicate IDs and out-of-order timestamps return **HTTP 409** before changing history. Invalid requests return **HTTP 422**. Equal timestamps are allowed and retain submission order. Unexpected inference failure marks the service unready and returns **HTTP 503**; restart rather than continuing with uncertain state.

All successfully processed requests, including simulated declines, contribute to **attempt history**. This is not confirmed-purchase accounting. Restart before replaying identical IDs, an independent source split, or a separate sample. Separate workers have separate state, so multiple workers would produce inconsistent histories.

## Replay and evaluate the test dataset

With a freshly started API in another terminal:

```bash
.venv/bin/python -m datathon_authorization.replay \
  --input generation_and_testing/data/simulation/test/requests.jsonl \
  --output artifacts/api_responses.jsonl \
  --ground-truth generation_and_testing/data/simulation/test/ground_truth.jsonl \
  --evaluation-output artifacts/api_evaluation.json
```

Replay accepts chronological JSONL or a JSON array. It validates the full input before sending anything, stops on HTTP/network failure, and saves successful responses as JSONL. Truth is read only locally after response capture, with an exact transaction-ID join for evaluation; it is never sent to the API. The report includes overall review-threshold metrics, per-tactic recall, and legitimate-bulk false positives. `--url` selects another local API address; the default is `http://127.0.0.1:8000`.

For the two-request smoke sample, restart the API and run:

```bash
.venv/bin/python -m datathon_authorization.replay \
  --input datathon_authorization/authorization/sample_transactions.json \
  --output artifacts/sample_api_responses.jsonl
```

Sample responses depend on the loaded model. The two requests are unlabeled and serve only as an HTTP smoke check.

## Test the integration

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest datathon_authorization/tests detection/tests/test_replay.py -q
```

Tests generate temporary data, train a real model, and compare HTTP responses to offline predictions. They check cross-site state, unknown events, input validation, replay, duplicate/late requests, and independent app instances. A running Uvicorn server is not needed for automated tests.

This local prototype uses in-memory state and a single worker. Authentication, persistent state, worker coordination, and Telum acceleration require deployment work. Validate model performance separately on representative authorized ticket traffic.
