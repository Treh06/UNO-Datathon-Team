# Ticket-scalper detection: generate, train, and score

This repository combines a payment-authorization simulator with a ticket-scalper detector. The simulator produces synthetic requests and separate ground truth. The detector learns payment patterns and returns a per-order risk score, observed evidence, and an advisory merchant action.

- [Generator guide](generation_and_testing/README.md): configuration, tactics, chronological splits, and JSON format.
- [Detector guide](detection/README.md): model behavior, input fields, merchant policy, and deployment limits.

Run the commands below **from the repository root**. The generator guide retains its original subfolder workflow. No generated dataset or trained model is bundled.

## Set up

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e './detection[dev]'
```

The simulator uses the Python standard library. The detector uses PyTorch, NumPy, pandas, and scikit-learn. Optional ONNX tools can be installed with `-e './detection[dev,deploy]'`.

## Generate a dataset

```bash
.venv/bin/python -m generation_and_testing.simulator.generate \
  --config generation_and_testing/configs/default.json \
  --output generation_and_testing/data/simulation
```

Choose a **new or empty output directory** for every run; the generator refuses to overwrite existing data. The default configuration produces 11,640 authorization requests. Each of `train/`, `validation/`, and `test/` contains `requests.jsonl` and a separate `ground_truth.jsonl`; `manifest.json` records configuration, times, and tactic counts.

## Train on the simulation

```bash
.venv/bin/scalper-detector train-simulation \
  --simulation generation_and_testing/data/simulation \
  --output artifacts/ticket_model \
  --epochs 20
```

This joins requests to truth **offline** using `txn_id`. Source training orders fit the model. Source validation is divided into early-stopping and calibration groups while keeping linked cards together. The generator's source test split stays the test split, including any configured held-out tactic. Inspect the saved reports for overall results, per-tactic recall, and false positives on legitimate bulk purchases.

Ground truth supplies `is_scalper`, `actor_id`, and `tactic` for offline learning and evaluation. Actor IDs, tactic names, and labels never become model input features. Synthetic metrics describe the generator's assumptions; they do not establish real ticket-scalper accuracy.

## Score requests without ground truth

```bash
.venv/bin/scalper-detector predict-authorizations \
  --requests generation_and_testing/data/simulation/test/requests.jsonl \
  --model artifacts/ticket_model \
  --output artifacts/scores.csv
```

Only authorization requests enter this scoring command. Supply the chronological history needed for useful velocity and linkage signals. Cross-site evidence requires an authorized payment feed and consistent tokens across sites; a single merchant's export cannot provide unseen activity elsewhere.

The adapter maps the simulator's JSON to detector fields:

| Authorization field | Detector field |
| --- | --- |
| `txn_id` | `order_id`, an identifier rather than a model feature |
| `event_ts` | UTC `timestamp` |
| `card_token`, `bin` | Tokenized `card_id`, `card_bin` |
| `ship_addr_hash`, `bill_addr_hash` | `shipping_hash`, `billing_hash` |
| `merchant_id`, `event_id`, `quantity`, `amount` | Corresponding order fields |

Amounts are already emitted in **USD units**, not cents. The simulator's `quantity_cap` is not an event-level per-buyer ticket limit. Request dates are not release dates. The adapter does not invent ticket limits or release timing from those fields; unavailable optional evidence remains unknown.

For an optional offline CSV conversion:

```bash
.venv/bin/scalper-detector convert-authorizations \
  --requests generation_and_testing/data/simulation/train/requests.jsonl \
  --ground-truth generation_and_testing/data/simulation/train/ground_truth.jsonl \
  --output data/train_orders.csv
```

Omit `--ground-truth` when converting requests for scoring. Never send ground-truth files to an authorization endpoint.

## Run both test suites

```bash
.venv/bin/python -m pytest -q
```

The root test configuration collects `detection/tests/` and `generation_and_testing/tests/`. Tests create their own temporary fixtures and models; they do not require a saved simulation or trained artifact.

## Interpret results and keep outputs local

Risk is a model estimate, confidence is heuristic evidence quality, and reasons describe observed inputs rather than proving resale intent. Shared addresses/devices can belong to legitimate families or corporate buyers. Merchant recommendations never execute a block, cap, cancellation, or refund.

The implemented scoring path runs on **CPU**. Optional neural ONNX export does not establish IBM Telum compatibility or acceleration; target compilation, placement, and hardware benchmarks remain pending. See the [deployment guide](detection/docs/telum_deployment.md).

Keep generated transaction data, models, `.venv/`, and evaluation outputs out of Git. The repository and generator ignore rules cover the default data/artifact locations; use ignored output directories for additional runs too.
