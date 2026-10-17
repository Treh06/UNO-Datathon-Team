# Ticket-only implementation

The current task is detecting suspected ticket scalping from purchase authorizations across ticketing sites. The [preserved project abstract](project_abstract.txt) proposes a wider payment-side service; this implementation narrows its scope to concert, sports, and theatre tickets. Inline scoring and ranked post-purchase review are implemented locally. Real-world validation, production integration, and IBM accelerator measurements remain outstanding.

The repository contains source code, automated tests, merchant-policy configuration, and a companion authorization simulator. It does not include generated transaction datasets, saved prediction examples, or pretrained models. Train using the simulator's output or the team's ticket orders and reviewed labels.

## Companion authorization simulator

The simulator remains in `generation_and_testing/`; the detector lives in `detection/`. Follow the [repository quick start](../../README.md) for generation, `train-simulation`, and request-only scoring.

`authorizations.py` maps `txn_id` to `order_id`, `event_ts` to UTC time, `card_token`/`bin` to card identifiers, and source address hashes to canonical linkage fields. It accepts the simulator's declared ticket context (`mcc=7922`) and USD amounts in their existing units. Card age is derived from the supplied issue date at the request timestamp. Event allocation and release timing remain unknown when absent. The source `quantity_cap` is not an allocation input.

Request files cannot contain ground-truth labels, actor IDs, or tactics. An explicit offline truth-file argument joins labels one-to-one by transaction ID. Actor/tactic metadata is retained separately for split checks and evaluation, never encoded as features.

`simulation.py` verifies the manifest, source boundaries, row counts, labels, tactics, and identity/actor isolation. Original train and test populations remain intact, preserving chronological evaluation and held-out tactics. Only source validation is divided into linked-component-isolated early-stopping and calibration groups. BIN context is computed causally within each original source population before validation is subdivided. At least two positive and two all-negative validation components are needed. Saved `simulation_evaluation.json` reports per-tactic test recall and legitimate-bulk false positives; these synthetic results do not establish accuracy on real buyers.

## HTTP authorization and replay

`datathon_authorization.authorization.main` exposes a FastAPI app factory and the default app. Startup loads the ensemble from `SCALPER_MODEL_DIR` (default `artifacts/ticket_model`); missing/incompatible models stop startup. One locked session shares causal card, BIN, and linkage context across all merchants. The request model accepts generator IDs and omitted event IDs, rejects explicit non-ticket scope and truth fields, and normalizes timezone-aware times and decimal USD payments through the same adapter used by offline scoring.

Expected validation failures, duplicates, and late timestamps are rejected before updating state. Unexpected inference failure makes the app unready until restart, avoiding continued scoring from uncertain context. State is local to one worker and resets on restart. Simulated declines still enter attempt history; they do not represent confirmed purchases.

The response includes the detector's probability, confidence, evidence, calibrated review flag, and advisory action. APPROVE/REVIEW/DECLINE maps the default merchant action policy to a simulated authorization response; it does not reuse the review threshold as a bank decision or enforce an invented four-ticket limit. `datathon_authorization.replay` sends only validated requests, saves responses, then optionally joins exact transaction IDs to a separate truth file locally. API integration tests compare HTTP and offline probabilities on generated streams. See the [authorization guide](../../datathon_authorization/README.md) for commands and state boundaries.

## Train with team-supplied data

Run commands from the repository root after the [README setup](../README.md). These paths describe files the team supplies; they are not bundled with the project.

```bash
.venv/bin/scalper-detector train \
  --data data/ticket_orders.csv \
  --data-source user_supplied \
  --output artifacts/ticket_model

.venv/bin/scalper-detector predict \
  --data data/new_ticket_orders.csv \
  --model artifacts/ticket_model \
  --output artifacts/ticket_model/predictions.csv

.venv/bin/scalper-detector evaluate \
  --data artifacts/ticket_model/test_orders.csv \
  --model artifacts/ticket_model
```

Training creates the model and a held-out `test_orders.csv` from the supplied data. That file supports repeatable evaluation; it is not an included test dataset. For a later external check, supply separately reviewed future ticket orders to `evaluate` instead. Prediction does not require labels.

## Input schema

Each row is an authorization observable at its timestamp. Batch processing sorts stably by global time; live sessions require nondecreasing timestamps and reject duplicate order IDs.

| Field | Contract |
| --- | --- |
| `order_id` | Required unique authorization/order identifier. |
| `timestamp` | Required ISO 8601 order time; offsets convert to UTC, naive values mean UTC. |
| `card_id`, `card_bin`, `merchant_id` | Required nonempty tokenized identifiers; strings preserve leading zeros. BIN is population context, never buyer identity. |
| `category` | Required; stripped/lowercased to `tickets`. Other categories and mixed batches are rejected, including online and engineered-feature entry points. |
| `amount` | Required finite positive total payment, normalized to a consistent currency/unit. |
| `shipping_hash`, `billing_hash`, `email_hash`, `device_id` | Optional consistently scoped identity tokens. Missing/empty values never link cards. |
| `event_id` | Optional common performance ID across merchants; repeated values are expected. Missing becomes empty with `event_id_known=0`. |
| `quantity` | Optional positive whole ticket count; missing defaults to 1 with `quantity_available=0`. |
| `ticket_limit` | Optional positive whole event-level per-buyer allocation applicable across sites; missing defaults to 0 with `ticket_limit_known=0`. |
| `seconds_since_release`, `card_age_days` | Optional nonnegative values; missing defaults to 0 with `release_time_known=0` / `card_age_known=0`. |
| `card_type` | Optional category; missing becomes `unknown`; unseen values receive an unknown encoding. |
| `processor_bin_orders_1m`, `processor_bin_orders_5m` | Optional trusted processor-wide counts including the current order. Supply both positive whole counts or neither; five-minute count must be at least one-minute count. |
| `is_scalper` | Training/evaluation only: independently reviewed per-order 0/1 label. Labels may change on one card and are ignored during inference. |

Availability flags survive repeated validation. Missing quantity's placeholder 1 never contributes to ticket totals. `performance_type` and `performance_date` may accompany exports as metadata; they are not model features. Payment-fraud labels or chargebacks are not interchangeable with scalping labels.

## Causal ticket evidence

A union-find graph joins cards through observed nonempty address, email, and device tokens. Shipping/billing share an address namespace. A bridge merges already observed histories; earlier emitted features remain unchanged. Card and cluster histories retain 24 hours; one-hour counters explicitly filter that window.

Provided processor BIN counts override stream-local BIN counters in both batch and live scoring. They must use the same BIN definition, event-time boundaries, and current-order timestamp, with no later arrivals included. When absent, counts reflect only the supplied stream; narrow exports cannot reconstruct wider population activity.

The current schema has 42 numeric fields plus category/card-type encodings. Training learns the categorical vocabularies, so the encoded width depends on the supplied data. Read `feature_names` and `model_config.input_size` in the trained model's `metadata.json` for its exact feature order and width. Numeric evidence includes:

- Card/BIN/cluster velocity, distinct merchants, linked cards, timing regularity, amount uniformity, and optional-field availability.
- Same-performance order, merchant, and card counts over five minutes; observed tickets over one hour and 24 hours.
- Known-quantity coverage and observed 24-hour tickets exceeding the current supplied known event allocation.
- Distinct known performances, the current performance's share of known-target orders, and event-identity coverage over five minutes.

Only observed quantities contribute to totals. Unknown performance IDs never form a common event: current-event counters/ticket totals are zero, and concentration is zero. Diversity counts only known targets; identity coverage exposes missing targets. Other performances do not increase same-event counts. Shared tokens may represent several legitimate buyers, so cluster allocation excess is evidence for review, not proof of abuse.

The LSTM receives up to seven same-card current/prior orders, with true lengths and right padding. Graph features supply cross-card context. Offline prediction reconstructs fresh state from the complete supplied stream; `OrderScoringSession.observe` warms history and `score_order` scores subsequent orders. Session state is in memory; persistence, concurrency, recovery, and authorized cross-site data access require deployment work.

## Training and evaluation

Before component splitting, training computes causal **unlabeled full-population BIN counts**, honoring supplied processor counts. Every order uses only population activity through its own timestamp. These background counts preserve population context when calibration contains only 10% of buyer components; no held-out labels or future events enter them. BIN never links buyers. Saved `test_orders.csv` includes this context so re-evaluation reproduces the original metrics.

Full identity components are split approximately **60% train / 15% early-stopping validation / 10% calibration / 15% test**, stratified by each component's maximum order label. At least four components per class are required. Full future links are used only to prevent partition leakage; per-card and identity-graph features are rebuilt causally within each partition. Scaling and vocabularies fit training orders only.

The defaults are a two-layer, 48-unit PyTorch [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), dropout 0.2, Adam, and [binary cross-entropy on logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), plus a 48-tree forest. Early-stopping validation selects neural weights. Separate calibration orders fit branch/final temperatures and select the F1 threshold. Their metrics are selection results because calibration fitting and threshold selection reuse that partition.

Optional `--target-precision 0.9` chooses a calibration operating point targeting that precision. If unattainable, selection falls back to highest calibration precision, then recall, and records that the target was unmet. This does not guarantee future precision. `decision_threshold` is the calibration-selected review cutoff or a prediction-time override. Merchant action cutoffs operate separately and are recorded as `policy_allow_below`, `policy_step_up_below`, and `policy_cap_below` for auditing. A custom action policy can produce a critical recommendation with a false review flag; merchants should align both controls with their decisions and review capacity.

Untouched test components supply per-order precision/recall/F1, ranking and calibration metrics, a prevalence baseline, and legitimate family/shared-identity/multi-event-fan false-positive slices. Results are written to `metrics.json` only after training. Their meaning depends on the supplied labels and sampling; evaluate representative future performances and merchants before interpreting real-world probabilities.

## Verify the software

```bash
.venv/bin/python -m pip install -e './detection[dev,api]'
.venv/bin/python -m pytest -q
```

Automated tests construct temporary synthetic inputs to check feature timing, missing-data handling, linked-component separation, calibration, saved-model reloads, merchant policy, and batch/online agreement. They do not require a supplied dataset or pretrained model, and they do not populate the project's `data/` or `artifacts/` directories. Passing these checks verifies software behavior; it does not measure detection accuracy on real ticket buyers. Optional ONNX checks run when the deployment dependencies are installed.

## Outputs and deployment boundaries

Each response includes probability/risk, branch scores, heuristic confidence, observed reasons, event evidence, and an advisory action. Confidence is not calibrated certainty, and reasons are not model attributions. Ranked review uses final linked component **plus performance**, leaving unknown-target orders in separate review groups. Later evidence can prioritize review without rewriting historical probabilities or propagating another performance's risk.

Models require `format_version=3`, `task=ticket_scalping`, and matching feature names/input width. Old mixed-category models must be retrained. CPU replay includes feature lookup, encoding, both models, calibration, reasons, and policy; compressed arrivals and queue delay are simulations, excluding concurrent/network/persistent-store load. Benchmarking removes supplied processor BIN counters from **both normal and burst passes** because those counters describe the original clock, then rebuilds BIN context from each replayed stream; the report records `bin_counter_context`. Normal scoring/evaluation still honors supplied population counters. IBM/Telum remains unmeasured. Neural-only ONNX export and target compiler/placement checks are described in the [Telum guide](telum_deployment.md).

Protect shared tokens and minimize merchant-visible cross-site data. Review false positives among families, legitimate group buyers, shared households/devices, and new or incomplete histories. Track missingness, linkage changes, calibration, reviewed-label precision/recall, drift across performances, and appeals; production monitoring and appeals are planned controls, not implemented services.
