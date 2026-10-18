# Train and use the detector

The detector scores ticket purchases for scalping patterns across merchants. It combines an LSTM (recent orders on one card) and a random forest (current transaction features), then calibrates the combined score. It returns observed reasons and simulated policy advice.

Start with the [root walkthrough](../README.md#run-the-complete-demo) for the complete datathon demonstration. This guide covers detector-specific tasks and team-supplied data. Exact features and model behavior are in the [implementation reference](docs/implementation.md).

## Prerequisites and command conventions

Complete the matching [platform setup](../README.md#start-here) and [native checks](../docs/environment.md#check-native-packages-before-the-full-pipeline). Run from the repository root. Do not run a generic venv-creation command over an existing environment.

Commands below use Linux/macOS paths. In PowerShell replace `.venv/bin/python` with `.venv\Scripts\python.exe`; commands are single lines. On IBM, use the interpreter from the validated [LinuxONE environment](../docs/platforms/linuxone.md).

`python -m scalper_detector` uses the selected interpreter directly. The installed `scalper-detector` launcher is equivalent when it belongs to the same environment. Always pass `--model` explicitly: some standalone commands default to a different demo-model folder.

No pretrained model or training dataset is committed. Automated tests build temporary models; they do not create the model needed for your presentation.

## Use generated authorizations

After generating the dataset with the [generator](../generation_and_testing/README.md):

```bash
.venv/bin/python -m scalper_detector train-simulation --simulation generation_and_testing/data/simulation --output artifacts/ticket_model --epochs 20
.venv/bin/python -m scalper_detector predict-authorizations --requests generation_and_testing/data/simulation/test/requests.jsonl --model artifacts/ticket_model --output artifacts/authorization_scores.csv
```

Training joins separate truth by `txn_id` offline. It preserves source training/test populations and splits source validation into model-selection and calibration groups with linked cards kept together. Scoring uses requests alone. Reports include per-tactic recall and false positives on legitimate bulk buyers.

Amounts retain USD units. Missing release times and event allocations stay unknown. The generator's per-order quantity cap is not a known per-buyer event allocation.

## Use the team's own CSV data

These example files are **not supplied**. Put approved input files under `data/` and change paths to match the team's files.

| Required column | Meaning |
| --- | --- |
| `order_id` | Unique authorization/order identifier |
| `timestamp` | ISO time; use UTC or an explicit offset. CSV-only naive times are interpreted as UTC. |
| `card_id`, `card_bin`, `merchant_id` | Consistent tokenized identifiers, preserved as strings |
| `category` | `tickets`; other categories and mixed-category inputs are rejected |
| `amount` | Finite positive payment in consistent units/currency |
| `is_scalper` | Per-order 0/1 label for training/evaluation only; prediction ignores it |

Optional useful columns are `quantity`, `event_id`, `shipping_hash`, `billing_hash`, `email_hash`, `device_id`, `card_type`, `card_age_days`, `seconds_since_release`, and `ticket_limit`. Leave unknown values blank. See the [full input contract](docs/implementation.md#input-schema) for optional processor BIN counters and availability flags.

Use a common event ID for the same specific performance across sites. Shared addresses/devices can belong to legitimate buyers. Token namespaces must be consistent across merchants; raw card numbers are unnecessary.

Before training:

1. Include reviewed positive and negative labels. Payment fraud/chargeback labels are not interchangeable with scalping labels.
2. Supply at least four independent linked-card groups per class. Repeated rows from one buyer are not independent groups.
3. Include only information observable at each order's time. Later investigations belong in labels.
4. Include prior orders needed for transaction history. A new-order-only file cannot reconstruct omitted activity.

### Train, then evaluate

```bash
.venv/bin/python -m scalper_detector train --data data/ticket_orders.csv --data-source user_supplied --output artifacts/ticket_model --epochs 20
.venv/bin/python -m scalper_detector evaluate --data artifacts/ticket_model/test_orders.csv --model artifacts/ticket_model --output artifacts/rechecked_evaluation.json
```

Direct CSV training splits linked groups into approximately 60% train, 15% model-selection validation, 10% calibration, and 15% test. This differs from the generator's source split workflow above. Preprocessing fits training data only.

Use another output directory when preserving an earlier model. Keep `model.pt`, `tree.json`, and `metadata.json` from the same training run together, along with the reports/configuration. Evaluation needs labels; prediction does not. For a later generalization check, evaluate independently reviewed future orders.

`--target-precision 0.9` selects a calibration operating point targeting 90% precision. An unmet target is recorded; it is not a promise about future traffic.

## Score and review purchases

Batch scoring rebuilds fresh chronological history from the supplied file:

```bash
.venv/bin/python -m scalper_detector predict --data data/orders_to_score.csv --model artifacts/ticket_model --output artifacts/order_scores.csv
.venv/bin/python -m scalper_detector report --data data/orders_to_score.csv --model artifacts/ticket_model --output artifacts/review_report.csv
```

The review report groups the same performance within a linked buyer group. Later evidence can prioritize an earlier order for review without rewriting its original score. Unknown event IDs and unrelated performances do not share review groups.

To score one JSON order after earlier CSV history:

```bash
.venv/bin/python -m scalper_detector score-order --order data/order.json --history data/history.csv --model artifacts/ticket_model --output artifacts/order_result.json
```

The JSON uses the CSV field names above, not the HTTP authorization names. Exclude the new order from history; IDs must be unique and history must precede it. For the original authorization request fields and continuing history, use the [API](../datathon_authorization/README.md).

Prediction/report commands accept `--threshold` for the review cutoff and `--policy detection/examples/merchant_policy.json` for merchant action advice. These control different decisions.

## Explain the results

| Output | How to present it |
| --- | --- |
| `scalper_probability` / `risk_score` | Combined model estimate, on 0–1 / 0–100 scales |
| `flagged_for_review`, `decision_threshold` | Whether the estimate meets the review cutoff |
| `confidence_tier` | Heuristic evidence quality from history, missingness, and model agreement |
| `reason_codes`, `reason_details` | Observed patterns; not proof of intent or causal model attribution |
| `same_event_sites_5m`, `observed_event_tickets_24h` | Observed cross-site concentration and known-quantity accumulation |
| `recommended_action`, `policy_*` | Simulated merchant advice and the cutoffs used |
| `limited_history`, `events_observed` | Whether sufficient recent card history is available |

Default merchant advice is allow below 0.35, step-up below 0.65, quantity-cap/throttle below 0.85, and cancel/refund at or above 0.85. These cutoffs are separate from the calibrated review threshold. Changing one does not change the other or execute an action.

Precision asks how many flagged orders are labeled scalping; recall asks how many labeled scalper orders were caught. Also report legitimate-buyer false positives, per-tactic recall, and ranking/calibration metrics. Synthetic results measure the generated scenario, not real-world accuracy.

## Benchmark and export

After training, with sufficient input rows:

```bash
.venv/bin/python -m scalper_detector benchmark --data data/orders_to_score.csv --model artifacts/ticket_model --orders 100 --warmup 20 --burst-rate 1000 --output artifacts/benchmark.json
```

Supply at least 120 rows for 100 measured orders plus 20 warmups. This measures serial CPU service time and simulates arrival pressure/queue delay; it excludes network, persistent-state I/O, and actual concurrent load. A requested burst rate is not achieved throughput. BIN counters are rebuilt for each replay clock.

For ONNX, install the optional deployment extras through your platform guide, then:

```bash
.venv/bin/python -m scalper_detector export-onnx --model artifacts/ticket_model --output artifacts/ticket_model/scalper_lstm.onnx
```

Only the neural branch is exported. Inspect the manifest for CPU parity and actual tensor shapes. See [IBM deployment](docs/telum_deployment.md); export is not hardware validation.

## Checks and common problems

Run the [root checks](../README.md#check-a-team-change) after source changes. Focused detector checks include:

```bash
.venv/bin/python -m pytest -c pytest.ini detection/tests/test_data.py detection/tests/test_authorizations.py -q
.venv/bin/python -m pytest -c pytest.ini detection/tests/test_pipeline.py detection/tests/test_simulation.py -q
```

| Problem | Next step |
| --- | --- |
| Native crash/import error | Use [environment diagnostics](../docs/environment.md) before changing the model. |
| Model/feature format mismatch | Use all files from one run or retrain with the current code. |
| Too few groups for training | Add independent labeled buyer groups in both classes. |
| Missing event totals or linkage | Check consistent tokens/IDs, supplied quantities, and prior history. Unknown values are not invented. |
| Duplicate/late orders | Keep unique IDs and globally chronological history. |
| Missing generated data after switching machines | Transfer ignored data/model folders or regenerate; Git carries source only. |

[Implementation reference](docs/implementation.md) · [Team workflow](docs/git_workflow.md) · [Live presentation](../live_demo/README.md)
