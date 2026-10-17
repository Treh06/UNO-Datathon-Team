# Synthetic ticket authorization generator

Generate seeded request streams for training and testing the ticket scalper detector. Requires Python 3.10 or later and uses only the Python standard library.

Run from the repository root. After the [shared environment setup](../README.md#1-install):

```bash
.venv/bin/python -m generation_and_testing.simulator.generate \
  --config generation_and_testing/configs/default.json \
  --output generation_and_testing/data/simulation
```

On Windows PowerShell, use `.venv\Scripts\python.exe` and put the command on one line. Choose a new or empty output directory for each run; existing data is never overwritten.

For standalone generation without the shared environment, use `python3` instead of `.venv/bin/python` (`py -3` on Windows). No detector or API installation is required.

## Generated dataset

The default configuration produces **11,640 orders** across three days: 8,640 baseline orders plus 3,000 release-burst orders. Each split contains:

- `requests.jsonl`: timestamp-sorted authorization inputs, with no labels or actor IDs.
- `ground_truth.jsonl`: `txn_id`, `is_scalper`, `actor_id`, and `tactic`.

`manifest.json` records the full configuration, split time boundaries, row counts, and tactic counts. Transaction IDs are unique across the entire run. Equal timestamps are allowed. Identical configuration and Python runtime produce identical files.

## Configure the scenario

Edit [configs/default.json](configs/default.json):

| Setting | Meaning |
| --- | --- |
| `seed` | Reproducible randomness and synthetic identifiers |
| `start`, `end` | Timezone-aware simulation range, end exclusive; output is UTC |
| `baseline_orders_per_hour` | Average baseline volume, before bursts; rounded separately in each split |
| `merchants`, `events`, `customers_per_split` | Entity pool sizes |
| `scalper_ratio` | Fraction of orders labeled scalper, rounded per baseline/drop population |
| `legit_bulk_ratio` | Fraction of legitimate orders assigned to bulk-buying decoys |
| `quantity_cap` | Typical scalper bulk-order quantity; legitimate decoys can buy up to six |
| `optional_field_missing_rate` | Independent omission probability for event, device and IP fields |
| `split_fractions` | Chronological duration fractions for train, validation and test |
| `held_out_tactic` | `null` or one scalper tactic excluded from train and validation |
| `tactic_weights` | Relative session-sampling weights for the four scalper tactics |
| `drops` | Release timestamps, additional order counts, arrival windows and event indices |

Baseline volume has a daily rhythm in **UTC**. Malicious sessions compress orders into minutes; releases mix faster scalper arrivals and slower human arrivals with overlapping timing distributions. A release belongs to the split containing its timestamp. Sessions and drops near a split boundary are clipped to that split, which can accumulate orders at the final millisecond; place releases away from boundaries.

Implemented tactics are `shared_address_ring`, `max_qty_bulk`, `card_testing`, and `cross_merchant_sweep`. Each has its own function in [simulator/generate.py](simulator/generate.py). `normal` and `legit_bulk` are negative labels. Households share addresses and devices; corporate decoys can use several cards. Card-testing sessions place tiny probes before larger purchases. Quantities, card types, billing mismatches, device rotation, and missing fields provide some overlap between classes. Each malicious session represents a separate actor; legitimate customers may return across sessions within their split.

Weights sample sessions, so small datasets need not contain every tactic or match exact tactic proportions. To test an unseen tactic, set `held_out_tactic` to (for example) `"card_testing"`, retain its positive weight, and generate enough test traffic. Check the manifest to confirm it occurs.

## Use with a model

Join requests to ground truth on `txn_id` **only in the offline training/evaluation pipeline**. Keep `txn_id`, `actor_id`, `tactic`, and `is_scalper` out of model features. Fit preprocessing and model parameters on training data only. Use validation for tuning and test for final evaluation. Never expose ground-truth files to an authorization endpoint.

Actors, card tokens, addresses, emails, and devices are isolated across splits as well as separated in time. This measures generalization to new entities; it does not model a returning-customer deployment. Merchant and event identifiers may recur. Build graph and velocity features causally from earlier requests, and reset state between these independent split populations. Do not compute features using future requests or all labels. Report per-tactic recall and false positives on `legit_bulk` alongside overall precision/recall.

All identifiers are synthetic hashes. Card tokens are visibly non-PAN strings; six-digit BIN values are fabricated issuer metadata without an issuer lookup claim. IP prefixes use documentation ranges. Amounts are calculated in integer cents, then emitted as JSON numbers in USD; JSON does not preserve trailing decimal zeroes. This is a simplified authorization format, not an ISO 8583 wire encoder.

Synthetic test performance does not establish performance on real payment traffic. This generator deliberately creates detectable patterns, and card-testing labels indicate membership in a simulated scalper campaign rather than every possible real-world reason for a small purchase. Use representative authorized real data before making deployment claims.

This implementation covers offline data generation. Authorization responses, model scores, replay networking, state storage, and response evaluation belong to the service pipeline and are not fabricated as training inputs.

## Tests

```bash
.venv/bin/python -m unittest discover -s generation_and_testing/tests -v
```

The tests use temporary output directories and check reproducibility, schema validation, chronology, identity isolation, release bursts, and tactic behavior. Run the complete repository suite with `.venv/bin/python -m pytest -q` after installing the shared development dependencies.
