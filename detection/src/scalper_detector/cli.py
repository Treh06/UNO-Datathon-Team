"""Ticket-only training, cross-site order scoring, review, and deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .benchmark import benchmark_orders
from .data import LABEL, TEXT_COLUMNS, read_events, validate_events
from .decision import MerchantPolicy
from .pipeline import ScalperDetector, TrainConfig, train_model
from .synthetic import generate_synthetic_data, generate_ticket_scenarios


def _save_csv(frame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    for column in ("reason_codes", "reason_details", "review_cluster_reason_codes"):
        if column in serializable:
            serializable[column] = serializable[column].map(json.dumps)
    serializable.to_csv(destination, index=False)


def _save_json(value, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _raw_events(path):
    return pd.read_csv(path, dtype={column: "string" for column in TEXT_COLUMNS})


def _policies(path):
    if path is None:
        return None, {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - {"default", "merchants"}:
        raise ValueError("Policy JSON must have default and/or merchants objects.")
    default = MerchantPolicy.from_dict(values.get("default", {}))
    merchants = values.get("merchants", {})
    if not isinstance(merchants, dict):
        raise ValueError("merchants must map merchant IDs to policy objects.")
    return default, {str(key): MerchantPolicy.from_dict(value) for key, value in merchants.items()}


def _training_arguments(parser):
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--neural-weight", type=float, default=0.5)
    parser.add_argument("--target-precision", type=float, help="Desired calibration precision; unmet targets are reported explicitly")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)


def _config(args):
    return TrainConfig(**{key: getattr(args, key) for key in (
        "epochs", "batch_size", "sequence_length", "hidden_size", "patience", "learning_rate", "neural_weight", "target_precision", "seed", "threads"
    )})


def _summary(report, output):
    metrics = report["test"]
    print(f"Saved ticket-scalper ensemble to {output}")
    print(f"Held-out orders: {metrics['n_samples']} | precision {metrics['precision']:.3f} | "
          f"recall {metrics['recall']:.3f} | F1 {metrics['f1']:.3f}")
    print(report["interpretation"])
    selection = report["threshold_selection"]
    if selection["target_precision"] is not None:
        print(f"Calibration precision target: {selection['target_precision']:.3f}; met: {selection['target_met']}. "
              "Future precision is not guaranteed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ticket-scalper detection using payment activity across ticketing sites.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate ticket payments, train, score fan/scalper scenarios, and benchmark CPU")
    demo.add_argument("--cards", "--accounts", dest="accounts", type=int, default=1000)
    demo.add_argument("--output", default="artifacts/ticket_demo")
    demo.add_argument("--data-dir", default="data/generated/tickets")
    _training_arguments(demo)
    generate = commands.add_parser("generate", help="Generate synthetic ticket purchases across ticketing sites")
    generate.add_argument("--cards", "--accounts", dest="accounts", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--scalper-fraction", type=float, default=0.2)
    generate.add_argument("--output", default="data/generated/tickets/training_orders.csv")
    train = commands.add_parser("train", help="Train on per-order labels with linked-entity-isolated splits")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="artifacts/ticket_model")
    train.add_argument("--data-source", choices=["synthetic", "proxy", "user_supplied"], default="user_supplied")
    _training_arguments(train)
    for name, help_text in (("predict", "Score every order using its causal payment history"),
                            ("report", "Rank flagged orders using retrospective cluster evidence")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--data", required=True)
        sub.add_argument("--model", default="artifacts/ticket_demo")
        sub.add_argument("--output", default=f"artifacts/{name}.csv")
        sub.add_argument("--threshold", type=float)
        sub.add_argument("--policy", help="JSON default and per-merchant policy thresholds")
    evaluate = commands.add_parser("evaluate", help="Evaluate per-order predictions on labeled payments")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", default="artifacts/ticket_demo")
    evaluate.add_argument("--output")
    inline = commands.add_parser("score-order", help="Score one JSON order in a simulated authorization path")
    inline.add_argument("--order", required=True)
    inline.add_argument("--history", help="Earlier orders used to warm velocity and linked-card context")
    inline.add_argument("--model", default="artifacts/ticket_demo")
    inline.add_argument("--policy")
    inline.add_argument("--output")
    benchmark = commands.add_parser("benchmark", help="Measure CPU replay and simulated drop-day arrival pressure")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--model", default="artifacts/ticket_demo")
    benchmark.add_argument("--orders", type=int, default=300)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--burst-multiplier", type=float, default=10)
    benchmark.add_argument("--burst-rate", type=float, default=1000, help="Minimum simulated arrival orders per second")
    benchmark.add_argument("--output", default="artifacts/benchmark.json")
    export = commands.add_parser("export-onnx", help="Export/check neural branch for subsequent IBM zDLC validation")
    export.add_argument("--model", default="artifacts/ticket_demo")
    export.add_argument("--output", default="artifacts/ticket_demo/scalper_lstm.onnx")
    scenarios = commands.add_parser("scenarios", help="Compare a fan, family booking, linked-card ticket burst, and different-event fan")
    scenarios.add_argument("--model", default="artifacts/ticket_demo")
    scenarios.add_argument("--output", default="artifacts/ticket_demo/scenario_scores.csv")
    simulation = commands.add_parser("train-simulation", help="Train on the authorization simulator's original train/validation/test splits")
    simulation.add_argument("--simulation", required=True, help="Directory containing manifest.json and the three simulator split folders")
    simulation.add_argument("--output", default="artifacts/ticket_model")
    _training_arguments(simulation)
    convert = commands.add_parser("convert-authorizations", help="Convert simulator authorization JSONL into canonical ticket-order CSV")
    convert.add_argument("--requests", required=True)
    convert.add_argument("--ground-truth", help="Offline-only exact txn_id label join; never needed for prediction")
    convert.add_argument("--output", required=True)
    authorizations = commands.add_parser("predict-authorizations", help="Score unlabeled simulator authorization JSONL directly")
    authorizations.add_argument("--requests", required=True)
    authorizations.add_argument("--model", default="artifacts/ticket_model")
    authorizations.add_argument("--output", default="artifacts/authorization_scores.csv")
    authorizations.add_argument("--threshold", type=float)
    authorizations.add_argument("--policy")
    args = parser.parse_args(argv)
    try:
        if args.command == "train-simulation":
            from .simulation import train_simulation
            report = train_simulation(args.simulation, args.output, _config(args))
            _summary(report, args.output)
            print(f"Original simulator boundaries preserved. Offline tactic analysis: {Path(args.output) / 'simulation_evaluation.json'}")
        elif args.command == "convert-authorizations":
            from .authorizations import read_authorizations
            orders = read_authorizations(args.requests, ground_truth_path=args.ground_truth)
            _save_csv(orders, args.output)
            print(f"Converted {len(orders)} ticket authorizations to {args.output}")
            if args.ground_truth:
                print("Labels were joined for offline training/evaluation only; actor IDs and tactics are excluded from order features.")
        elif args.command == "predict-authorizations":
            from .authorizations import read_authorizations
            policy, merchants = _policies(args.policy)
            orders = read_authorizations(args.requests)
            detector = ScalperDetector.load(args.model)
            scores = detector.predict(orders, threshold=args.threshold, policy=policy, merchant_policies=merchants)
            _save_csv(scores, args.output)
            print(f"Scored {len(scores)} ticket authorizations using requests only. Saved {args.output}")
            if detector.metadata["data_source"] == "synthetic":
                print("Synthetic training: scores require validation on representative real ticket traffic.")
        elif args.command == "generate":
            orders = generate_synthetic_data(args.accounts, args.seed, args.scalper_fraction)
            _save_csv(orders, args.output)
            print(f"Saved {len(orders)} synthetic ticket orders for {args.accounts} cards to {args.output}")
        elif args.command in {"demo", "train"}:
            config = _config(args)
            config.validate()
            orders = (generate_synthetic_data(args.accounts, args.seed) if args.command == "demo"
                      else read_events(args.data, require_labels=True))
            report = train_model(orders, args.output, config,
                                 data_source="synthetic" if args.command == "demo" else args.data_source)
            _summary(report, args.output)
            if args.command == "demo":
                _save_csv(orders, Path(args.data_dir) / "training_orders.csv")
                fresh = generate_synthetic_data(40, args.seed + 1).drop(columns=LABEL)
                for column in ("order_id", "card_id", "shipping_hash", "billing_hash", "email_hash", "device_id"):
                    fresh[column] = fresh[column].map(lambda value: "fresh_" + value if pd.notna(value) and value else value)
                _save_csv(fresh, Path(args.data_dir) / "new_orders.csv")
                detector = ScalperDetector.load(args.output)
                _save_csv(detector.predict(fresh), Path(args.output) / "predictions.csv")
                _save_csv(detector.ranked_report(fresh), Path(args.output) / "ranked_report.csv")
                cases = generate_ticket_scenarios()
                scores = detector.predict(cases)
                scores["scenario_name"] = scores.order_id.map(cases.set_index("order_id").scenario_name)
                _save_csv(scores, Path(args.output) / "scenario_scores.csv")
                _save_json(benchmark_orders(detector, fresh, max_orders=100), Path(args.output) / "benchmark.json")
                print(f"Saved per-order scores, ranked review report, and measured CPU benchmark in {args.output}")
        elif args.command in {"predict", "report"}:
            detector = ScalperDetector.load(args.model)
            policy, merchants = _policies(args.policy)
            method = detector.predict if args.command == "predict" else detector.ranked_report
            results = method(_raw_events(args.data), threshold=args.threshold, policy=policy, merchant_policies=merchants)
            _save_csv(results, args.output)
            print(f"Saved {len(results)} order results to {args.output}. Merchant actions are recommendations only.")
            if detector.metadata["data_source"] == "synthetic":
                print("Synthetic training: likelihoods and confidence tiers are demonstration estimates.")
        elif args.command == "scenarios":
            cases = generate_ticket_scenarios()
            scores = ScalperDetector.load(args.model).predict(cases)
            scores["scenario_name"] = scores.order_id.map(cases.set_index("order_id").scenario_name)
            _save_csv(scores, args.output)
            latest = scores.groupby("scenario_name", sort=False).tail(1)
            print(latest[["scenario_name", "risk_score", "confidence_tier", "same_event_sites_5m",
                          "observed_event_tickets_24h", "recommended_action"]].to_string(index=False))
            print(f"Saved illustrative ticket scenarios to {args.output}. Synthetic examples do not establish real-world accuracy.")
        elif args.command == "score-order":
            detector = ScalperDetector.load(args.model)
            policy, merchants = _policies(args.policy)
            session = detector.new_session(policy=policy, merchant_policies=merchants)
            if args.history:
                history = validate_events(_raw_events(args.history).drop(columns=[LABEL], errors="ignore"))
                for order in history.to_dict("records"):
                    session.observe(order)
            result = session.score_order(json.loads(Path(args.order).read_text(encoding="utf-8")))
            if args.output:
                _save_json(result, args.output)
            print(json.dumps(result, indent=2, allow_nan=False, default=str))
        elif args.command == "benchmark":
            result = benchmark_orders(ScalperDetector.load(args.model), _raw_events(args.data),
                                      max_orders=args.orders, burst_multiplier=args.burst_multiplier, warmup=args.warmup,
                                      burst_orders_per_second=args.burst_rate)
            _save_json(result, args.output)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "export-onnx":
            from .deployment import export_onnx
            result = export_onnx(args.model, args.output)
            print(f"Exported neural ONNX: {args.output}; CPU parity verified: {result['cpu_numerical_parity']['verified']}")
            print("IBM zDLC compilation, accelerator operator support, and Telum measurements remain to be verified on target hardware.")
        else:
            result = ScalperDetector.load(args.model).evaluate(read_events(args.data, require_labels=True))
            if args.output:
                _save_json(result, args.output)
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
