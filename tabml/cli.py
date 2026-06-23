"""Command-line interface: train · predict · audit · serve."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from . import core
from .report import format_audit, format_report, model_card


def _train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    result = core.train(df, target=args.target, test_size=args.test_size, cv=args.cv)
    print(format_report(result))
    core.save(result, args.out)
    print(f"\n  ✓ model saved → {args.out}")
    if args.card:
        with open(args.card, "w") as fh:
            fh.write(model_card(result))
        print(f"  ✓ model card → {args.card}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    out = core.predict(args.model, df)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"✓ predictions written → {args.out} ({len(out)} rows)")
    else:
        cols = [c for c in ("prediction", "confidence") if c in out.columns]
        print(out[cols].to_string(index=False))
    return 0


def _audit(args: argparse.Namespace) -> int:
    from .audit import audit

    df = pd.read_csv(args.csv)
    result = audit(df, target=args.target)
    print(format_audit(result))
    # non-zero exit if leakage was found, so it's CI-friendly
    return 2 if result.leaks else 0


def _serve(args: argparse.Namespace) -> int:
    from .serve import serve

    print(f"serving {args.model} on http://{args.host}:{args.port}  (POST /predict)")
    serve(args.model, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tabml",
        description="Audit, train, explain, and serve a model from any CSV.",
    )
    p.add_argument("--version", action="version", version=f"tabml {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train, compare, and evaluate models on a CSV")
    t.add_argument("csv", help="path to the training CSV")
    t.add_argument("--target", "-t", required=True, help="name of the target column")
    t.add_argument("--out", "-o", default="model.joblib", help="where to save the model")
    t.add_argument("--card", default=None, help="also write a markdown model card here")
    t.add_argument("--test-size", type=float, default=0.2, help="held-out fraction")
    t.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    t.set_defaults(func=_train)

    pr = sub.add_parser("predict", help="predict with a saved model on a CSV")
    pr.add_argument("csv", help="path to the CSV to predict on")
    pr.add_argument("--model", "-m", required=True, help="path to a saved .joblib model")
    pr.add_argument("--out", "-o", default=None, help="write predictions to this CSV")
    pr.set_defaults(func=_predict)

    au = sub.add_parser("audit", help="pre-flight data audit (leakage + quality checks)")
    au.add_argument("csv", help="path to the CSV to audit")
    au.add_argument("--target", "-t", required=True, help="name of the target column")
    au.set_defaults(func=_audit)

    sv = sub.add_parser("serve", help="serve a saved model as a REST API (needs tabml[serve])")
    sv.add_argument("model", help="path to a saved .joblib model")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
