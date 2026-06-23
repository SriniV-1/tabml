"""Command-line interface: `tabml train ...` and `tabml predict ...`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from . import core
from .report import format_report


def _train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    result = core.train(df, target=args.target, test_size=args.test_size, cv=args.cv)
    print(format_report(result))
    core.save(result, args.out)
    print(f"\n  ✓ model saved → {args.out}")
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tabml",
        description="Train, evaluate, and serve a model from any CSV.",
    )
    p.add_argument("--version", action="version", version=f"tabml {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train and evaluate a model on a CSV")
    t.add_argument("csv", help="path to the training CSV")
    t.add_argument("--target", "-t", required=True, help="name of the target column")
    t.add_argument("--out", "-o", default="model.joblib", help="where to save the model")
    t.add_argument("--test-size", type=float, default=0.2, help="held-out fraction")
    t.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    t.set_defaults(func=_train)

    pr = sub.add_parser("predict", help="predict with a saved model on a CSV")
    pr.add_argument("csv", help="path to the CSV to predict on")
    pr.add_argument("--model", "-m", required=True, help="path to a saved .joblib model")
    pr.add_argument("--out", "-o", default=None, help="write predictions to this CSV")
    pr.set_defaults(func=_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
