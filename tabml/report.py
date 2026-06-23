"""Pretty terminal reporting for a training run."""

from __future__ import annotations

from .core import TrainResult


def format_report(r: TrainResult) -> str:
    lines = []
    rule = "─" * 56
    lines.append(rule)
    lines.append(f"  tabml · {r.task.upper()}  —  target: {r.target}")
    lines.append(rule)
    lines.append(f"  rows: {r.n_rows:,}   features: {r.n_features}   "
                 f"trained in {r.elapsed_s}s")
    lines.append("")

    lines.append(f"  Model comparison ({r.cv_metric}, {len(r.cv_scores)} models, CV):")
    best = max(r.cv_scores.values()) if r.cv_scores else None
    for name, score in sorted(r.cv_scores.items(), key=lambda kv: kv[1], reverse=True):
        marker = "►" if score == best else " "
        bar = "█" * int(max(0.0, min(1.0, score)) * 24)
        lines.append(f"   {marker} {name:<20} {score:>7.4f}  {bar}")
    lines.append("")

    lines.append(f"  Best model: {r.best_model}")
    lines.append("  Held-out test metrics:")
    for k, v in r.test_metrics.items():
        lines.append(f"    {k:<14} {v}")
    if r.class_labels is not None:
        labels = ", ".join(str(c) for c in r.class_labels[:8])
        more = " …" if len(r.class_labels) > 8 else ""
        lines.append(f"    classes        {labels}{more}")
    lines.append(rule)
    return "\n".join(lines)
