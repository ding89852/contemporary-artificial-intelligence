"""Compare completed CIFAR-10 runs and create report-ready summary figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Label=result-directory pair.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path]] = []
    for item in args.run:
        label, path = item.split("=", 1)
        runs.append((label, Path(path)))

    summaries: dict[str, dict[str, object]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, result_dir in runs:
        history_path = result_dir / "history.json"
        if not history_path.exists():
            history_path = result_dir.parent / "history.json"
        history = json.loads(
            history_path.read_text(encoding="utf-8")
        )["epochs"]
        metrics = json.loads(
            (result_dir / "metrics.json").read_text(encoding="utf-8")
        )
        epochs = [item["epoch"] for item in history]
        axes[0].plot(
            epochs,
            [100 * item["val_accuracy"] for item in history],
            marker="o",
            markersize=3,
            label=label,
        )
        class_names = list(metrics["per_class_accuracy"].keys())
        class_values = [
            100 * metrics["per_class_accuracy"][name] for name in class_names
        ]
        axes[1].plot(class_names, class_values, marker="o", label=label)
        summaries[label] = {
            "test_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "num_parameters": metrics["num_parameters"],
            "completed_epochs": len(history),
            "best_val_accuracy": max(
                item["val_accuracy"] for item in history
            ),
            "training_seconds": sum(
                item["epoch_seconds"] for item in history
            ),
        }
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation accuracy (%)",
        title="Validation accuracy by run",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set(
        ylabel="Test accuracy (%)",
        title="Per-class test accuracy",
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "model_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "| Run | Test Acc. | Macro F1 | Params | Best Val Acc. | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in summaries.items():
        lines.append(
            f"| {label} | {100 * values['test_accuracy']:.2f}% | "
            f"{values['macro_f1']:.4f} | "
            f"{values['num_parameters']:,} | "
            f"{100 * values['best_val_accuracy']:.2f}% | "
            f"{values['training_seconds']:.1f} |"
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
