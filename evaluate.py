"""Evaluate a trained CIFAR-10 model and generate report-ready figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import (
    CIFAR_MEAN,
    CIFAR_STD,
    CLASS_NAMES,
    PROJECT_DIR,
    SimpleCNN,
    create_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple_cnn", "resnet18"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.model == "simple_cnn":
        model = SimpleCNN(num_classes=len(CLASS_NAMES))
    else:
        cifar_stem = bool(
            checkpoint.get("config", {}).get("cifar_stem", False)
        )
        model = create_model(
            "resnet18", pretrained=False, cifar_stem=cifar_stem
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)
        targets_all.append(targets.numpy())
        predictions_all.append(probabilities.argmax(dim=1).cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    return (
        np.concatenate(targets_all),
        np.concatenate(predictions_all),
        np.concatenate(probabilities_all),
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, [x["train_loss"] for x in history], label="Train")
    axes[0].plot(epochs, [x["val_loss"] for x in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        epochs,
        [100 * x["train_accuracy"] for x in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [100 * x["val_accuracy"] for x in history],
        label="Validation",
    )
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, output_path: Path
) -> np.ndarray:
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Row-normalized rate"},
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CIFAR-10 test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return normalized


def plot_per_class_accuracy(
    per_class_accuracy: np.ndarray, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(CLASS_NAMES, 100 * per_class_accuracy, color="#3b82a0")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class test accuracy")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, per_class_accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.1,
            f"{100 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu() * torch.tensor(CIFAR_STD).view(3, 1, 1)
    image += torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_misclassified(
    test_set: datasets.CIFAR10,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> list[dict[str, Any]]:
    wrong_indices = np.flatnonzero(targets != predictions)
    confidences = probabilities[wrong_indices, predictions[wrong_indices]]
    ranked = wrong_indices[np.argsort(-confidences)]
    selected = ranked[:16]
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.2))
    records: list[dict[str, Any]] = []
    for ax, index in zip(axes.flat, selected):
        image, _ = test_set[int(index)]
        ax.imshow(denormalize(image))
        true_name = CLASS_NAMES[int(targets[index])]
        pred_name = CLASS_NAMES[int(predictions[index])]
        confidence = probabilities[index, predictions[index]]
        ax.set_title(
            f"true: {true_name}\npred: {pred_name} ({confidence:.2f})",
            fontsize=8,
        )
        ax.axis("off")
        records.append(
            {
                "index": int(index),
                "true": true_name,
                "predicted": pred_name,
                "confidence": float(confidence),
            }
        )
    fig.suptitle("High-confidence misclassified test images", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return records


class GradCAM:
    """Minimal Grad-CAM implementation for the final ResNet stage."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def generate(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, class_index].backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def plot_gradcam(
    model: nn.Module,
    test_set: datasets.CIFAR10,
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> bool:
    if not hasattr(model, "layer4"):
        return False
    selected: list[int] = []
    correct_indices = np.flatnonzero(targets == predictions)[:4]
    wrong_indices = np.flatnonzero(targets != predictions)[:4]
    selected.extend(correct_indices.tolist())
    selected.extend(wrong_indices.tolist())
    cam_builder = GradCAM(model, model.layer4[-1])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    try:
        for ax, index in zip(axes.flat, selected):
            image, _ = test_set[int(index)]
            input_tensor = image.unsqueeze(0).to(next(model.parameters()).device)
            cam = cam_builder.generate(
                input_tensor, int(predictions[index])
            )
            base = denormalize(image)
            ax.imshow(base)
            ax.imshow(cam, cmap="jet", alpha=0.48)
            outcome = "correct" if targets[index] == predictions[index] else "wrong"
            ax.set_title(
                f"{outcome}: {CLASS_NAMES[int(targets[index])]} -> "
                f"{CLASS_NAMES[int(predictions[index])]}",
                fontsize=8,
            )
            ax.axis("off")
        fig.suptitle("Grad-CAM: model attention regions", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    finally:
        plt.close(fig)
        cam_builder.close()
        model.eval()
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=False,
        transform=eval_transform,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    test_loader = DataLoader(test_set, **loader_options)
    targets, predictions, probabilities = collect_predictions(
        model, test_loader, device
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
    )

    labels = list(range(len(CLASS_NAMES)))
    report_text = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    report_dict = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(targets, predictions, labels=labels)
    per_class_accuracy = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    predicted_confidence = probabilities[
        np.arange(len(predictions)), predictions
    ]
    correct_mask = targets == predictions
    metrics = {
        "accuracy": float((targets == predictions).mean()),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "per_class_accuracy": {
            name: float(value)
            for name, value in zip(CLASS_NAMES, per_class_accuracy)
        },
        "confusion_matrix": cm.tolist(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "test_samples": int(len(targets)),
        "mean_confidence_correct": float(
            predicted_confidence[correct_mask].mean()
        ),
        "mean_confidence_wrong": float(
            predicted_confidence[~correct_mask].mean()
        ),
    }
    save_json(args.output_dir / "metrics.json", metrics)

    plot_training_curves(
        args.checkpoint.parent / "history.json",
        args.output_dir / "training_curves.png",
    )
    plot_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(
        per_class_accuracy, args.output_dir / "per_class_accuracy.png"
    )
    misclassified = plot_misclassified(
        test_set,
        targets,
        predictions,
        probabilities,
        args.output_dir / "misclassified.png",
    )
    save_json(
        args.output_dir / "misclassified_examples.json",
        {"examples": misclassified},
    )
    gradcam_generated = plot_gradcam(
        model,
        test_set,
        targets,
        predictions,
        args.output_dir / "gradcam.png",
    )
    save_json(
        args.output_dir / "evaluation_summary.json",
        {
            "accuracy": metrics["accuracy"],
            "num_parameters": metrics["num_parameters"],
            "gradcam_generated": gradcam_generated,
        },
    )
    print(report_text)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
