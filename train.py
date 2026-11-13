"""Train image-classification models on CIFAR-10.

The script supports a compact custom CNN and a pretrained ResNet-18. It uses a
deterministic 90/10 split of the original training set, keeps the official
10,000-image test set untouched, and writes checkpoints and history to a run
directory for later evaluation and reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms import RandAugment


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions followed by batch normalization and pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """A compact three-stage CNN designed specifically for 32x32 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("simple_cnn", "resnet18"), default="resnet18"
    )
    parser.add_argument(
        "--augmentation",
        choices=("none", "basic", "strong"),
        default="strong",
        help="Training-time augmentation strategy.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 avoids costly process startup on Windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to results/<model>_<augmentation>.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize ResNet-18 from random weights.",
    )
    parser.add_argument(
        "--cifar-stem",
        action="store_true",
        help="Use a 3x3 stride-1 ResNet stem without max pooling for CIFAR.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Development-only limit for a smoke test.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Development-only limit for a smoke test.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    augmentation: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    if augmentation == "none":
        train_ops: list[Any] = []
    elif augmentation == "basic":
        train_ops = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        train_ops = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=7),
        ]

    train_transform = transforms.Compose(
        train_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, eval_transform


def build_loaders(
    data_dir: Path,
    augmentation: str,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    train_transform, eval_transform = build_transforms(augmentation)
    train_view = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=train_transform,
    )
    val_view = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_view), generator=generator).tolist()
    val_size = 5_000
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    train_set = Subset(train_view, train_indices)
    val_set = Subset(val_view, val_indices)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)
    return train_loader, val_loader, test_loader, len(val_set)


def create_model(
    model_name: str, pretrained: bool, cifar_stem: bool = False
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=len(CLASS_NAMES))

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if cifar_stem:
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
    return loss_sum / max(seen, 1), correct / max(seen, 1)


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or (
        PROJECT_DIR / "results" / f"{args.model}_{args.augmentation}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = args.model == "resnet18" and not args.no_pretrained
    model = create_model(
        args.model, pretrained=pretrained, cifar_stem=args.cifar_stem
    ).to(device)
    train_loader, val_loader, _, val_size = build_loaders(
        data_dir=args.data_dir,
        augmentation=args.augmentation,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )

    default_lr = 8e-4 if args.model == "simple_cnn" else 3e-4
    default_weight_decay = 1e-4 if args.model == "simple_cnn" else 5e-4
    default_smoothing = 0.05 if args.model == "simple_cnn" else 0.1
    lr = args.lr if args.lr is not None else default_lr
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else default_weight_decay
    )
    smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else default_smoothing
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )

    config = {
        "model": args.model,
        "pretrained": pretrained,
        "cifar_stem": args.cifar_stem,
        "augmentation": args.augmentation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "label_smoothing": smoothing,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "seed": args.seed,
        "training_samples": 45_000,
        "validation_samples": val_size,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
    }
    write_json(output_dir / "config.json", config)
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)

    history: list[dict[str, Any]] = []
    best_accuracy = 0.0
    epochs_without_improvement = 0
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.max_train_batches,
        )
        val_loss, val_accuracy = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            args.max_val_batches,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "epoch_seconds": elapsed,
        }
        history.append(record)
        write_json(output_dir / "history.json", {"epochs": history})

        improved = val_accuracy > best_accuracy
        if improved:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "classes": CLASS_NAMES,
                    "config": config,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"lr={current_lr:.6f} time={elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best validation accuracy={best_accuracy:.4f}",
                flush=True,
            )
            break

    total_seconds = time.perf_counter() - run_start
    summary = {
        "best_val_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "training_seconds": total_seconds,
        "config": config,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
