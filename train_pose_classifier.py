import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["sit", "stand"]


class CropDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sit/stand image classifier on person crops."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/pose_classifier/labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/pose_classifier"))
    parser.add_argument("--model-name", choices=["efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_labeled_samples(dataset_root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((image_path, class_index))
    if not samples:
        raise FileNotFoundError(
            f"No labeled images found under {dataset_root}. "
            "Expected folders like labeled/sit and labeled/stand."
        )
    return samples


def split_samples(
    samples: List[Tuple[Path, int]],
    val_split: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    by_class: Dict[int, List[Tuple[Path, int]]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample[1]].append(sample)

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    rng = random.Random(seed)

    for class_index, class_samples in by_class.items():
        rng.shuffle(class_samples)
        split_index = max(1, int(len(class_samples) * (1.0 - val_split)))
        if len(class_samples) == 1:
            train_samples.extend(class_samples)
            continue
        train_samples.extend(class_samples[:split_index])
        val_samples.extend(class_samples[split_index:])

    return train_samples, val_samples


def build_model(model_name: str) -> nn.Module:
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        return model

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    samples = collect_labeled_samples(args.dataset_root)
    train_samples, val_samples = split_samples(samples, args.val_split, args.seed)
    train_transform, val_transform = make_transforms(args.image_size)

    train_dataset = CropDataset(train_samples, train_transform)
    val_dataset = CropDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_classifier.pt"
    metadata_path = args.output_dir / "training_metadata.json"

    best_val_accuracy = -1.0
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            seen += int(labels.size(0))

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
        }
        history.append(epoch_metrics)
        print(epoch_metrics)

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                },
                best_model_path,
            )

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_accuracy": round(float(best_val_accuracy), 4),
        "class_names": CLASS_NAMES,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
