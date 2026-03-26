import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["focused", "distracted"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
GROUP_NAME_PATTERN = re.compile(r"^(?P<video>.+?)_f\d+_id(?P<track>[^_]+)_det\d+$")


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    group_key: str


class CropDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_sample_image(sample.path)
        return self.transform(image), sample.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a distracted/focused image classifier on person crops."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/distracted_classifier/labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/distracted_classifier"))
    parser.add_argument("--model-name", choices=["efficientnet_b0", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.02)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_group_key(image_path: Path) -> str:
    match = GROUP_NAME_PATTERN.match(image_path.stem)
    if match:
        return f"{match.group('video')}|track_{match.group('track')}"
    return f"{image_path.parent.name}|{image_path.stem}"


def load_sample_image(sample_path: Path) -> Image.Image:
    suffix = sample_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return Image.open(sample_path).convert("RGB")
    if suffix in VIDEO_EXTENSIONS:
        import cv2

        cap = cv2.VideoCapture(str(sample_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open sample video: {sample_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            target_frame = max(0, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Unable to decode a frame from sample video: {sample_path}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        finally:
            cap.release()
    raise ValueError(f"Unsupported sample extension: {sample_path.suffix}")


def collect_labeled_samples(dataset_root: Path) -> List[ImageSample]:
    samples: List[ImageSample] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                continue
            samples.append(
                ImageSample(
                    path=image_path,
                    label=class_index,
                    group_key=infer_group_key(image_path),
                )
            )
    if not samples:
        raise FileNotFoundError(
            f"No labeled images found under {dataset_root}. "
            "Expected folders like labeled/focused and labeled/distracted."
        )
    return samples


def split_samples(
    samples: Sequence[ImageSample],
    val_split: float,
    seed: int,
) -> Tuple[List[ImageSample], List[ImageSample]]:
    grouped: Dict[Tuple[int, str], List[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.label, sample.group_key), []).append(sample)

    by_class_groups: Dict[int, List[List[ImageSample]]] = {index: [] for index in range(len(CLASS_NAMES))}
    for (label, _group_key), group_samples in grouped.items():
        by_class_groups[label].append(group_samples)

    rng = random.Random(seed)
    train_samples: List[ImageSample] = []
    val_samples: List[ImageSample] = []

    for class_index, groups in by_class_groups.items():
        rng.shuffle(groups)
        if len(groups) <= 1:
            for group in groups:
                train_samples.extend(group)
            continue

        val_group_count = max(1, int(round(len(groups) * val_split)))
        val_group_count = min(val_group_count, len(groups) - 1)
        val_groups = groups[:val_group_count]
        train_groups = groups[val_group_count:]

        for group in train_groups:
            train_samples.extend(group)
        for group in val_groups:
            val_samples.extend(group)

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


def make_transforms(image_size: int, gaussian_noise_std: float):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda tensor: (tensor + torch.randn_like(tensor) * gaussian_noise_std).clamp(0.0, 1.0)
                if gaussian_noise_std > 0
                else tensor
            ),
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


def compute_class_weights(samples: Sequence[ImageSample], device: torch.device) -> torch.Tensor:
    counts = [0 for _ in CLASS_NAMES]
    for sample in samples:
        counts[sample.label] += 1
    total = sum(counts)
    weights = [total / max(count * len(CLASS_NAMES), 1) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

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
            true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
            false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
            false_negative += int(((predictions == 0) & (labels == 1)).sum().item())

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    samples = collect_labeled_samples(args.dataset_root)
    train_samples, val_samples = split_samples(samples, args.val_split, args.seed)
    train_transform, val_transform = make_transforms(args.image_size, args.gaussian_noise_std)

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
    class_weights = compute_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_distracted_classifier.pt"
    metadata_path = args.output_dir / "training_metadata.json"

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
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
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1": round(val_metrics["f1"], 4),
        }
        history.append(epoch_metrics)
        print(epoch_metrics)

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_f1 improvement for {args.early_stopping_patience} epoch(s)."
            )
            break

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_f1": round(float(best_val_f1), 4),
        "best_epoch": best_epoch,
        "class_names": CLASS_NAMES,
        "gaussian_noise_std": args.gaussian_noise_std,
        "early_stopping_patience": args.early_stopping_patience,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
