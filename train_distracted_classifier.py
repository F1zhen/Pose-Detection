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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLASS_NAMES = ["focused", "distracted"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
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
<<<<<<< HEAD
        frames = load_clip_frames(sample.path, self.frames_per_clip)
        clip_tensor = self.transform(frames)
        return clip_tensor, sample.label


class ClipTransform:
    def __init__(
        self,
        image_size: int,
        train: bool,
        gaussian_noise_std: float = 0.03,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.gaussian_noise_std = gaussian_noise_std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.12,
            hue=0.03,
        )

    def __call__(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        processed = list(frames)

        if self.train:
            processed = self._augment_clip(processed)
        else:
            processed = [
                TF.resize(frame, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
                for frame in processed
            ]

        tensor_frames = []
        for frame in processed:
            tensor = TF.to_tensor(frame)
            if self.train and self.gaussian_noise_std > 0:
                tensor = tensor + torch.randn_like(tensor) * self.gaussian_noise_std
                tensor = tensor.clamp(0.0, 1.0)
            tensor_frames.append(self.normalize(tensor))
        return torch.stack(tensor_frames, dim=0)

    def _augment_clip(self, frames: Sequence[Image.Image]) -> List[Image.Image]:
        do_flip = random.random() < 0.5
        rotation = random.uniform(-8.0, 8.0)
        crop_scale = random.uniform(0.88, 1.0)
        crop_size = max(1, int(self.image_size * crop_scale))
        max_offset = self.image_size - crop_size
        top = random.randint(0, max_offset) if max_offset > 0 else 0
        left = random.randint(0, max_offset) if max_offset > 0 else 0
        brightness_factor = random.uniform(0.82, 1.18)
        contrast_factor = random.uniform(0.82, 1.18)
        saturation_factor = random.uniform(0.88, 1.12)
        hue_factor = random.uniform(-0.03, 0.03)

        augmented: List[Image.Image] = []
        for frame in frames:
            img = TF.resize(frame, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            if do_flip:
                img = TF.hflip(img)
            img = TF.rotate(img, rotation, interpolation=InterpolationMode.BILINEAR, fill=0)
            if crop_scale < 0.999:
                img = TF.resized_crop(
                    img,
                    top=top,
                    left=left,
                    height=crop_size,
                    width=crop_size,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)
            img = TF.adjust_saturation(img, saturation_factor)
            img = TF.adjust_hue(img, hue_factor)
            augmented.append(img)
        return augmented


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        transformer_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frame_encoder, encoder_dim = build_frame_encoder(backbone_name)
        self.projection = nn.Linear(encoder_dim, transformer_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.position = PositionalEncoding(transformer_dim, dropout=dropout, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Linear(transformer_dim, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = clips.shape
        frames = clips.view(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(frames)
        frame_features = frame_features.view(batch_size, time_steps, -1)
        tokens = self.projection(frame_features)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.position(tokens)
        encoded = self.transformer(tokens)
        cls_embedding = self.norm(encoded[:, 0])
        return self.head(cls_embedding)
=======
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label
>>>>>>> origin/feature/map


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
<<<<<<< HEAD
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=3,
        help="Freeze the frame encoder for the first N epochs to stabilise training on a small dataset.",
    )
    parser.add_argument("--gaussian-noise-std", type=float, default=0.03)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
=======
>>>>>>> origin/feature/map
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


def collect_labeled_samples(dataset_root: Path) -> List[ImageSample]:
    samples: List[ImageSample] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
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


<<<<<<< HEAD
def make_transforms(image_size: int, gaussian_noise_std: float):
    train_transform = ClipTransform(
        image_size=image_size,
        train=True,
        gaussian_noise_std=gaussian_noise_std,
=======
def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=6),
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
>>>>>>> origin/feature/map
    )
    val_transform = ClipTransform(image_size=image_size, train=False, gaussian_noise_std=0.0)
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
<<<<<<< HEAD
        "transformer_dim": args.transformer_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "dropout": args.dropout,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "gaussian_noise_std": args.gaussian_noise_std,
        "early_stopping_patience": args.early_stopping_patience,
=======
>>>>>>> origin/feature/map
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
