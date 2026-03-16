import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["focused", "distracted"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CLIP_NAME_PATTERN = re.compile(r"^(?P<video>.+?)_id(?P<track>\d+)_f\d+_f\d+$")


@dataclass(frozen=True)
class ClipSample:
    path: Path
    label: int
    group_key: str


class ClipDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ClipSample],
        transform,
        frames_per_clip: int,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frames = load_clip_frames(sample.path, self.frames_per_clip)
        tensor_frames = [self.transform(frame) for frame in frames]
        clip_tensor = torch.stack(tensor_frames, dim=0)
        return clip_tensor, sample.label


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a distracted/focused clip classifier with a temporal Transformer head."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/distracted_classifier/labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/distracted_classifier"))
    parser.add_argument("--model-name", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=1,
        help="Freeze the frame encoder for the first N epochs to stabilise training on a small dataset.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_group_key(clip_path: Path) -> str:
    match = CLIP_NAME_PATTERN.match(clip_path.stem)
    if match:
        return f"{match.group('video')}|track_{match.group('track')}"
    return f"{clip_path.parent.name}|{clip_path.stem}"


def collect_labeled_samples(dataset_root: Path) -> List[ClipSample]:
    samples: List[ClipSample] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for clip_path in sorted(class_dir.rglob("*")):
            if clip_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            samples.append(
                ClipSample(
                    path=clip_path,
                    label=class_index,
                    group_key=infer_group_key(clip_path),
                )
            )
    if not samples:
        raise FileNotFoundError(
            f"No labeled clips found under {dataset_root}. "
            "Expected folders like labeled/focused and labeled/distracted."
        )
    return samples


def split_samples(
    samples: Sequence[ClipSample],
    val_split: float,
    seed: int,
) -> Tuple[List[ClipSample], List[ClipSample]]:
    grouped: Dict[Tuple[int, str], List[ClipSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.label, sample.group_key), []).append(sample)

    by_class_groups: Dict[int, List[List[ClipSample]]] = {0: [], 1: []}
    for (label, _group_key), group_samples in grouped.items():
        by_class_groups[label].append(group_samples)

    rng = random.Random(seed)
    train_samples: List[ClipSample] = []
    val_samples: List[ClipSample] = []

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


def build_frame_encoder(model_name: str) -> Tuple[nn.Module, int]:
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        encoder = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        )
        return encoder, 1280

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return encoder, model.fc.in_features


def load_clip_frames(clip_path: Path, frames_per_clip: int) -> List[Image.Image]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open clip: {clip_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Clip contains no readable frames: {clip_path}")

    if len(frames) >= frames_per_clip:
        indices = np.linspace(0, len(frames) - 1, frames_per_clip, dtype=int)
        sampled = [frames[idx] for idx in indices]
    else:
        sampled = list(frames)
        while len(sampled) < frames_per_clip:
            sampled.append(sampled[-1])

    return [Image.fromarray(frame) for frame in sampled]


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
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


def set_backbone_trainable(model: TemporalTransformerClassifier, trainable: bool) -> None:
    for parameter in model.frame_encoder.parameters():
        parameter.requires_grad = trainable


def compute_class_weights(samples: Sequence[ClipSample], device: torch.device) -> torch.Tensor:
    counts = [0 for _ in CLASS_NAMES]
    for sample in samples:
        counts[sample.label] += 1
    total = sum(counts)
    weights = [total / max(count * len(CLASS_NAMES), 1) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct = 0

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)

            total_loss += loss.item() * clips.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))

            true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
            false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
            false_negative += int(((predictions == 0) & (labels == 1)).sum().item())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = correct / max(total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy,
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
    train_transform, val_transform = make_transforms(args.image_size)

    train_dataset = ClipDataset(train_samples, train_transform, args.frames_per_clip)
    val_dataset = ClipDataset(val_samples, val_transform, args.frames_per_clip)

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

    model = TemporalTransformerClassifier(
        backbone_name=args.model_name,
        num_classes=len(CLASS_NAMES),
        transformer_dim=args.transformer_dim,
        num_heads=args.transformer_heads,
        num_layers=args.transformer_layers,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_distracted_transformer.pt"
    metadata_path = args.output_dir / "training_metadata.json"

    best_val_f1 = -1.0
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train clips: {len(train_samples)} | Val clips: {len(val_samples)}")

    for epoch in range(1, args.epochs + 1):
        train_backbone = epoch > args.freeze_backbone_epochs
        set_backbone_trainable(model, train_backbone)
        model.train()

        running_loss = 0.0
        seen = 0
        correct = 0

        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            seen += int(labels.size(0))

        scheduler.step()

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
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "backbone_trainable": train_backbone,
        }
        history.append(epoch_metrics)
        print(epoch_metrics)

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "frames_per_clip": args.frames_per_clip,
                    "transformer_dim": args.transformer_dim,
                    "transformer_heads": args.transformer_heads,
                    "transformer_layers": args.transformer_layers,
                    "dropout": args.dropout,
                },
                best_model_path,
            )

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames_per_clip": args.frames_per_clip,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "train_clips": len(train_samples),
        "val_clips": len(val_samples),
        "best_val_f1": round(float(best_val_f1), 4),
        "class_names": CLASS_NAMES,
        "transformer_dim": args.transformer_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "dropout": args.dropout,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
