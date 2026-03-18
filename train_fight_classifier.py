import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import video as video_models


CLASS_NAMES = ["NonFight", "Fight"]
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fight / non-fight video classifier on RWF-2000-style datasets."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/RWF-2000"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fight_classifier"))
    parser.add_argument("--model-name", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-videos", type=int, default=0, help="Use only the first N training videos. 0 = all.")
    parser.add_argument("--max-val-videos", type=int, default=0, help="Use only the first N validation videos. 0 = all.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_video_samples(split_root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset directory: {class_dir}")
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_SUFFIXES:
                samples.append((video_path, class_index))
    if not samples:
        raise FileNotFoundError(f"No video files found under {split_root}")
    return samples


def limit_samples(samples: List[Tuple[Path, int]], max_items: int) -> List[Tuple[Path, int]]:
    if max_items <= 0:
        return samples
    return samples[:max_items]


class VideoFightDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        num_frames: int,
        image_size: int,
        training: bool,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        clip = self._load_clip(video_path)
        return clip, label

    def _load_clip(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frames: List[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                frames.append(frame_resized)
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        clip = self._sample_frames(frames)
        if self.training and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])

        clip = clip / 255.0
        clip = (clip - KINETICS_MEAN) / KINETICS_STD
        return clip

    def _sample_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        frame_count = len(frames)
        if frame_count >= self.num_frames:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, frame_count - 1, self.num_frames)
            indices = np.clip(np.round(indices).astype(int), 0, frame_count - 1)

        sampled = np.stack([frames[int(idx)] for idx in indices], axis=0)
        clip = torch.from_numpy(sampled).permute(3, 0, 1, 2).float()
        return clip


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "r3d_18":
        weights = video_models.R3D_18_Weights.DEFAULT
        model = video_models.r3d_18(weights=weights)
    elif model_name == "mc3_18":
        weights = video_models.MC3_18_Weights.DEFAULT
        model = video_models.mc3_18(weights=weights)
    else:
        weights = video_models.R2Plus1D_18_Weights.DEFAULT
        model = video_models.r2plus1d_18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)
            loss = criterion(logits, labels)

            total_loss += loss.item() * clips.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))

            true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
            false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
            false_negative += int(((predictions == 0) & (labels == 1)).sum().item())

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

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

    train_samples = collect_video_samples(args.dataset_root / "train")
    val_samples = collect_video_samples(args.dataset_root / "val")
    train_samples = limit_samples(train_samples, args.max_train_videos)
    val_samples = limit_samples(val_samples, args.max_val_videos)

    train_dataset = VideoFightDataset(
        samples=train_samples,
        num_frames=args.num_frames,
        image_size=args.image_size,
        training=True,
    )
    val_dataset = VideoFightDataset(
        samples=val_samples,
        num_frames=args.num_frames,
        image_size=args.image_size,
        training=False,
    )

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

    model = build_model(args.model_name, num_classes=len(CLASS_NAMES))
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_fight_classifier.pt"
    metadata_path = args.output_dir / "training_metadata.json"

    best_val_f1 = -1.0
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train videos: {len(train_samples)} | Val videos: {len(val_samples)}")
    print(
        f"Model: {args.model_name} | Frames: {args.num_frames} | Size: {args.image_size} | "
        f"Freeze backbone: {args.freeze_backbone}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0

        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * clips.size(0)
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
            torch.save(
                {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_frames": args.num_frames,
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
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "train_videos": len(train_samples),
        "val_videos": len(val_samples),
        "best_val_f1": round(float(best_val_f1), 4),
        "class_names": CLASS_NAMES,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
