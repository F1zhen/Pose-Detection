import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from analyze_distracted_model import build_transform, load_model_bundle, resolve_device
from train_distracted_classifier import VIDEO_EXTENSIONS, load_clip_frames


@dataclass(frozen=True)
class UnlabeledClipSample:
    path: Path
    relative_path: Path


class UnlabeledClipDataset(Dataset):
    def __init__(self, samples: Sequence[UnlabeledClipSample], transform, frames_per_clip: int) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frames = load_clip_frames(sample.path, self.frames_per_clip)
        clip_tensor = torch.stack([self.transform(frame) for frame in frames], dim=0)
        return clip_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-automatic prelabeling for distracted clips using the trained Transformer."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/distracted_classifier/unlabeled"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/distracted_classifier/best_distracted_transformer.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distracted_prelabel"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--focused-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold to auto-suggest focused.",
    )
    parser.add_argument(
        "--distracted-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold to auto-suggest distracted.",
    )
    parser.add_argument(
        "--copy-clips",
        action="store_true",
        help="Copy clips into suggested/review folders for manual verification.",
    )
    return parser.parse_args()


def collect_unlabeled_samples(input_dir: Path) -> List[UnlabeledClipSample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    samples: List[UnlabeledClipSample] = []
    for clip_path in sorted(input_dir.rglob("*")):
        if not clip_path.is_file() or clip_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        samples.append(
            UnlabeledClipSample(
                path=clip_path,
                relative_path=clip_path.relative_to(input_dir),
            )
        )
    if not samples:
        raise FileNotFoundError(f"No clip files found in: {input_dir}")
    return samples


def determine_bucket(
    pred_label: str,
    confidence: float,
    focused_threshold: float,
    distracted_threshold: float,
) -> str:
    if pred_label == "focused" and confidence >= focused_threshold:
        return "suggested_focused"
    if pred_label == "distracted" and confidence >= distracted_threshold:
        return "suggested_distracted"
    return "review"


def build_target_path(base_dir: Path, bucket: str, relative_path: Path) -> Path:
    return base_dir / bucket / relative_path


def copy_clip_if_needed(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        shutil.copy2(source_path, target_path)


def run_inference(
    samples: Sequence[UnlabeledClipSample],
    model_bundle,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> List[Dict[str, object]]:
    transform = build_transform(model_bundle["image_size"])
    dataset = UnlabeledClipDataset(samples, transform, model_bundle["frames_per_clip"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = model_bundle["model"]
    rows: List[Dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for clips in loader:
            clips = clips.to(device)
            logits = model(clips)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for idx in range(clips.size(0)):
                sample = samples[sample_index]
                pred_idx = int(predictions[idx].item())
                prob_row = probabilities[idx].detach().cpu().numpy().tolist()
                row = {
                    "clip_path": str(sample.path),
                    "relative_path": str(sample.relative_path),
                    "pred_label": model_bundle["class_names"][pred_idx],
                    "confidence": float(confidences[idx].item()),
                }
                for class_pos, class_name in enumerate(model_bundle["class_names"]):
                    row[f"prob_{class_name}"] = float(prob_row[class_pos])
                rows.append(row)
                sample_index += 1
    return rows


def save_rows(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_bundle = load_model_bundle(args.checkpoint, device)
    samples = collect_unlabeled_samples(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_inference(
        samples=samples,
        model_bundle=model_bundle,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    bucket_counts: Dict[str, int] = {
        "suggested_focused": 0,
        "suggested_distracted": 0,
        "review": 0,
    }
    for row in rows:
        bucket = determine_bucket(
            pred_label=str(row["pred_label"]),
            confidence=float(row["confidence"]),
            focused_threshold=args.focused_threshold,
            distracted_threshold=args.distracted_threshold,
        )
        row["bucket"] = bucket
        relative_path = Path(str(row["relative_path"]))
        suggested_path = build_target_path(args.output_dir, bucket, relative_path)
        row["suggested_path"] = str(suggested_path)
        bucket_counts[bucket] += 1
        if args.copy_clips:
            copy_clip_if_needed(Path(str(row["clip_path"])), suggested_path)

    save_rows(rows, args.output_dir / "prelabel_predictions.csv")
    for bucket in bucket_counts:
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        save_rows(bucket_rows, args.output_dir / f"{bucket}.csv")

    print(f"Device: {device}")
    print(f"Clips analyzed: {len(rows)}")
    print(
        f"suggested_focused={bucket_counts['suggested_focused']} "
        f"suggested_distracted={bucket_counts['suggested_distracted']} "
        f"review={bucket_counts['review']}"
    )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
