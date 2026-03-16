import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from train_distracted_classifier import (
    CLASS_NAMES,
    ClipDataset,
    TemporalTransformerClassifier,
    collect_labeled_samples,
    load_clip_frames,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a distracted/focused Transformer model: metrics, confusion matrix, and error galleries."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/distracted_classifier/labeled"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/distracted_classifier/best_distracted_transformer.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distracted_analysis"))
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-errors-per-type", type=int, default=50)
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model_bundle(checkpoint_path: Path, device: torch.device):
    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    model = TemporalTransformerClassifier(
        backbone_name=checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        transformer_dim=checkpoint["transformer_dim"],
        num_heads=checkpoint["transformer_heads"],
        num_layers=checkpoint["transformer_layers"],
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "class_names": checkpoint["class_names"],
        "frames_per_clip": checkpoint["frames_per_clip"],
        "image_size": checkpoint["image_size"],
        "checkpoint": checkpoint,
    }


def select_samples(args: argparse.Namespace):
    all_samples = collect_labeled_samples(args.dataset_root)
    train_samples, val_samples = split_samples(all_samples, args.val_split, args.seed)
    if args.split == "train":
        return list(train_samples)
    if args.split == "all":
        return list(all_samples)
    return list(val_samples)


def build_predictions(
    samples,
    transform,
    frames_per_clip: int,
    model,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> List[Dict[str, object]]:
    dataset = ClipDataset(samples, transform, frames_per_clip)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    rows: List[Dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            batch_size_actual = clips.size(0)
            for idx in range(batch_size_actual):
                sample = samples[sample_index]
                true_idx = int(labels[idx].item())
                pred_idx = int(predictions[idx].item())
                prob_row = probabilities[idx].detach().cpu().numpy().tolist()
                row = {
                    "clip_path": str(sample.path),
                    "group_key": sample.group_key,
                    "true_label": class_names[true_idx],
                    "pred_label": class_names[pred_idx],
                    "confidence": float(confidences[idx].item()),
                    "is_correct": bool(pred_idx == true_idx),
                }
                for class_pos, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(prob_row[class_pos])
                rows.append(row)
                sample_index += 1
    return rows


def compute_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for row in rows if row["true_label"] == "distracted" and row["pred_label"] == "distracted")
    fp = sum(1 for row in rows if row["true_label"] == "focused" and row["pred_label"] == "distracted")
    fn = sum(1 for row in rows if row["true_label"] == "distracted" and row["pred_label"] == "focused")
    correct = sum(1 for row in rows if row["is_correct"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = correct / max(len(rows), 1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def build_confusion_matrix(rows: Sequence[Dict[str, object]], class_names: Sequence[str]) -> np.ndarray:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int32)
    for row in rows:
        true_idx = class_to_idx[str(row["true_label"])]
        pred_idx = class_to_idx[str(row["pred_label"])]
        matrix[true_idx, pred_idx] += 1
    return matrix


def save_confusion_matrix_artifacts(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred", *class_names])
        for row_idx, class_name in enumerate(class_names):
            writer.writerow([class_name, *matrix[row_idx].tolist()])

    cell_size = 180
    margin = 140
    width = margin + cell_size * len(class_names)
    height = margin + cell_size * len(class_names)
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    max_value = max(int(matrix.max()), 1)
    for row_idx, true_name in enumerate(class_names):
        cv2.putText(
            image,
            true_name,
            (20, margin + row_idx * cell_size + cell_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for col_idx, pred_name in enumerate(class_names):
            if row_idx == 0:
                cv2.putText(
                    image,
                    pred_name,
                    (margin + col_idx * cell_size + 15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
            value = int(matrix[row_idx, col_idx])
            intensity = int(255 - (value / max_value) * 190)
            color = (255, intensity, intensity) if row_idx != col_idx else (intensity, 255, intensity)
            x1 = margin + col_idx * cell_size
            y1 = margin + row_idx * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), thickness=2)
            cv2.putText(
                image,
                str(value),
                (x1 + 60, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_dir / "confusion_matrix.png"), image)


def save_predictions_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_clip_preview(clip_path: Path, title_lines: Sequence[str], output_path: Path) -> None:
    frames = load_clip_frames(clip_path, frames_per_clip=3)
    preview_frames = []
    for image in frames:
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        preview_frames.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR))

    strip = np.hstack(preview_frames)
    canvas = np.full((300, strip.shape[1], 3), 245, dtype=np.uint8)
    canvas[76 : 76 + strip.shape[0], :, :] = strip
    y = 28
    for line in title_lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 24
    cv2.imwrite(str(output_path), canvas)


def save_error_gallery(
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
    max_errors_per_type: int,
) -> None:
    false_positives = [
        row for row in rows if row["true_label"] == "focused" and row["pred_label"] == "distracted"
    ]
    false_negatives = [
        row for row in rows if row["true_label"] == "distracted" and row["pred_label"] == "focused"
    ]

    for error_name, error_rows in {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }.items():
        error_dir = output_dir / error_name
        error_dir.mkdir(parents=True, exist_ok=True)
        sorted_rows = sorted(error_rows, key=lambda item: float(item["confidence"]), reverse=True)
        manifest_rows = []
        for index, row in enumerate(sorted_rows[:max_errors_per_type], start=1):
            clip_path = Path(str(row["clip_path"]))
            preview_name = f"{index:03d}_{clip_path.stem}.jpg"
            title_lines = [
                f"{error_name[:-1]} #{index}",
                f"true={row['true_label']} pred={row['pred_label']} conf={float(row['confidence']):.3f}",
                clip_path.name,
            ]
            make_clip_preview(clip_path, title_lines, error_dir / preview_name)
            manifest_rows.append(row)
        save_predictions_csv(manifest_rows, error_dir / "manifest.csv")


def save_summary(
    args: argparse.Namespace,
    bundle,
    sample_count: int,
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "val_split": args.val_split,
        "seed": args.seed,
        "sample_count": sample_count,
        "class_names": bundle["class_names"],
        "frames_per_clip": bundle["frames_per_clip"],
        "image_size": bundle["image_size"],
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        "confusion_matrix": confusion_matrix.tolist(),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bundle = load_model_bundle(args.checkpoint, device)
    transform = build_transform(bundle["image_size"])
    samples = select_samples(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_predictions(
        samples=samples,
        transform=transform,
        frames_per_clip=bundle["frames_per_clip"],
        model=bundle["model"],
        class_names=bundle["class_names"],
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    metrics = compute_metrics(rows)
    matrix = build_confusion_matrix(rows, bundle["class_names"])

    save_predictions_csv(rows, args.output_dir / "predictions.csv")
    save_confusion_matrix_artifacts(matrix, bundle["class_names"], args.output_dir)
    save_error_gallery(rows, args.output_dir, args.max_errors_per_type)
    save_summary(args, bundle, len(rows), metrics, matrix, args.output_dir / "summary.json")

    print(f"Device: {device}")
    print(f"Samples analyzed: {len(rows)}")
    print(
        f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
