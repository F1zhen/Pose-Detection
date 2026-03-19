import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import cv2
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_MODEL = "yolov8s-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export person crops for focused/distracted photo labeling."
    )
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/distracted_classifier/unlabeled"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--frame-step", type=int, default=10, help="Export every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-box-width", type=int, default=40)
    parser.add_argument("--min-box-height", type=int, default=80)
    parser.add_argument("--bbox-padding", type=float, default=0.12, help="Padding ratio around bbox.")
    return parser.parse_args()


def find_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(f"Unsupported file extension: {input_path.suffix}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    videos = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {input_path}")
    return videos


def expand_bbox(
    bbox: Iterable[float],
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(frame_width, int(x2 + pad_x))
    bottom = min(frame_height, int(y2 + pad_y))
    return left, top, right, bottom


def export_crops_for_video(args: argparse.Namespace, model: YOLO, video_path: Path, device: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_output_dir = args.output_dir / video_path.stem
    crops_dir = video_output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = video_output_dir / "metadata.csv"

    metadata_rows: List[dict[str, object]] = []
    frame_index = 0
    crop_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            if frame_index % max(args.frame_step, 1) != 0:
                frame_index += 1
                continue

            result = model.predict(
                frame,
                verbose=False,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=[0],
                device=device,
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for det_index, bbox in enumerate(boxes):
                    x1, y1, x2, y2 = bbox.tolist()
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    if width < args.min_box_width or height < args.min_box_height:
                        continue

                    left, top, right, bottom = expand_bbox(
                        bbox=bbox,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        padding_ratio=args.bbox_padding,
                    )
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue

                    timestamp_sec = frame_index / fps if fps else 0.0
                    crop_name = (
                        f"{video_path.stem}_f{frame_index:06d}_"
                        f"idna_"
                        f"det{det_index:02d}.jpg"
                    )
                    crop_path = crops_dir / crop_name
                    cv2.imwrite(str(crop_path), crop)

                    metadata_rows.append(
                        {
                            "image_path": str(crop_path),
                            "video_path": str(video_path),
                            "frame": frame_index,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "track_id": None,
                            "det_confidence": float(scores[det_index]),
                            "bbox_x1": left,
                            "bbox_y1": top,
                            "bbox_x2": right,
                            "bbox_y2": bottom,
                            "crop_width": int(right - left),
                            "crop_height": int(bottom - top),
                        }
                    )
                    crop_index += 1

            frame_index += 1
    finally:
        cap.release()

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "image_path",
            "video_path",
            "frame",
            "timestamp_sec",
            "track_id",
            "det_confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "crop_width",
            "crop_height",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"{video_path.name}: exported {crop_index} crops -> {crops_dir}")


def main() -> None:
    args = parse_args()
    videos = find_videos(args.input)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    print(f"Found {len(videos)} video(s). Device: {device}")
    for video_path in videos:
        export_crops_for_video(args, model, video_path, device)


if __name__ == "__main__":
    main()
