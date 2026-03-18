import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_MODEL = "yolov8s-pose.pt"


@dataclass
class PairSample:
    frame_index: int
    timestamp_sec: float
    crop: "cv2.typing.MatLike"
    pair_distance_px: float
    det_confidence_a: float
    det_confidence_b: float


@dataclass
class PairBuffer:
    samples: List[PairSample] = field(default_factory=list)
    last_seen_frame: Optional[int] = None
    last_export_end_frame: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pair clips for horseplay labeling based on proximity candidates."
    )
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/horseplay_classifier/unlabeled"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--clip-length", type=int, default=32)
    parser.add_argument("--clip-stride", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-box-width", type=int, default=40)
    parser.add_argument("--min-box-height", type=int, default=80)
    parser.add_argument("--bbox-padding", type=float, default=0.15)
    parser.add_argument("--clip-size", type=int, default=224)
    parser.add_argument("--max-gap-frames", type=int, default=9)
    parser.add_argument("--proximity-px", type=float, default=140.0)
    parser.add_argument("--codec", default="mp4v")
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


def compute_center(bbox: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pair_key(track_id_a: int, track_id_b: int) -> str:
    first, second = sorted((int(track_id_a), int(track_id_b)))
    return f"{first}_{second}"


def union_bbox(
    bbox_a: Iterable[float],
    bbox_b: Iterable[float],
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1 = min(float(bbox_a[0]), float(bbox_b[0]))
    y1 = min(float(bbox_a[1]), float(bbox_b[1]))
    x2 = max(float(bbox_a[2]), float(bbox_b[2]))
    y2 = max(float(bbox_a[3]), float(bbox_b[3]))
    return expand_bbox((x1, y1, x2, y2), frame_width, frame_height, padding_ratio)


def write_clip(clip_path: Path, samples: List[PairSample], clip_size: int, fps: float, codec: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (clip_size, clip_size))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {clip_path}")
    try:
        for sample in samples:
            frame = cv2.resize(sample.crop, (clip_size, clip_size), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()


def trim_pair_buffer(buffer: PairBuffer, clip_length: int, clip_stride: int) -> None:
    keep_count = max(clip_length + clip_stride, clip_length)
    if len(buffer.samples) > keep_count:
        buffer.samples = buffer.samples[-keep_count:]


def maybe_export_pair_clip(
    args: argparse.Namespace,
    video_path: Path,
    clips_dir: Path,
    metadata_rows: List[dict[str, object]],
    buffer: PairBuffer,
    pair_id: str,
    sampled_fps: float,
) -> int:
    if len(buffer.samples) < args.clip_length:
        return 0

    window = buffer.samples[-args.clip_length:]
    start_frame = window[0].frame_index
    end_frame = window[-1].frame_index
    if buffer.last_export_end_frame is not None:
        frame_delta = end_frame - buffer.last_export_end_frame
        min_delta = max(args.clip_stride, 1) * max(args.frame_step, 1)
        if frame_delta < min_delta:
            return 0

    clip_name = f"{video_path.stem}_pair{pair_id}_f{start_frame:06d}_f{end_frame:06d}.mp4"
    clip_path = clips_dir / clip_name
    write_clip(clip_path, window, args.clip_size, sampled_fps, args.codec)

    buffer.last_export_end_frame = end_frame
    metadata_rows.append(
        {
            "clip_path": str(clip_path),
            "video_path": str(video_path),
            "pair_id": pair_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_timestamp_sec": round(window[0].timestamp_sec, 3),
            "end_timestamp_sec": round(window[-1].timestamp_sec, 3),
            "num_frames": len(window),
            "sampled_fps": round(sampled_fps, 3),
            "frame_step": args.frame_step,
            "clip_length": args.clip_length,
            "clip_stride": args.clip_stride,
            "avg_pair_distance_px": round(sum(item.pair_distance_px for item in window) / len(window), 2),
            "avg_det_confidence_a": round(sum(item.det_confidence_a for item in window) / len(window), 4),
            "avg_det_confidence_b": round(sum(item.det_confidence_b for item in window) / len(window), 4),
        }
    )
    return 1


def export_horseplay_candidates_for_video(
    args: argparse.Namespace,
    model: YOLO,
    video_path: Path,
    device: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sampled_fps = fps / max(args.frame_step, 1)

    video_output_dir = args.output_dir / video_path.stem
    clips_dir = video_output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = video_output_dir / "metadata.csv"

    pair_buffers: Dict[str, PairBuffer] = {}
    metadata_rows: List[dict[str, object]] = []
    frame_index = 0
    clip_count = 0

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

            result = model.track(
                frame,
                persist=True,
                verbose=False,
                tracker="botsort.yaml",
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=[0],
                device=device,
            )[0]

            active_pair_ids: set[str] = set()
            if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                valid_indices: List[int] = []
                centers: Dict[int, Tuple[float, float]] = {}
                for det_index, bbox in enumerate(boxes):
                    x1, y1, x2, y2 = bbox.tolist()
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    if width < args.min_box_width or height < args.min_box_height:
                        continue
                    valid_indices.append(det_index)
                    centers[det_index] = compute_center(bbox)

                for i, det_idx_a in enumerate(valid_indices):
                    for det_idx_b in valid_indices[i + 1:]:
                        center_a = centers[det_idx_a]
                        center_b = centers[det_idx_b]
                        pair_distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
                        if pair_distance > args.proximity_px:
                            continue

                        track_id_a = int(track_ids[det_idx_a])
                        track_id_b = int(track_ids[det_idx_b])
                        pid = pair_key(track_id_a, track_id_b)
                        active_pair_ids.add(pid)

                        left, top, right, bottom = union_bbox(
                            boxes[det_idx_a],
                            boxes[det_idx_b],
                            frame_width,
                            frame_height,
                            args.bbox_padding,
                        )
                        crop = frame[top:bottom, left:right]
                        if crop.size == 0:
                            continue

                        buffer = pair_buffers.setdefault(pid, PairBuffer())
                        if (
                            buffer.last_seen_frame is not None
                            and frame_index - buffer.last_seen_frame > max(args.max_gap_frames, args.frame_step)
                        ):
                            buffer.samples.clear()
                            buffer.last_export_end_frame = None

                        timestamp_sec = frame_index / fps if fps else 0.0
                        buffer.samples.append(
                            PairSample(
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec,
                                crop=crop.copy(),
                                pair_distance_px=pair_distance,
                                det_confidence_a=float(scores[det_idx_a]),
                                det_confidence_b=float(scores[det_idx_b]),
                            )
                        )
                        buffer.last_seen_frame = frame_index
                        clip_count += maybe_export_pair_clip(
                            args=args,
                            video_path=video_path,
                            clips_dir=clips_dir,
                            metadata_rows=metadata_rows,
                            buffer=buffer,
                            pair_id=pid,
                            sampled_fps=sampled_fps,
                        )
                        trim_pair_buffer(buffer, args.clip_length, args.clip_stride)

            for pid, buffer in pair_buffers.items():
                if pid in active_pair_ids:
                    continue
                if (
                    buffer.last_seen_frame is not None
                    and frame_index - buffer.last_seen_frame > max(args.max_gap_frames, args.frame_step)
                ):
                    buffer.samples.clear()
                    buffer.last_export_end_frame = None

            frame_index += 1
    finally:
        cap.release()

    fieldnames = [
        "clip_path",
        "video_path",
        "pair_id",
        "start_frame",
        "end_frame",
        "start_timestamp_sec",
        "end_timestamp_sec",
        "num_frames",
        "sampled_fps",
        "frame_step",
        "clip_length",
        "clip_stride",
        "avg_pair_distance_px",
        "avg_det_confidence_a",
        "avg_det_confidence_b",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"{video_path.name}: exported {clip_count} horseplay candidate clips -> {clips_dir}")


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
        export_horseplay_candidates_for_video(args, model, video_path, device)


if __name__ == "__main__":
    main()
