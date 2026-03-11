import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_MODEL = "yolov8s-pose.pt"
PERSON_CLASS_ID = 0
KEYPOINT_CONF_THRESHOLD = 0.4

STAND_HEIGHT_RATIO = 1.15
SHOULDER_Y_THRESHOLD = 0.40
MOVEMENT_THRESHOLD = 80
STANDING_DURATION_SEC = 10
FRAME_SKIP = 10
LOCAL_DEPTH_WINDOW = 0.12
SHOULDER_OFFSET_THRESHOLD = 0.02
POSE_SMOOTHING_WINDOW = 3
POSE_SCORE_THRESHOLD = 0.15
MIN_STATE_FRAMES = 3
MIN_EVENT_DURATION_SEC = 1.0

LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16

POSE_SIT = "sit"
POSE_STAND = "stand"
POSE_UNKNOWN = "unknown"

COLOR_SIT = (0, 200, 0)
COLOR_STAND = (0, 165, 255)
COLOR_UNKNOWN = (180, 180, 180)
COLOR_VIOLATION = (0, 0, 255)
COLOR_TEXT_BG = (25, 25, 25)


@dataclass
class DetectionState:
    person_id: int
    bbox: Tuple[int, int, int, int]
    pose: str
    raw_pose: str
    classifier_pose: Optional[str]
    classifier_confidence: Optional[float]
    violation_type: str
    relative_height_ratio: Optional[float]
    shoulder_y_normalized: Optional[float]
    shoulder_offset: Optional[float]
    local_height_median: Optional[float]
    local_shoulder_median: Optional[float]
    knee_angle: Optional[float]
    center: Tuple[float, float]
    frame_index: int
    timestamp_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classroom video analytics: step 1 sitting/standing classification."
    )
    parser.add_argument("--mode", choices=["calibrate", "analyze"], required=True)
    parser.add_argument("--input", type=Path, default=None, help="Video file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose model path.")
    parser.add_argument("--pose-classifier", type=Path, default=None, help="Path to trained sit/stand classifier checkpoint.")
    parser.add_argument(
        "--classifier-mode",
        choices=["hybrid", "classifier_only"],
        default="hybrid",
        help="How to use the trained classifier if provided.",
    )
    parser.add_argument("--classifier-threshold", type=float, default=0.65)
    parser.add_argument("--classifier-padding", type=float, default=0.12)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda:0, cpu, etc.",
    )
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size for YOLO.")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. 0 means process the whole video.",
    )
    parser.add_argument("--stand-height-ratio", type=float, default=STAND_HEIGHT_RATIO)
    parser.add_argument("--shoulder-y-threshold", type=float, default=SHOULDER_Y_THRESHOLD)
    parser.add_argument("--local-depth-window", type=float, default=LOCAL_DEPTH_WINDOW)
    parser.add_argument("--shoulder-offset-threshold", type=float, default=SHOULDER_OFFSET_THRESHOLD)
    parser.add_argument("--pose-smoothing-window", type=int, default=POSE_SMOOTHING_WINDOW)
    parser.add_argument("--pose-score-threshold", type=float, default=POSE_SCORE_THRESHOLD)
    parser.add_argument("--min-state-frames", type=int, default=MIN_STATE_FRAMES)
    parser.add_argument("--min-event-duration-sec", type=float, default=MIN_EVENT_DURATION_SEC)
    parser.add_argument("--movement-threshold", type=float, default=MOVEMENT_THRESHOLD)
    parser.add_argument("--standing-duration-sec", type=float, default=STANDING_DURATION_SEC)
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="Write every source frame to output video, reusing last known annotations.",
    )
    return parser.parse_args()


def find_input_videos(input_path: Optional[Path]) -> List[Path]:
    if input_path is None:
        input_path = Path("data")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(f"Unsupported file extension: {input_path.suffix}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    video_paths = [
        file_path
        for file_path in sorted(input_path.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if video_paths:
        return video_paths

    raise FileNotFoundError(f"No supported video found in: {input_path}")


def build_output_paths(output_dir: Path, video_path: Path, mode: str) -> Dict[str, Path]:
    stem = video_path.stem
    run_dir = output_dir / f"{stem}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "video": run_dir / f"{stem}_{mode}_annotated.mp4",
        "csv": run_dir / f"{stem}_{mode}_report.csv",
        "xlsx": run_dir / f"{stem}_{mode}_report.xlsx",
        "raw_csv": run_dir / f"{stem}_{mode}_raw_detections.csv",
        "raw_xlsx": run_dir / f"{stem}_{mode}_raw_detections.xlsx",
        "events_csv": run_dir / f"{stem}_{mode}_events.csv",
        "events_xlsx": run_dir / f"{stem}_{mode}_events.xlsx",
        "events_json": run_dir / f"{stem}_{mode}_events.json",
    }


def format_timestamp(timestamp_sec: float) -> str:
    total_ms = int(round(timestamp_sec * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if np.isnan(value):
        return None
    return float(value)


def compute_person_center(bbox_xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_bbox_tuple(bbox_xyxy: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    return (int(x1), int(y1), int(x2), int(y2))


def get_bbox_height(bbox_xyxy: np.ndarray) -> float:
    return float(max(1.0, bbox_xyxy[3] - bbox_xyxy[1]))


def expand_bbox(
    bbox_xyxy: np.ndarray,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(frame_width, int(x2 + pad_x)),
        min(frame_height, int(y2 + pad_y)),
    )


def get_bbox_bottom_normalized(bbox_xyxy: np.ndarray, frame_height: int) -> float:
    if frame_height <= 0:
        return 0.0
    return float(bbox_xyxy[3] / frame_height)


def compute_shoulder_y_normalized(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    frame_height: int,
) -> Optional[float]:
    valid_y: List[float] = []
    for idx in (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX):
        if idx < len(keypoints_conf) and keypoints_conf[idx] > KEYPOINT_CONF_THRESHOLD:
            valid_y.append(float(keypoints_xy[idx][1]))
    if not valid_y or frame_height <= 0:
        return None
    return float(np.mean(valid_y) / frame_height)


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_knee_angle(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> Optional[float]:
    candidates = [
        (LEFT_HIP_IDX, LEFT_KNEE_IDX, LEFT_ANKLE_IDX),
        (RIGHT_HIP_IDX, RIGHT_KNEE_IDX, RIGHT_ANKLE_IDX),
    ]
    angles: List[float] = []
    for hip_idx, knee_idx, ankle_idx in candidates:
        if max(hip_idx, knee_idx, ankle_idx) >= len(keypoints_conf):
            continue
        if (
            keypoints_conf[hip_idx] > KEYPOINT_CONF_THRESHOLD
            and keypoints_conf[knee_idx] > KEYPOINT_CONF_THRESHOLD
            and keypoints_conf[ankle_idx] > KEYPOINT_CONF_THRESHOLD
        ):
            angle = angle_between_points(
                keypoints_xy[hip_idx],
                keypoints_xy[knee_idx],
                keypoints_xy[ankle_idx],
            )
            if angle is not None:
                angles.append(angle)
    if not angles:
        return None
    return float(np.mean(angles))


def classify_pose(
    relative_height_ratio: Optional[float],
    shoulder_y_normalized: Optional[float],
    shoulder_offset: Optional[float],
    stand_height_ratio: float,
    shoulder_y_threshold: float,
    shoulder_offset_threshold: float,
) -> str:
    by_height: Optional[str] = None
    by_shoulder: Optional[str] = None
    by_offset: Optional[str] = None

    if relative_height_ratio is not None:
        by_height = POSE_STAND if relative_height_ratio >= stand_height_ratio else POSE_SIT

    if shoulder_y_normalized is not None:
        by_shoulder = (
            POSE_STAND if shoulder_y_normalized <= shoulder_y_threshold else POSE_SIT
        )

    if shoulder_offset is not None:
        by_offset = (
            POSE_STAND if shoulder_offset >= shoulder_offset_threshold else POSE_SIT
        )

    decisions = [decision for decision in (by_height, by_shoulder, by_offset) if decision]
    if not decisions:
        return POSE_UNKNOWN
    if decisions.count(POSE_STAND) > decisions.count(POSE_SIT):
        return POSE_STAND
    if decisions.count(POSE_SIT) > decisions.count(POSE_STAND):
        return POSE_SIT
    return POSE_UNKNOWN


def compute_local_reference(values: np.ndarray, positions: np.ndarray, index: int, window: float) -> Optional[float]:
    if len(values) == 0 or index >= len(values):
        return None
    position = positions[index]
    mask = np.abs(positions - position) <= window
    local_values = values[mask]
    if len(local_values) == 0:
        return None
    return float(np.median(local_values))


def smooth_pose(
    person_id: int,
    raw_pose: str,
    pose_history: Dict[int, List[str]],
    window_size: int,
    score_threshold: float,
) -> str:
    history = pose_history.setdefault(person_id, [])
    history.append(raw_pose)
    if len(history) > max(window_size, 1):
        del history[:-window_size]

    if not history:
        return POSE_UNKNOWN

    score_map = {POSE_STAND: 1.0, POSE_SIT: -1.0, POSE_UNKNOWN: 0.0}
    score = float(np.mean([score_map[item] for item in history]))
    if score >= score_threshold:
        return POSE_STAND
    if score <= -score_threshold:
        return POSE_SIT
    return POSE_UNKNOWN


def build_classifier_model(model_name: str, num_classes: int, models_module, nn_module):
    if model_name == "efficientnet_b0":
        model = models_module.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn_module.Linear(in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models_module.resnet18(weights=None)
        model.fc = nn_module.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier model_name: {model_name}")


def load_pose_classifier(classifier_path: Optional[Path], device, torch_module):
    if classifier_path is None:
        return None

    from PIL import Image
    from torchvision import models, transforms

    checkpoint = torch_module.load(classifier_path, map_location=device)
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = build_classifier_model(
        model_name=model_name,
        num_classes=len(class_names),
        models_module=models,
        nn_module=torch_module.nn,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "torch": torch_module,
        "pil_image": Image,
    }


def classify_crop_with_model(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    classifier_bundle,
    threshold: float,
    padding_ratio: float,
) -> Tuple[Optional[str], Optional[float]]:
    if classifier_bundle is None:
        return None, None

    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = expand_bbox(
        bbox_xyxy=bbox_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=padding_ratio,
    )
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None, None

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = classifier_bundle["pil_image"].fromarray(rgb_crop)
    tensor = classifier_bundle["transform"](pil_image).unsqueeze(0)
    tensor = tensor.to(next(classifier_bundle["model"].parameters()).device)

    with classifier_bundle["torch"].no_grad():
        logits = classifier_bundle["model"](tensor)
        probabilities = classifier_bundle["torch"].softmax(logits, dim=1)[0]
        confidence, index = probabilities.max(dim=0)

    confidence_value = float(confidence.item())
    if confidence_value < threshold:
        return POSE_UNKNOWN, confidence_value
    return str(classifier_bundle["class_names"][int(index.item())]), confidence_value


def select_box_color(pose: str, violation_type: str) -> Tuple[int, int, int]:
    if violation_type:
        return COLOR_VIOLATION
    if pose == POSE_SIT:
        return COLOR_SIT
    if pose == POSE_STAND:
        return COLOR_STAND
    return COLOR_UNKNOWN


def draw_label(frame: np.ndarray, origin: Tuple[int, int], lines: List[str], color: Tuple[int, int, int]) -> None:
    x, y = origin
    line_height = 18
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines) + 10
    height = line_height * len(lines) + 8
    top = max(0, y - height)
    cv2.rectangle(frame, (x, top), (x + width, y), COLOR_TEXT_BG, thickness=-1)
    cv2.rectangle(frame, (x, top), (x + width, y), color, thickness=1)
    for idx, line in enumerate(lines):
        line_y = top + 18 + idx * line_height
        cv2.putText(
            frame,
            line,
            (x + 5, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def annotate_frame(
    frame: np.ndarray,
    states: Dict[int, DetectionState],
    frame_index: int,
    timestamp_sec: float,
    mode: str,
) -> np.ndarray:
    annotated = frame.copy()
    for state in states.values():
        x1, y1, x2, y2 = state.bbox
        color = select_box_color(state.pose, state.violation_type)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        lines = [f"ID {state.person_id}"]
        cls_value = state.classifier_pose or state.pose
        cls_line = f"cls={cls_value}"
        if state.classifier_confidence is not None:
            cls_line += f" {state.classifier_confidence:.2f}"
        lines.append(cls_line)
        if state.violation_type:
            lines.append(f"violation: {state.violation_type}")

        draw_label(annotated, (x1, max(25, y1)), lines, color)

    timestamp_label = f"{format_timestamp(timestamp_sec)} | frame {frame_index}"
    cv2.rectangle(annotated, (10, 10), (300, 38), COLOR_TEXT_BG, thickness=-1)
    cv2.putText(
        annotated,
        timestamp_label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def results_to_states(
    result,
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    frame_height: int,
    args: argparse.Namespace,
    standing_started_at: Dict[int, float],
    pose_history: Dict[int, List[str]],
    classifier_bundle,
) -> Tuple[Dict[int, DetectionState], List[Dict[str, object]]]:
    states: Dict[int, DetectionState] = {}
    report_rows: List[Dict[str, object]] = []

    if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
        return states, report_rows

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    box_confidences = result.boxes.conf.cpu().numpy()
    heights = np.array([get_bbox_height(box) for box in boxes_xyxy], dtype=float)
    bottom_positions = np.array(
        [get_bbox_bottom_normalized(box, frame_height) for box in boxes_xyxy],
        dtype=float,
    )

    keypoints_xy = None
    keypoints_conf = None
    if result.keypoints is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()

    shoulder_positions = np.array([np.nan] * len(track_ids), dtype=float)
    if keypoints_xy is not None and keypoints_conf is not None:
        for idx in range(min(len(track_ids), len(keypoints_xy))):
            shoulder_positions[idx] = (
                compute_shoulder_y_normalized(keypoints_xy[idx], keypoints_conf[idx], frame_height)
                if idx < len(keypoints_xy)
                else np.nan
            )

    for idx, person_id in enumerate(track_ids):
        bbox = boxes_xyxy[idx]
        bbox_height = get_bbox_height(bbox)
        local_height_median = compute_local_reference(
            values=heights,
            positions=bottom_positions,
            index=idx,
            window=args.local_depth_window,
        )
        relative_height_ratio = (
            float(bbox_height / local_height_median)
            if local_height_median and local_height_median > 0
            else None
        )

        shoulder_y_normalized = None
        local_shoulder_median = None
        shoulder_offset = None
        knee_angle = None
        if keypoints_xy is not None and keypoints_conf is not None and idx < len(keypoints_xy):
            shoulder_y_normalized = safe_float(shoulder_positions[idx])
            valid_shoulder_mask = ~np.isnan(shoulder_positions)
            if shoulder_y_normalized is not None and np.any(valid_shoulder_mask):
                local_shoulder_median = compute_local_reference(
                    values=shoulder_positions[valid_shoulder_mask],
                    positions=bottom_positions[valid_shoulder_mask],
                    index=int(np.sum(valid_shoulder_mask[: idx + 1]) - 1),
                    window=args.local_depth_window,
                )
                if local_shoulder_median is not None:
                    shoulder_offset = float(local_shoulder_median - shoulder_y_normalized)
            knee_angle = compute_knee_angle(keypoints_xy[idx], keypoints_conf[idx])

        heuristic_pose = classify_pose(
            relative_height_ratio=relative_height_ratio,
            shoulder_y_normalized=shoulder_y_normalized,
            shoulder_offset=shoulder_offset,
            stand_height_ratio=args.stand_height_ratio,
            shoulder_y_threshold=args.shoulder_y_threshold,
            shoulder_offset_threshold=args.shoulder_offset_threshold,
        )
        classifier_pose, classifier_confidence = classify_crop_with_model(
            frame=frame,
            bbox_xyxy=bbox,
            classifier_bundle=classifier_bundle,
            threshold=args.classifier_threshold,
            padding_ratio=args.classifier_padding,
        )
        raw_pose = heuristic_pose
        if classifier_bundle is not None:
            if args.classifier_mode == "classifier_only":
                raw_pose = classifier_pose or POSE_UNKNOWN
            elif classifier_pose in (POSE_SIT, POSE_STAND):
                raw_pose = classifier_pose
        pose = smooth_pose(
            person_id=person_id,
            raw_pose=raw_pose,
            pose_history=pose_history,
            window_size=args.pose_smoothing_window,
            score_threshold=args.pose_score_threshold,
        )

        violation_type = ""
        if args.mode == "analyze":
            if pose == POSE_STAND:
                standing_started_at.setdefault(person_id, timestamp_sec)
                standing_time = timestamp_sec - standing_started_at[person_id]
                if standing_time >= args.standing_duration_sec:
                    violation_type = "standing_too_long"
            else:
                standing_started_at.pop(person_id, None)

        center = compute_person_center(bbox)
        state = DetectionState(
            person_id=person_id,
            bbox=get_bbox_tuple(bbox),
            pose=pose,
            raw_pose=raw_pose,
            classifier_pose=classifier_pose,
            classifier_confidence=safe_float(classifier_confidence),
            violation_type=violation_type,
            relative_height_ratio=safe_float(relative_height_ratio),
            shoulder_y_normalized=safe_float(shoulder_y_normalized),
            shoulder_offset=safe_float(shoulder_offset),
            local_height_median=safe_float(local_height_median),
            local_shoulder_median=safe_float(local_shoulder_median),
            knee_angle=safe_float(knee_angle),
            center=center,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        states[person_id] = state
        report_rows.append(
            {
                "timestamp": format_timestamp(timestamp_sec),
                "timestamp_sec": round(timestamp_sec, 3),
                "frame": frame_index,
                "person_id": person_id,
                "track_confidence": safe_float(float(box_confidences[idx])),
                "violation_type": violation_type,
                "pose": pose,
                "raw_pose": raw_pose,
                "classifier_pose": classifier_pose,
                "classifier_confidence": safe_float(classifier_confidence),
                "heuristic_pose": heuristic_pose,
                "bbox_x1": state.bbox[0],
                "bbox_y1": state.bbox[1],
                "bbox_x2": state.bbox[2],
                "bbox_y2": state.bbox[3],
                "relative_height_ratio": safe_float(relative_height_ratio),
                "shoulder_y_normalized": safe_float(shoulder_y_normalized),
                "shoulder_offset": safe_float(shoulder_offset),
                "local_height_median": safe_float(local_height_median),
                "local_shoulder_median": safe_float(local_shoulder_median),
                "knee_angle": safe_float(knee_angle),
            }
        )

    return states, report_rows


def _merge_short_pose_runs(poses: List[str], min_state_frames: int) -> List[str]:
    if not poses or min_state_frames <= 1:
        return poses[:]

    merged = poses[:]
    changed = True
    while changed:
        changed = False
        runs: List[List[object]] = []
        start = 0
        for idx in range(1, len(merged) + 1):
            if idx == len(merged) or merged[idx] != merged[start]:
                runs.append([start, idx - 1, merged[start]])
                start = idx

        for run_index, (start_idx, end_idx, pose_value) in enumerate(runs):
            run_length = end_idx - start_idx + 1
            if run_length >= min_state_frames:
                continue

            prev_pose = runs[run_index - 1][2] if run_index > 0 else None
            next_pose = runs[run_index + 1][2] if run_index + 1 < len(runs) else None
            replacement = prev_pose or next_pose or pose_value
            if prev_pose == next_pose and prev_pose is not None:
                replacement = prev_pose
            elif pose_value == POSE_UNKNOWN and prev_pose is not None:
                replacement = prev_pose
            elif prev_pose is None and next_pose is not None:
                replacement = next_pose

            for pos in range(start_idx, end_idx + 1):
                merged[pos] = replacement
            changed = True
            break

    return merged


def build_state_report(raw_df: pd.DataFrame, min_state_frames: int) -> pd.DataFrame:
    if raw_df.empty:
        state_df = raw_df.copy()
        state_df["stable_pose"] = pd.Series(dtype="object")
        return state_df

    ordered = raw_df.sort_values(["person_id", "frame"]).reset_index(drop=True).copy()
    ordered["stable_pose"] = ordered["pose"]

    for person_id, group in ordered.groupby("person_id", sort=False):
        merged = _merge_short_pose_runs(group["pose"].tolist(), min_state_frames)
        ordered.loc[group.index, "stable_pose"] = merged

    ordered["violation_type"] = ordered["violation_type"].astype(str)
    standing_mask = ordered["stable_pose"] == POSE_STAND
    ordered.loc[standing_mask, "violation_type"] = ordered.loc[standing_mask, "violation_type"].replace("nan", "")
    return ordered


def build_events(state_df: pd.DataFrame, min_event_duration_sec: float) -> pd.DataFrame:
    if state_df.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "event_type",
                "start_timestamp",
                "end_timestamp",
                "start_frame",
                "end_frame",
                "duration_sec",
                "num_observations",
            ]
        )

    events: List[Dict[str, object]] = []
    ordered = state_df.sort_values(["person_id", "frame"]).reset_index(drop=True)
    for person_id, group in ordered.groupby("person_id", sort=False):
        current_pose = None
        start_idx = None
        rows = group.reset_index(drop=True)
        for idx, row in rows.iterrows():
            pose = row["stable_pose"]
            if pose != current_pose:
                if current_pose == POSE_STAND and start_idx is not None:
                    start_row = rows.iloc[start_idx]
                    end_row = rows.iloc[idx - 1]
                    duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                    if duration_sec >= min_event_duration_sec:
                        events.append(
                            {
                                "person_id": int(person_id),
                                "event_type": "standing_interval",
                                "start_timestamp": start_row["timestamp"],
                                "end_timestamp": end_row["timestamp"],
                                "start_frame": int(start_row["frame"]),
                                "end_frame": int(end_row["frame"]),
                                "duration_sec": round(duration_sec, 3),
                                "num_observations": int(idx - start_idx),
                            }
                        )
                current_pose = pose
                start_idx = idx

        if current_pose == POSE_STAND and start_idx is not None:
            start_row = rows.iloc[start_idx]
            end_row = rows.iloc[len(rows) - 1]
            duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
            if duration_sec >= min_event_duration_sec:
                events.append(
                    {
                        "person_id": int(person_id),
                        "event_type": "standing_interval",
                        "start_timestamp": start_row["timestamp"],
                        "end_timestamp": end_row["timestamp"],
                        "start_frame": int(start_row["frame"]),
                        "end_frame": int(end_row["frame"]),
                        "duration_sec": round(duration_sec, 3),
                        "num_observations": int(len(rows) - start_idx),
                    }
                )

    return pd.DataFrame(events)


def create_video_writer(output_path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def analyze_video(args: argparse.Namespace) -> Dict[str, Path]:
    try:
        import torch
        from ultralytics import YOLO
    except OSError as exc:
        raise RuntimeError(
            "Failed to import torch/ultralytics. "
            "Use the project virtual environment: "
            ".\\.venv\\Scripts\\python.exe classroom_analytics.py ... "
            f"Original error: {exc}"
        ) from exc

    video_path = args.input_video
    output_paths = build_output_paths(args.output_dir, video_path, args.mode)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = YOLO(args.model)
    classifier_bundle = load_pose_classifier(args.pose_classifier, torch_device, torch)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = fps if args.save_every_frame else max(fps / max(args.frame_skip, 1), 1.0)
    writer = create_video_writer(output_paths["video"], output_fps, (frame_width, frame_height))

    report_rows: List[Dict[str, object]] = []
    standing_started_at: Dict[int, float] = {}
    pose_history: Dict[int, List[str]] = {}
    last_states: Dict[int, DetectionState] = {}

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            timestamp_sec = frame_index / fps if fps else 0.0
            should_process = frame_index % max(args.frame_skip, 1) == 0

            if should_process:
                results = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    classes=[PERSON_CLASS_ID],
                    device=device,
                )
                result = results[0]
                last_states, frame_rows = results_to_states(
                    result=result,
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    frame_height=frame_height,
                    args=args,
                    standing_started_at=standing_started_at,
                    pose_history=pose_history,
                    classifier_bundle=classifier_bundle,
                )
                report_rows.extend(frame_rows)

            annotated = annotate_frame(
                frame=frame,
                states=last_states,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                mode=args.mode,
            )

            if args.save_every_frame or should_process:
                writer.write(annotated)

            frame_index += 1
    finally:
        cap.release()
        writer.release()

    raw_df = pd.DataFrame(report_rows)
    if raw_df.empty:
        raw_df = pd.DataFrame(
            columns=[
                "timestamp",
                "timestamp_sec",
                "frame",
                "person_id",
                "track_confidence",
                "violation_type",
                "pose",
                "raw_pose",
                "classifier_pose",
                "classifier_confidence",
                "heuristic_pose",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "relative_height_ratio",
                "shoulder_y_normalized",
                "shoulder_offset",
                "local_height_median",
                "local_shoulder_median",
                "knee_angle",
            ]
        )
    state_df = build_state_report(raw_df, args.min_state_frames)
    state_df["pose"] = state_df["stable_pose"]

    events_df = build_events(state_df, args.min_event_duration_sec)
    if not events_df.empty and args.mode == "analyze":
        long_standing = events_df["duration_sec"] >= args.standing_duration_sec
        events_df.loc[long_standing, "event_type"] = "standing_too_long"

    raw_df.to_csv(output_paths["raw_csv"], index=False)
    raw_df.to_excel(output_paths["raw_xlsx"], index=False)
    state_df.to_csv(output_paths["csv"], index=False)
    state_df.to_excel(output_paths["xlsx"], index=False)
    events_df.to_csv(output_paths["events_csv"], index=False)
    events_df.to_excel(output_paths["events_xlsx"], index=False)
    output_paths["events_json"].write_text(
        json.dumps(events_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Device: {device}")
    if args.pose_classifier:
        print(f"Pose classifier: {args.pose_classifier}")
    print(f"Annotated video: {output_paths['video']}")
    print(f"Raw detections CSV: {output_paths['raw_csv']}")
    print(f"CSV report: {output_paths['csv']}")
    print(f"Excel report: {output_paths['xlsx']}")
    print(f"Events CSV: {output_paths['events_csv']}")
    print(f"Events JSON: {output_paths['events_json']}")

    return output_paths


def main() -> None:
    args = parse_args()
    video_paths = find_input_videos(args.input)
    print(f"Found {len(video_paths)} video(s) to process.")
    for video_path in video_paths:
        print(f"Starting: {video_path}")
        args.input_video = video_path
        analyze_video(args)


if __name__ == "__main__":
    main()
