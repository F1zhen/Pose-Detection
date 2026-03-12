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

MOVEMENT_THRESHOLD = 80
FRAME_SKIP = 10
POSE_SMOOTHING_WINDOW = 3
POSE_SCORE_THRESHOLD = 0.15
MIN_STATE_FRAMES = 3
MIN_EVENT_DURATION_SEC = 1.0

# --- Horseplay detection defaults ---
HORSEPLAY_PROXIMITY_PX = 120
HORSEPLAY_SYNC_MOVE_THRESHOLD = 0.70
HORSEPLAY_OSCILLATION_WINDOW = 10
HORSEPLAY_OSCILLATION_MIN = 3
HORSEPLAY_BURST_WINDOW_SEC = 4.0
HORSEPLAY_BURST_MIN = 3
HORSEPLAY_SCORE_THRESHOLD = 2.0
HORSEPLAY_STANDING_WEIGHT = 0.5

POSE_SIT = "sit"
POSE_STAND = "stand"
POSE_UNKNOWN = "unknown"

COLOR_SIT = (0, 200, 0)
COLOR_STAND = (0, 165, 255)
COLOR_UNKNOWN = (180, 180, 180)
COLOR_VIOLATION = (0, 0, 255)
COLOR_TEXT_BG = (25, 25, 25)
COLOR_HORSEPLAY = (0, 200, 255)


@dataclass
class DetectionState:
    person_id: int
    bbox: Tuple[int, int, int, int]
    pose: str
    classifier_pose: Optional[str]
    classifier_confidence: Optional[float]
    violation_type: str
    movement_px: Optional[float]
    rapid_motion: bool
    center: Tuple[float, float]
    frame_index: int
    timestamp_sec: float
    horseplay_score: Optional[float] = None
    horseplay: bool = False
    proximity_ids: str = ""
    oscillation_count: int = 0
    burst_count: int = 0


@dataclass
class LiveReportStats:
    analyzed_frames: int = 0
    unique_people: set[int] | None = None
    sit_frames: int = 0
    stand_frames: int = 0
    unknown_frames: int = 0
    rapid_motion_frames: int = 0
    horseplay_frames: int = 0
    current_people: int = 0
    current_sit: int = 0
    current_stand: int = 0
    current_unknown: int = 0
    current_rapid_motion: int = 0
    current_horseplay: int = 0

    def __post_init__(self) -> None:
        if self.unique_people is None:
            self.unique_people = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classroom video analytics: step 1 sitting/standing classification."
    )
    parser.add_argument("--mode", choices=["calibrate", "analyze"], required=True)
    parser.add_argument("--input", type=Path, default=None, help="Video file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose model path.")
    parser.add_argument("--pose-classifier", type=Path, default=None, help="Path to trained sit/stand classifier checkpoint.")
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
    parser.add_argument("--pose-smoothing-window", type=int, default=POSE_SMOOTHING_WINDOW)
    parser.add_argument("--pose-score-threshold", type=float, default=POSE_SCORE_THRESHOLD)
    parser.add_argument("--min-state-frames", type=int, default=MIN_STATE_FRAMES)
    parser.add_argument("--min-event-duration-sec", type=float, default=MIN_EVENT_DURATION_SEC)
    parser.add_argument("--movement-threshold", type=float, default=MOVEMENT_THRESHOLD)
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="Write every source frame to output video, reusing last known annotations.",
    )
    parser.add_argument(
        "--show-live",
        action="store_true",
        help="Show annotated frames in a live preview window while processing.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Fast mode: only count tracked people, skip pose/classifier/behavior analysis.",
    )

    # --- Horseplay detection ---
    parser.add_argument("--horseplay-proximity-px", type=float, default=HORSEPLAY_PROXIMITY_PX,
                        help="Max center distance (px) to count persons as close.")
    parser.add_argument("--horseplay-sync-threshold", type=float, default=HORSEPLAY_SYNC_MOVE_THRESHOLD,
                        help="Cosine similarity threshold for synchronised movement.")
    parser.add_argument("--horseplay-oscillation-window", type=int, default=HORSEPLAY_OSCILLATION_WINDOW,
                        help="Frame window to count sit/stand flips.")
    parser.add_argument("--horseplay-oscillation-min", type=int, default=HORSEPLAY_OSCILLATION_MIN,
                        help="Min flips in window to count as suspicious.")
    parser.add_argument("--horseplay-burst-window-sec", type=float, default=HORSEPLAY_BURST_WINDOW_SEC,
                        help="Time window (sec) to count rapid-motion bursts.")
    parser.add_argument("--horseplay-burst-min", type=int, default=HORSEPLAY_BURST_MIN,
                        help="Min bursts in window to count as suspicious.")
    parser.add_argument("--horseplay-score-threshold", type=float, default=HORSEPLAY_SCORE_THRESHOLD,
                        help="Composite score threshold to flag horseplay.")
    parser.add_argument("--horseplay-standing-weight", type=float, default=HORSEPLAY_STANDING_WEIGHT,
                        help="Score weight for standing-while-close component.")
    parser.add_argument("--disable-horseplay", action="store_true",
                        help="Disable horseplay analysis entirely.")
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
        "xlsx": run_dir / f"{stem}_{mode}_report.xlsx",
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


def append_violation(existing: str, new_value: str) -> str:
    if not existing:
        return new_value
    existing_items = [item.strip() for item in existing.split(";") if item.strip()]
    if new_value not in existing_items:
        existing_items.append(new_value)
    return ";".join(existing_items)


def console_safe(value: object) -> str:
    text = str(value)
    try:
        text.encode("cp1251")
        return text
    except UnicodeEncodeError:
        return text.encode("cp1251", errors="replace").decode("cp1251")


def compute_person_center(bbox_xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_bbox_tuple(bbox_xyxy: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    return (int(x1), int(y1), int(x2), int(y2))


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


def smooth_pose(
    person_id: int,
    pose_value: str,
    pose_history: Dict[int, List[str]],
    window_size: int,
    score_threshold: float,
) -> str:
    history = pose_history.setdefault(person_id, [])
    history.append(pose_value)
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


class HorseplayTracker:
    """Lightweight per-person sliding history for horseplay signals."""

    def __init__(self) -> None:
        self.prev_pose: Dict[int, str] = {}
        self.pose_flips: Dict[int, List[int]] = {}          # frame indices of sit<->stand
        self.motion_bursts: Dict[int, List[float]] = {}     # timestamps of rapid_motion
        self.movement_vectors: Dict[int, Tuple[float, float]] = {}  # last (dx, dy)

    def update_pose_flip(self, person_id: int, pose: str, frame_index: int) -> None:
        prev = self.prev_pose.get(person_id)
        if prev is not None and prev != pose and prev != POSE_UNKNOWN and pose != POSE_UNKNOWN:
            self.pose_flips.setdefault(person_id, []).append(frame_index)
        self.prev_pose[person_id] = pose

    def update_motion_burst(self, person_id: int, rapid_motion: bool, timestamp_sec: float) -> None:
        if rapid_motion:
            self.motion_bursts.setdefault(person_id, []).append(timestamp_sec)

    def update_movement_vector(self, person_id: int, dx: float, dy: float) -> None:
        self.movement_vectors[person_id] = (dx, dy)

    def count_oscillations(self, person_id: int, frame_index: int, window: int) -> int:
        flips = self.pose_flips.get(person_id, [])
        cutoff = frame_index - window
        recent = [f for f in flips if f >= cutoff]
        self.pose_flips[person_id] = recent  # prune old
        return len(recent)

    def count_bursts(self, person_id: int, timestamp_sec: float, window_sec: float) -> int:
        bursts = self.motion_bursts.get(person_id, [])
        cutoff = timestamp_sec - window_sec
        recent = [t for t in bursts if t >= cutoff]
        self.motion_bursts[person_id] = recent  # prune old
        return len(recent)

    def get_movement_vector(self, person_id: int) -> Optional[Tuple[float, float]]:
        return self.movement_vectors.get(person_id)


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


def select_box_color(pose: str, violation_type: str, is_horseplay: bool = False) -> Tuple[int, int, int]:
    if is_horseplay:
        return COLOR_HORSEPLAY
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


def update_live_report_stats(
    live_stats: LiveReportStats,
    frame_rows: List[Dict[str, object]],
    states: Dict[int, DetectionState],
    count_only: bool,
) -> None:
    if frame_rows:
        live_stats.analyzed_frames += 1
        for row in frame_rows:
            person_id = row.get("person_id")
            if person_id is not None: #УНИКАЛЬНЫЕ ID лЮДЕЙ ЗА ВСЕ ВРЕМЯ, ВОЗМОЖНО НАДО ДОБАВИТЬ ТАЙМЕР ДОБАВЛЕНИЙ И УДАЛЕНИЙ ИЗ НАБОРА, ЧТОБЫ НЕ СЧИТАТЬ ЧЕЛОВЕКА, КОТОРЫЙ ПОЯВИЛСЯ, ПРОПАЛ И ПОЯВИЛСЯ СНОВА, КАК ДВОИХ РАЗНЫХ ЧЕЛОВЕКОВ
                live_stats.unique_people.add(int(person_id))

            if count_only:
                continue

            pose = str(row.get("pose", POSE_UNKNOWN))
            if pose == POSE_SIT:
                live_stats.sit_frames += 1
            elif pose == POSE_STAND:
                live_stats.stand_frames += 1
            else:
                live_stats.unknown_frames += 1

            if bool(row.get("rapid_motion", False)):
                live_stats.rapid_motion_frames += 1
            if bool(row.get("horseplay", False)):
                live_stats.horseplay_frames += 1
# текущее число людей на экране.
    live_stats.current_people = len(states)
    if count_only:
        return

    live_stats.current_sit = sum(1 for state in states.values() if state.pose == POSE_SIT)
    live_stats.current_stand = sum(1 for state in states.values() if state.pose == POSE_STAND)
    live_stats.current_unknown = sum(1 for state in states.values() if state.pose == POSE_UNKNOWN)
    live_stats.current_rapid_motion = sum(1 for state in states.values() if state.rapid_motion)
    live_stats.current_horseplay = sum(1 for state in states.values() if state.horseplay)


def draw_live_report_panel(
    frame: np.ndarray,
    live_stats: LiveReportStats,
    mode: str,
    count_only: bool,
) -> None:
    lines = [
        f"mode {mode}",
        f"now people={live_stats.current_people}",
        f"seen ids={len(live_stats.unique_people)} frames={live_stats.analyzed_frames}",
    ]
    if count_only:
        lines.append("count-only")
    else:
        lines.extend(
            [
                f"now sit={live_stats.current_sit} stand={live_stats.current_stand} unk={live_stats.current_unknown}",
            ]
        )

    line_height = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines) + 18
    height = line_height * len(lines) + 12
    x2 = frame.shape[1] - 10
    x1 = max(10, x2 - width)
    y1 = 10
    y2 = y1 + height

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TEXT_BG, thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), thickness=1)
    for idx, line in enumerate(lines):
        line_y = y1 + 22 + idx * line_height
        cv2.putText(
            frame,
            line,
            (x1 + 8, line_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def annotate_frame(
    frame: np.ndarray,
    states: Dict[int, DetectionState],
    frame_index: int,
    timestamp_sec: float,
    mode: str,
    live_stats: Optional[LiveReportStats] = None,
    count_only: bool = False,
) -> np.ndarray:
    annotated = frame.copy()

    # Draw proximity lines between horseplay partners first (under boxes)
    for state in states.values():
        if state.horseplay and state.proximity_ids:
            cx = (state.bbox[0] + state.bbox[2]) // 2
            cy = (state.bbox[1] + state.bbox[3]) // 2
            for pid_str in state.proximity_ids.split(","):
                pid_str = pid_str.strip()
                if not pid_str:
                    continue
                partner_id = int(pid_str)
                if partner_id in states:
                    p = states[partner_id]
                    pcx = (p.bbox[0] + p.bbox[2]) // 2
                    pcy = (p.bbox[1] + p.bbox[3]) // 2
                    overlay = annotated.copy()
                    cv2.line(overlay, (cx, cy), (pcx, pcy), COLOR_HORSEPLAY, 2)
                    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

    for state in states.values():
        x1, y1, x2, y2 = state.bbox
        color = select_box_color(state.pose, state.violation_type, is_horseplay=state.horseplay)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        lines = [f"ID {state.person_id}"]
        if not count_only:
            cls_value = state.classifier_pose or state.pose
            cls_line = f"cls={cls_value}"
            if state.classifier_confidence is not None:
                cls_line += f" {state.classifier_confidence:.2f}"
            lines.append(cls_line)
        if state.horseplay:
            lines.append(f"!! horseplay {state.horseplay_score:.1f}")
        elif state.violation_type:
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
    if count_only:
        people_count = len(states)
        cv2.rectangle(annotated, (320, 10), (520, 38), COLOR_TEXT_BG, thickness=-1)
        cv2.putText(
            annotated,
            f"people {people_count}",
            (328, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if live_stats is not None:
        draw_live_report_panel(annotated, live_stats=live_stats, mode=mode, count_only=count_only)
    return annotated

# ЗДЕСЬ РЕАЛИЗОВАНА ПОДСЧЕТ ЛЮДЕЙ И СОСТОЯНИЙ БЕЗ АНАЛИЗА ПОЗЫ И КЛАССИФИКАТОРА, ДЛЯ БЫСТРОГО РЕЖИМА count-only
def count_only_states(result, frame_index: int, timestamp_sec: float) -> Tuple[Dict[int, DetectionState], List[Dict[str, object]]]:
    states: Dict[int, DetectionState] = {}
    rows: List[Dict[str, object]] = []

    if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
        return states, rows

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    box_confidences = result.boxes.conf.cpu().numpy()
    people_in_frame = len(track_ids)

    for idx, person_id in enumerate(track_ids):
        bbox = boxes_xyxy[idx]
        state = DetectionState(
            person_id=person_id,
            bbox=get_bbox_tuple(bbox),
            pose=POSE_UNKNOWN,
            classifier_pose=None,
            classifier_confidence=None,
            violation_type="",
            movement_px=None,
            rapid_motion=False,
            center=compute_person_center(bbox),
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        states[person_id] = state
        rows.append(
            {
                "timestamp": format_timestamp(timestamp_sec),
                "timestamp_sec": round(timestamp_sec, 3),
                "frame": frame_index,
                "person_id": person_id,
                "track_confidence": safe_float(float(box_confidences[idx])),
                "people_in_frame": people_in_frame,
            }
        )

    return states, rows


def results_to_states(
    result,
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    args: argparse.Namespace,
    pose_history: Dict[int, List[str]],
    previous_centers: Dict[int, Tuple[float, float]],
    classifier_bundle,
    horseplay_tracker: Optional["HorseplayTracker"] = None,
) -> Tuple[Dict[int, DetectionState], List[Dict[str, object]]]:
    states: Dict[int, DetectionState] = {}
    report_rows: List[Dict[str, object]] = []
    _person_data: Dict[int, Dict] = {}

    if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
        return states, report_rows

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    box_confidences = result.boxes.conf.cpu().numpy()

    for idx, person_id in enumerate(track_ids):
        bbox = boxes_xyxy[idx]
        classifier_pose, classifier_confidence = classify_crop_with_model(
            frame=frame,
            bbox_xyxy=bbox,
            classifier_bundle=classifier_bundle,
            threshold=args.classifier_threshold,
            padding_ratio=args.classifier_padding,
        )
        pose_value = classifier_pose or POSE_UNKNOWN
        pose = smooth_pose(
            person_id=person_id,
            pose_value=pose_value,
            pose_history=pose_history,
            window_size=args.pose_smoothing_window,
            score_threshold=args.pose_score_threshold,
        )

        center = compute_person_center(bbox)
        previous_center = previous_centers.get(person_id)
        movement_px = None
        rapid_motion = False
        if previous_center is not None:
            movement_px = float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]))
            rapid_motion = movement_px >= args.movement_threshold

        violation_type = ""
        if args.mode == "analyze":
            if rapid_motion:
                violation_type = append_violation(violation_type, "rapid_motion")

        previous_centers[person_id] = center

        # --- First pass: collect per-person base data, store temporarily ---
        _person_data[person_id] = {
            "idx": idx,
            "bbox": bbox,
            "pose": pose,
            "classifier_pose": classifier_pose,
            "classifier_confidence": classifier_confidence,
            "violation_type": violation_type,
            "movement_px": movement_px,
            "rapid_motion": rapid_motion,
            "center": center,
            "dx": center[0] - previous_center[0] if previous_center is not None else 0.0,
            "dy": center[1] - previous_center[1] if previous_center is not None else 0.0,
        }

    # --- Horseplay scoring (second pass, needs all centers computed) ---
    hp_enabled = (
        args.mode == "analyze"
        and horseplay_tracker is not None
        and not getattr(args, "disable_horseplay", False)
    )
    person_ids_list = list(_person_data.keys())

    # Pre-compute pairwise proximity
    proximity_map: Dict[int, List[int]] = {pid: [] for pid in person_ids_list}
    if hp_enabled:
        for i, pid_a in enumerate(person_ids_list):
            ca = _person_data[pid_a]["center"]
            for pid_b in person_ids_list[i + 1:]:
                cb = _person_data[pid_b]["center"]
                dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if dist < args.horseplay_proximity_px:
                    proximity_map[pid_a].append(pid_b)
                    proximity_map[pid_b].append(pid_a)

    for person_id in person_ids_list:
        d = _person_data[person_id]
        violation_type = d["violation_type"]

        horseplay_score = 0.0
        horseplay_flag = False
        proximity_ids_str = ""
        oscillation_count = 0
        burst_count = 0

        if hp_enabled:
            ht = horseplay_tracker
            ht.update_pose_flip(person_id, d["pose"], frame_index)
            ht.update_motion_burst(person_id, d["rapid_motion"], timestamp_sec)
            ht.update_movement_vector(person_id, d["dx"], d["dy"])

            # 1. Proximity: +0.5 per close neighbour, max 1.5
            close_ids = proximity_map.get(person_id, [])
            proximity_ids_str = ",".join(str(cid) for cid in close_ids)
            horseplay_score += min(len(close_ids) * 0.5, 1.5)

            # 2. Synchronised movement: +1.0 per synced partner, max 2.0
            sync_bonus = 0.0
            my_vec = ht.get_movement_vector(person_id)
            if my_vec is not None:
                my_mag = float(np.hypot(my_vec[0], my_vec[1]))
                for cid in close_ids:
                    other_vec = ht.get_movement_vector(cid)
                    if other_vec is None:
                        continue
                    other_mag = float(np.hypot(other_vec[0], other_vec[1]))
                    if my_mag > 30 and other_mag > 30:
                        dot = my_vec[0] * other_vec[0] + my_vec[1] * other_vec[1]
                        cos_sim = dot / (my_mag * other_mag)
                        if cos_sim >= args.horseplay_sync_threshold:
                            sync_bonus += 1.0
            horseplay_score += min(sync_bonus, 2.0)

            # 3. Pose oscillation
            oscillation_count = ht.count_oscillations(
                person_id, frame_index, args.horseplay_oscillation_window
            )
            if oscillation_count >= args.horseplay_oscillation_min:
                horseplay_score += 1.0

            # 4. Motion bursts
            burst_count = ht.count_bursts(
                person_id, timestamp_sec, args.horseplay_burst_window_sec
            )
            if burst_count >= args.horseplay_burst_min:
                horseplay_score += 1.0

            # 5. Standing while close to someone
            if d["pose"] == POSE_STAND and len(close_ids) > 0:
                horseplay_score += args.horseplay_standing_weight

            if horseplay_score >= args.horseplay_score_threshold:
                horseplay_flag = True
                violation_type = append_violation(violation_type, "horseplay")

        state = DetectionState(
            person_id=person_id,
            bbox=get_bbox_tuple(d["bbox"]),
            pose=d["pose"],
            classifier_pose=d["classifier_pose"],
            classifier_confidence=safe_float(d["classifier_confidence"]),
            violation_type=violation_type,
            movement_px=safe_float(d["movement_px"]),
            rapid_motion=d["rapid_motion"],
            center=d["center"],
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            horseplay_score=safe_float(horseplay_score) if hp_enabled else None,
            horseplay=horseplay_flag,
            proximity_ids=proximity_ids_str,
            oscillation_count=oscillation_count,
            burst_count=burst_count,
        )
        states[person_id] = state
        report_rows.append(
            {
                "timestamp": format_timestamp(timestamp_sec),
                "timestamp_sec": round(timestamp_sec, 3),
                "frame": frame_index,
                "person_id": person_id,
                "track_confidence": safe_float(float(box_confidences[d["idx"]])),
                "violation_type": violation_type,
                "pose": d["pose"],
                "classifier_pose": d["classifier_pose"],
                "classifier_confidence": safe_float(d["classifier_confidence"]),
                "movement_px": safe_float(d["movement_px"]),
                "rapid_motion": d["rapid_motion"],
                "bbox_x1": state.bbox[0],
                "bbox_y1": state.bbox[1],
                "bbox_x2": state.bbox[2],
                "bbox_y2": state.bbox[3],
                "horseplay_score": safe_float(horseplay_score) if hp_enabled else None,
                "horseplay": horseplay_flag,
                "proximity_ids": proximity_ids_str,
                "oscillation_count": oscillation_count,
                "burst_count": burst_count,
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

    for person_id, group in ordered.groupby("person_id", sort=False):
        rows = group.reset_index(drop=True)
        start_idx = None
        for idx, row in rows.iterrows():
            is_motion = bool(row.get("rapid_motion", False))
            if is_motion and start_idx is None:
                start_idx = idx
            if not is_motion and start_idx is not None:
                start_row = rows.iloc[start_idx]
                end_row = rows.iloc[idx - 1]
                duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                    events.append(
                        {
                            "person_id": int(person_id),
                            "event_type": "rapid_motion_interval",
                            "start_timestamp": start_row["timestamp"],
                            "end_timestamp": end_row["timestamp"],
                            "start_frame": int(start_row["frame"]),
                            "end_frame": int(end_row["frame"]),
                            "duration_sec": round(max(duration_sec, 0.0), 3),
                            "num_observations": int(idx - start_idx),
                        }
                    )
                start_idx = None

        if start_idx is not None:
            start_row = rows.iloc[start_idx]
            end_row = rows.iloc[len(rows) - 1]
            duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
            if duration_sec >= min_event_duration_sec or len(rows) - start_idx >= 1:
                events.append(
                    {
                        "person_id": int(person_id),
                        "event_type": "rapid_motion_interval",
                        "start_timestamp": start_row["timestamp"],
                        "end_timestamp": end_row["timestamp"],
                        "start_frame": int(start_row["frame"]),
                        "end_frame": int(end_row["frame"]),
                        "duration_sec": round(max(duration_sec, 0.0), 3),
                        "num_observations": int(len(rows) - start_idx),
                    }
                )

    return pd.DataFrame(events)


def build_summary(
    state_df: pd.DataFrame,
    events_df: pd.DataFrame,
    video_path: Path,
    mode: str,
    total_frames_read: int,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], str]:
    if state_df.empty:
        summary_row = {
            "video_name": video_path.name,
            "mode": mode,
            "frames_analyzed": 0,
            "unique_people": 0,
            "pose_sit_frames": 0,
            "pose_stand_frames": 0,
            "pose_unknown_frames": 0,
            "rapid_motion_frames": 0,
            "horseplay_frames": 0,
            "standing_intervals": 0,
            "rapid_motion_intervals": 0,
            "horseplay_intervals": 0,
            "standing_too_long_intervals": 0,
            "analysis_start": "",
            "analysis_end": "",
        }
        text = (
            f"Video: {video_path.name}\n"
            f"Mode: {mode}\n"
            f"No detections were recorded.\n"
        )
        return pd.DataFrame([summary_row]), [summary_row], text

    pose_counts = state_df["stable_pose"].value_counts(dropna=False).to_dict()
    rapid_motion_frames = int(state_df["rapid_motion"].fillna(False).astype(bool).sum()) if "rapid_motion" in state_df.columns else 0
    horseplay_frames = int(state_df["horseplay"].fillna(False).astype(bool).sum()) if "horseplay" in state_df.columns else 0

    def count_events(event_name: str) -> int:
        if events_df.empty or "event_type" not in events_df.columns:
            return 0
        return int((events_df["event_type"] == event_name).sum())

    summary_row = {
        "video_name": video_path.name,
        "mode": mode,
        "frames_analyzed": int(state_df["frame"].nunique()),
        "frames_read": int(total_frames_read),
        "unique_people": int(state_df["person_id"].nunique()),
        "pose_sit_frames": int(pose_counts.get(POSE_SIT, 0)),
        "pose_stand_frames": int(pose_counts.get(POSE_STAND, 0)),
        "pose_unknown_frames": int(pose_counts.get(POSE_UNKNOWN, 0)),
        "rapid_motion_frames": rapid_motion_frames,
        "horseplay_frames": horseplay_frames,
        "standing_intervals": count_events("standing_interval"),
        "rapid_motion_intervals": count_events("rapid_motion_interval"),
        "horseplay_intervals": count_events("horseplay_interval"),
        "standing_too_long_intervals": count_events("standing_too_long"),
        "analysis_start": str(state_df["timestamp"].min()),
        "analysis_end": str(state_df["timestamp"].max()),
    }

    per_person_rows: List[Dict[str, object]] = []
    for person_id, group in state_df.groupby("person_id", sort=True):
        person_pose_counts = group["stable_pose"].value_counts(dropna=False).to_dict()
        person_events = events_df[events_df["person_id"] == person_id] if not events_df.empty else pd.DataFrame()
        per_person_rows.append(
            {
                "person_id": int(person_id),
                "frames": int(len(group)),
                "sit_frames": int(person_pose_counts.get(POSE_SIT, 0)),
                "stand_frames": int(person_pose_counts.get(POSE_STAND, 0)),
                "unknown_frames": int(person_pose_counts.get(POSE_UNKNOWN, 0)),
                "rapid_motion_frames": int(group["rapid_motion"].fillna(False).astype(bool).sum()) if "rapid_motion" in group.columns else 0,
                "horseplay_frames": int(group["horseplay"].fillna(False).astype(bool).sum()) if "horseplay" in group.columns else 0,
                "standing_intervals": int((person_events["event_type"] == "standing_interval").sum()) if not person_events.empty else 0,
                "rapid_motion_intervals": int((person_events["event_type"] == "rapid_motion_interval").sum()) if not person_events.empty else 0,
                "horseplay_intervals": int((person_events["event_type"] == "horseplay_interval").sum()) if not person_events.empty else 0,
            }
        )

    text_lines = [
        f"Video: {video_path.name}",
        f"Mode: {mode}",
        f"Frames read: {total_frames_read}",
        f"Frames analyzed: {summary_row['frames_analyzed']}",
        f"Unique people: {summary_row['unique_people']}",
        f"Sit frames: {summary_row['pose_sit_frames']}",
        f"Stand frames: {summary_row['pose_stand_frames']}",
        f"Unknown frames: {summary_row['pose_unknown_frames']}",
        f"Rapid-motion frames: {summary_row['rapid_motion_frames']}",
        f"Horseplay frames: {summary_row['horseplay_frames']}",
        f"Standing intervals: {summary_row['standing_intervals']}",
        f"Rapid-motion intervals: {summary_row['rapid_motion_intervals']}",
        f"Horseplay intervals: {summary_row['horseplay_intervals']}",
        f"Standing-too-long intervals: {summary_row['standing_too_long_intervals']}",
        "",
        "Per-person summary:",
    ]
    for row in per_person_rows:
        text_lines.append(
            f"ID {row['person_id']}: frames={row['frames']}, "
            f"sit={row['sit_frames']}, stand={row['stand_frames']}, unknown={row['unknown_frames']}, "
            f"rapid_motion={row['rapid_motion_frames']}, horseplay={row['horseplay_frames']}"
        )

    return pd.DataFrame([summary_row]), per_person_rows, "\n".join(text_lines) + "\n"


def _build_horseplay_events(
    state_df: pd.DataFrame,
    min_event_duration_sec: float,
) -> pd.DataFrame:
    """Build horseplay_interval events from consecutive horseplay=True rows."""
    if state_df.empty or "horseplay" not in state_df.columns:
        return pd.DataFrame()

    events: List[Dict[str, object]] = []
    ordered = state_df.sort_values(["person_id", "frame"]).reset_index(drop=True)
    for person_id, group in ordered.groupby("person_id", sort=False):
        rows = group.reset_index(drop=True)
        start_idx = None
        peak_score = 0.0
        for idx, row in rows.iterrows():
            is_hp = bool(row.get("horseplay", False))
            if is_hp and start_idx is None:
                start_idx = idx
                peak_score = float(row.get("horseplay_score", 0) or 0)
            elif is_hp and start_idx is not None:
                peak_score = max(peak_score, float(row.get("horseplay_score", 0) or 0))
            if not is_hp and start_idx is not None:
                start_row = rows.iloc[start_idx]
                end_row = rows.iloc[idx - 1]
                duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                    events.append(
                        {
                            "person_id": int(person_id),
                            "event_type": "horseplay_interval",
                            "start_timestamp": start_row["timestamp"],
                            "end_timestamp": end_row["timestamp"],
                            "start_frame": int(start_row["frame"]),
                            "end_frame": int(end_row["frame"]),
                            "duration_sec": round(max(duration_sec, 0.0), 3),
                            "num_observations": int(idx - start_idx),
                            "peak_score": round(peak_score, 2),
                        }
                    )
                start_idx = None
                peak_score = 0.0

        if start_idx is not None:
            start_row = rows.iloc[start_idx]
            end_row = rows.iloc[len(rows) - 1]
            duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
            if duration_sec >= min_event_duration_sec or len(rows) - start_idx >= 1:
                events.append(
                    {
                        "person_id": int(person_id),
                        "event_type": "horseplay_interval",
                        "start_timestamp": start_row["timestamp"],
                        "end_timestamp": end_row["timestamp"],
                        "start_frame": int(start_row["frame"]),
                        "end_frame": int(end_row["frame"]),
                        "duration_sec": round(max(duration_sec, 0.0), 3),
                        "num_observations": int(len(rows) - start_idx),
                        "peak_score": round(peak_score, 2),
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
    if not args.count_only and args.pose_classifier is None:
        raise ValueError("`--pose-classifier` is required when not using `--count-only`.")
    classifier_bundle = None if args.count_only else load_pose_classifier(args.pose_classifier, torch_device, torch)
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
    pose_history: Dict[int, List[str]] = {}
    previous_centers: Dict[int, Tuple[float, float]] = {}
    last_states: Dict[int, DetectionState] = {}
    live_stats = LiveReportStats()
    horseplay_tracker = None if args.count_only else HorseplayTracker()

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
                if args.count_only:
                    last_states, frame_rows = count_only_states(
                        result=result,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                else:
                    last_states, frame_rows = results_to_states(
                        result=result,
                        frame=frame,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        args=args,
                        pose_history=pose_history,
                        previous_centers=previous_centers,
                        classifier_bundle=classifier_bundle,
                        horseplay_tracker=horseplay_tracker,
                    )
                report_rows.extend(frame_rows)
                update_live_report_stats(
                    live_stats=live_stats,
                    frame_rows=frame_rows,
                    states=last_states,
                    count_only=args.count_only,
                )

            annotated = annotate_frame(
                frame=frame,
                states=last_states,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                mode=args.mode,
                live_stats=live_stats,
                count_only=args.count_only,
            )

            if args.save_every_frame or should_process:
                writer.write(annotated)
            if args.show_live:
                cv2.imshow("Classroom Analytics Live", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if args.show_live:
            cv2.destroyAllWindows()

    raw_df = pd.DataFrame(report_rows)
    if args.count_only:
        if raw_df.empty:
            raw_df = pd.DataFrame(columns=["timestamp", "timestamp_sec", "frame", "person_id", "track_confidence", "people_in_frame"])
        summary_row = {
            "video_name": video_path.name,
            "mode": args.mode,
            "frames_read": int(frame_index),
            "frames_analyzed": int(raw_df["frame"].nunique()) if not raw_df.empty else 0,
            "unique_people": int(raw_df["person_id"].nunique()) if not raw_df.empty else 0,
            "max_people_in_frame": int(raw_df["people_in_frame"].max()) if not raw_df.empty else 0,
            "avg_people_in_frame": round(float(raw_df["people_in_frame"].mean()), 2) if not raw_df.empty else 0.0,
            "analysis_start": str(raw_df["timestamp"].min()) if not raw_df.empty else "",
            "analysis_end": str(raw_df["timestamp"].max()) if not raw_df.empty else "",
        }
        summary_df = pd.DataFrame([summary_row])
        per_person_df = (
            raw_df.groupby("person_id", as_index=False)
            .agg(
                frames=("frame", "count"),
                first_timestamp=("timestamp", "min"),
                last_timestamp=("timestamp", "max"),
                mean_track_confidence=("track_confidence", "mean"),
            )
            .sort_values("person_id")
            if not raw_df.empty
            else pd.DataFrame(columns=["person_id", "frames", "first_timestamp", "last_timestamp", "mean_track_confidence"])
        )
        events_df = pd.DataFrame(columns=["person_id", "event_type", "start_timestamp", "end_timestamp", "start_frame", "end_frame", "duration_sec", "num_observations"])
        summary_text = (
            f"Video: {video_path.name}\n"
            f"Mode: {args.mode} (count-only)\n"
            f"Frames read: {frame_index}\n"
            f"Frames analyzed: {summary_row['frames_analyzed']}\n"
            f"Unique people: {summary_row['unique_people']}\n"
            f"Max people in frame: {summary_row['max_people_in_frame']}\n"
            f"Average people in frame: {summary_row['avg_people_in_frame']}\n"
        )
    elif raw_df.empty:
        raw_df = pd.DataFrame(
            columns=[
                "timestamp",
                "timestamp_sec",
                "frame",
                "person_id",
                "track_confidence",
                "violation_type",
                "pose",
                "classifier_pose",
                "classifier_confidence",
                "movement_px",
                "rapid_motion",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "horseplay_score",
                "horseplay",
                "proximity_ids",
                "oscillation_count",
                "burst_count",
            ]
        )
    if not args.count_only:
        state_df = build_state_report(raw_df, args.min_state_frames)
        state_df["pose"] = state_df["stable_pose"]

        events_df = build_events(state_df, args.min_event_duration_sec)
        hp_events_df = _build_horseplay_events(state_df, args.min_event_duration_sec)
        if not hp_events_df.empty:
            events_df = pd.concat([events_df, hp_events_df], ignore_index=True)
        summary_df, per_person_rows, summary_text = build_summary(
            state_df=state_df,
            events_df=events_df,
            video_path=video_path,
            mode=args.mode,
            total_frames_read=frame_index,
        )
        per_person_df = pd.DataFrame(per_person_rows)

    with pd.ExcelWriter(output_paths["xlsx"], engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_person_df.to_excel(writer, sheet_name="per_person", index=False)
        events_df.to_excel(writer, sheet_name="events", index=False)

    print(f"Input video: {console_safe(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"Device: {device}")
    if args.pose_classifier and not args.count_only:
        print(f"Pose classifier: {console_safe(args.pose_classifier)}")
    print(f"Annotated video: {console_safe(output_paths['video'])}")
    print(f"Excel report: {console_safe(output_paths['xlsx'])}")
    print(summary_text.strip())

    return output_paths


def main() -> None:
    args = parse_args()
    video_paths = find_input_videos(args.input)
    print(f"Found {len(video_paths)} video(s) to process.")
    for video_path in video_paths:
        print(f"Starting: {console_safe(video_path)}")
        args.input_video = video_path
        analyze_video(args)


if __name__ == "__main__":
    main()
