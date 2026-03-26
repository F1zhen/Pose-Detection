import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_MODEL = "yolov8s-pose.pt"
PERSON_CLASS_ID = 0

MOVEMENT_THRESHOLD = 80
FRAME_SKIP = 10
POSE_SMOOTHING_WINDOW = 3
POSE_SCORE_THRESHOLD = 0.15
MIN_STATE_FRAMES = 3
MIN_EVENT_DURATION_SEC = 1.0
RED_BOX_MIN_DURATION_SEC = 3.0

# --- Horseplay detection defaults ---
HORSEPLAY_PROXIMITY_PX = 120
HORSEPLAY_SYNC_MOVE_THRESHOLD = 0.70
HORSEPLAY_OSCILLATION_WINDOW = 10
HORSEPLAY_OSCILLATION_MIN = 3
HORSEPLAY_BURST_WINDOW_SEC = 4.0
HORSEPLAY_BURST_MIN = 3
HORSEPLAY_SCORE_THRESHOLD = 2.0
HORSEPLAY_STANDING_WEIGHT = 0.5
CROWDING_PROXIMITY_PX = 120
CROWDING_MIN_DURATION_SEC = 2.0
CROWDING_SEATED_MAX_MOVEMENT_PX = 25.0
CROWDING_APPROACH_MOVEMENT_PX = 35.0
FIGHT_PARTICIPANT_PROXIMITY_PX = 140.0
FIGHT_PARTICIPANT_MOVEMENT_PX = 45.0
SOUND_WINDOW_SEC = 1.0
SOUND_THRESHOLD = 0.75
SOUND_NEGATIVE_LABEL = "background"
SOUND_POSITIVE_LABEL = "gunshot_or_explosion"
TEACHER_MIN_BOX_HEIGHT_PX = 140.0
TEACHER_ENTER_SEC = 1.0
TEACHER_EXIT_SEC = 3.0

POSE_SIT = "sit"
POSE_STAND = "stand"
POSE_UNKNOWN = "unknown"
BEHAVIOR_UNKNOWN = "unknown"
NORMAL_BEHAVIOR_LABELS = {"normal", "focused", BEHAVIOR_UNKNOWN}
NON_FIGHT_LABELS = {"nonfight", "non-fight", "non_fight", "normal", "no_fight", "no-fight", "negative"}
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)

COLOR_SIT = (0, 200, 0)
COLOR_STAND = (0, 165, 255)
COLOR_UNKNOWN = (180, 180, 180)
COLOR_VIOLATION = (0, 0, 255)
COLOR_TEXT_BG = (25, 25, 25)
COLOR_HORSEPLAY = (0, 200, 255)
COLOR_SOUND_ALERT = (0, 70, 255)
COLOR_TEACHER = (255, 220, 0)
COLOR_TEACHER_ROI = (255, 170, 0)
COLOR_TEACHER_DESK_ROI = (255, 255, 0)


@dataclass
class DetectionState:
    person_id: int
    bbox: Tuple[int, int, int, int]
    pose: str
    classifier_pose: Optional[str]
    classifier_confidence: Optional[float]
    behavior_label: Optional[str]
    behavior_confidence: Optional[float]
    violation_type: str
    movement_px: Optional[float]
    rapid_motion: bool
    center: Tuple[float, float]
    frame_index: int
    timestamp_sec: float
    red_box: bool = False
    red_box_duration_sec: float = 0.0
    red_box_confirmed: bool = False
    horseplay_score: Optional[float] = None
    horseplay: bool = False
    proximity_ids: str = ""
    oscillation_count: int = 0
    burst_count: int = 0
    crowding: bool = False
    crowding_ids: str = ""
    crowding_duration_sec: float = 0.0
    fight_detected: bool = False
    fight_ids: str = ""
    is_teacher: bool = False
    teacher_zone: str = ""


@dataclass
class TrackClipState:
    frames: List[np.ndarray]
    last_label: Optional[str] = None
    last_confidence: Optional[float] = None


@dataclass
class GlobalClipState:
    frames: List[np.ndarray]
    last_label: Optional[str] = None
    last_confidence: Optional[float] = None


@dataclass
class SoundWindow:
    start_sec: float
    end_sec: float
    label: Optional[str]
    confidence: Optional[float]
    alert: bool


@dataclass
class TeacherStatus:
    present: bool = False
    track_id: Optional[int] = None
    zone: str = ""


@dataclass
class LiveReportStats:
    analyzed_frames: int = 0
    unique_people: set[int] | None = None
    sit_frames: int = 0
    stand_frames: int = 0
    unknown_frames: int = 0
    rapid_motion_frames: int = 0
    horseplay_frames: int = 0
    crowding_frames: int = 0
    current_people: int = 0
    current_sit: int = 0
    current_stand: int = 0
    current_unknown: int = 0
    current_rapid_motion: int = 0
    current_horseplay: int = 0
    current_crowding: int = 0
    current_red_boxes: int = 0
    fight_frames: int = 0
    current_fight: bool = False
    sound_alert_frames: int = 0
    current_sound_alert: bool = False
    current_teacher_present: bool = False
    current_teacher_track_id: Optional[int] = None

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
        "--behavior-classifier",
        type=Path,
        default=None,
        help="Path to behavior classifier checkpoint. Supports frame models and the distracted Transformer checkpoint.",
    )
    parser.add_argument("--behavior-threshold", type=float, default=0.65)
    parser.add_argument("--fight-classifier", type=Path, default=None, help="Path to trained fight / non-fight video classifier checkpoint.")
    parser.add_argument("--fight-threshold", type=float, default=0.75)
    parser.add_argument("--sound-classifier", type=Path, default=None, help="Path to audio classifier checkpoint for gunshot / explosion sounds.")
    parser.add_argument("--sound-threshold", type=float, default=SOUND_THRESHOLD)
    parser.add_argument("--sound-window-sec", type=float, default=SOUND_WINDOW_SEC,
                        help="Audio analysis window size in seconds for the sound classifier.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg executable used to extract audio.")
    parser.add_argument("--teacher-board-roi", default="",
                        help="Teacher board ROI polygon. Format: x1,y1;x2,y2;... Use 0..1 normalized coords or pixels.")
    parser.add_argument("--teacher-desk-roi", default="",
                        help="Teacher desk ROI polygon. Format: x1,y1;x2,y2;... Use 0..1 normalized coords or pixels.")
    parser.add_argument("--teacher-min-box-height-px", type=float, default=TEACHER_MIN_BOX_HEIGHT_PX,
                        help="Min bbox height in pixels for teacher candidate filtering.")
    parser.add_argument("--teacher-enter-sec", type=float, default=TEACHER_ENTER_SEC,
                        help="Required time inside teacher ROI before teacher_present becomes true.")
    parser.add_argument("--teacher-exit-sec", type=float, default=TEACHER_EXIT_SEC,
                        help="Allowed time without teacher candidate before teacher_present becomes false.")
    parser.add_argument("--fight-participant-proximity-px", type=float, default=FIGHT_PARTICIPANT_PROXIMITY_PX,
                        help="Max center distance (px) to localise likely fight participants.")
    parser.add_argument("--fight-participant-movement-px", type=float, default=FIGHT_PARTICIPANT_MOVEMENT_PX,
                        help="Min movement (px) to treat a nearby student as actively involved in a fight.")
    parser.add_argument("--red-box-min-duration-sec", type=float, default=RED_BOX_MIN_DURATION_SEC)
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
    parser.add_argument("--crowding-proximity-px", type=float, default=CROWDING_PROXIMITY_PX,
                        help="Max center distance (px) to consider students crowded together.")
    parser.add_argument("--crowding-min-duration-sec", type=float, default=CROWDING_MIN_DURATION_SEC,
                        help="Min duration (sec) of close contact before crowding is confirmed.")
    parser.add_argument("--crowding-seated-max-movement-px", type=float, default=CROWDING_SEATED_MAX_MOVEMENT_PX,
                        help="If both students are seated and move less than this, ignore as same-desk seating.")
    parser.add_argument("--crowding-approach-movement-px", type=float, default=CROWDING_APPROACH_MOVEMENT_PX,
                        help="Movement threshold (px) to treat close interaction as meaningful approach.")
    parser.add_argument("--disable-crowding", action="store_true",
                        help="Disable crowding / approach analysis entirely.")
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
        "video_silent": run_dir / f"{stem}_{mode}_annotated_silent.mp4",
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


def get_bbox_bottom_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def get_bbox_height(bbox: Tuple[int, int, int, int]) -> float:
    return float(max(0, bbox[3] - bbox[1]))


def parse_roi_polygon_spec(spec: str, frame_width: int, frame_height: int) -> Optional[np.ndarray]:
    spec = (spec or "").strip()
    if not spec:
        return None

    tokens = [token.strip() for token in spec.replace("|", ";").split(";") if token.strip()]
    points: List[Tuple[float, float]] = []
    for token in tokens:
        values = [value.strip() for value in token.split(",")]
        if len(values) != 2:
            raise ValueError(
                f"Invalid ROI point '{token}'. Expected format x,y;x,y;..."
            )
        try:
            x_value = float(values[0])
            y_value = float(values[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid ROI coordinate '{token}'. Expected numeric x,y values."
            ) from exc
        points.append((x_value, y_value))

    if len(points) < 3:
        raise ValueError("ROI polygon must contain at least 3 points.")

    use_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    scaled_points = []
    for x_value, y_value in points:
        if use_normalized:
            scaled_x = int(round(x_value * frame_width))
            scaled_y = int(round(y_value * frame_height))
        else:
            scaled_x = int(round(x_value))
            scaled_y = int(round(y_value))
        scaled_points.append(
            (
                min(max(scaled_x, 0), max(frame_width - 1, 0)),
                min(max(scaled_y, 0), max(frame_height - 1, 0)),
            )
        )

    return np.array(scaled_points, dtype=np.int32)


def point_inside_polygon(point: Tuple[float, float], polygon: Optional[np.ndarray]) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def draw_roi_polygon(frame: np.ndarray, polygon: Optional[np.ndarray], color: Tuple[int, int, int], label: str) -> None:
    if polygon is None or len(polygon) < 3:
        return
    contour = polygon.reshape((-1, 1, 2))
    cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=2)
    anchor_x = int(np.min(polygon[:, 0]))
    anchor_y = int(np.min(polygon[:, 1]))
    text_origin = (max(10, anchor_x), max(18, anchor_y - 6))
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


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


class RedBoxTracker:
    def __init__(self) -> None:
        self.active_start_times: Dict[int, float] = {}

    def update(self, person_id: int, is_red: bool, timestamp_sec: float) -> float:
        if not is_red:
            self.active_start_times.pop(person_id, None)
            return 0.0

        start_time = self.active_start_times.setdefault(person_id, timestamp_sec)
        return max(0.0, timestamp_sec - start_time)


class PairDurationTracker:
    def __init__(self) -> None:
        self.active_start_times: Dict[Tuple[int, int], float] = {}

    @staticmethod
    def normalise_pair(person_a: int, person_b: int) -> Tuple[int, int]:
        return (person_a, person_b) if person_a < person_b else (person_b, person_a)

    def update(self, pair_key: Tuple[int, int], is_active: bool, timestamp_sec: float) -> float:
        if not is_active:
            self.active_start_times.pop(pair_key, None)
            return 0.0

        start_time = self.active_start_times.setdefault(pair_key, timestamp_sec)
        return max(0.0, timestamp_sec - start_time)

    def retain_only(self, active_pair_keys: set[Tuple[int, int]]) -> None:
        stale_keys = [pair_key for pair_key in self.active_start_times if pair_key not in active_pair_keys]
        for pair_key in stale_keys:
            self.active_start_times.pop(pair_key, None)


class TeacherTracker:
    def __init__(self, enter_sec: float, exit_sec: float) -> None:
        self.enter_sec = max(enter_sec, 0.0)
        self.exit_sec = max(exit_sec, 0.0)
        self.roi_dwell_sec: Dict[int, float] = {}
        self.pending_track_id: Optional[int] = None
        self.pending_since_sec: Optional[float] = None
        self.current_teacher_id: Optional[int] = None
        self.current_teacher_zone: str = ""
        self.teacher_present: bool = False
        self.absence_since_sec: Optional[float] = None
        self.last_timestamp_sec: Optional[float] = None

    def update(
        self,
        states: Dict[int, DetectionState],
        board_roi: Optional[np.ndarray],
        desk_roi: Optional[np.ndarray],
        min_box_height_px: float,
        timestamp_sec: float,
    ) -> TeacherStatus:
        if board_roi is None and desk_roi is None:
            self.teacher_present = False
            self.current_teacher_id = None
            self.current_teacher_zone = ""
            self.pending_track_id = None
            self.pending_since_sec = None
            self.absence_since_sec = None
            self.last_timestamp_sec = timestamp_sec
            return TeacherStatus()

        delta_sec = 0.0
        if self.last_timestamp_sec is not None:
            delta_sec = max(0.0, timestamp_sec - self.last_timestamp_sec)
        self.last_timestamp_sec = timestamp_sec

        candidates: Dict[int, Dict[str, object]] = {}
        for person_id, state in states.items():
            bbox = state.bbox
            if state.pose != POSE_STAND:
                continue
            if get_bbox_height(bbox) < min_box_height_px:
                continue

            foot_point = get_bbox_bottom_center(bbox)
            in_board = point_inside_polygon(foot_point, board_roi)
            in_desk = point_inside_polygon(foot_point, desk_roi)
            if not in_board and not in_desk:
                continue

            zone = "board" if in_board else "desk"
            self.roi_dwell_sec[person_id] = self.roi_dwell_sec.get(person_id, 0.0) + delta_sec
            candidates[person_id] = {
                "zone": zone,
                "dwell_sec": self.roi_dwell_sec.get(person_id, 0.0),
            }

        best_track_id: Optional[int] = None
        best_zone = ""
        if candidates:
            def rank(item: Tuple[int, Dict[str, object]]) -> Tuple[float, int, int]:
                person_id, data = item
                zone_priority = 1 if str(data["zone"]) == "board" else 0
                current_priority = 1 if person_id == self.current_teacher_id else 0
                return (float(data["dwell_sec"]), zone_priority, current_priority)

            best_track_id, best_data = max(candidates.items(), key=rank)
            best_zone = str(best_data["zone"])

        if best_track_id is not None:
            self.absence_since_sec = None
            if self.teacher_present:
                self.current_teacher_id = best_track_id
                self.current_teacher_zone = best_zone
                self.pending_track_id = None
                self.pending_since_sec = None
            else:
                if self.pending_track_id != best_track_id:
                    self.pending_track_id = best_track_id
                    self.pending_since_sec = timestamp_sec
                if self.pending_since_sec is None:
                    self.pending_since_sec = timestamp_sec
                if timestamp_sec - self.pending_since_sec >= self.enter_sec:
                    self.teacher_present = True
                    self.current_teacher_id = best_track_id
                    self.current_teacher_zone = best_zone
                    self.pending_track_id = None
                    self.pending_since_sec = None
        else:
            self.pending_track_id = None
            self.pending_since_sec = None
            if self.teacher_present:
                if self.absence_since_sec is None:
                    self.absence_since_sec = timestamp_sec
                if timestamp_sec - self.absence_since_sec >= self.exit_sec:
                    self.teacher_present = False
                    self.current_teacher_id = None
                    self.current_teacher_zone = ""
                    self.absence_since_sec = None
            else:
                self.current_teacher_id = None
                self.current_teacher_zone = ""

        return TeacherStatus(
            present=self.teacher_present,
            track_id=self.current_teacher_id,
            zone=self.current_teacher_zone,
        )


class PositionalEncoding:
    def __init__(self, dim: int, torch_module, dropout: float = 0.1, max_len: int = 512) -> None:
        self.dropout = torch_module.nn.Dropout(dropout)
        position = torch_module.arange(max_len).unsqueeze(1)
        div_term = torch_module.exp(torch_module.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch_module.zeros(max_len, dim)
        pe[:, 0::2] = torch_module.sin(position * div_term)
        pe[:, 1::2] = torch_module.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def to(self, device):
        self.pe = self.pe.to(device)
        return self

    def __call__(self, x):
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
        models_module,
        nn_module,
    ) -> None:
        super().__init__()
        self.frame_encoder, encoder_dim = build_temporal_frame_encoder(backbone_name, models_module, nn_module)
        self.projection = nn_module.Linear(encoder_dim, transformer_dim)
        self.cls_token = nn_module.Parameter(torch.zeros(1, 1, transformer_dim))
        nn_module.init.normal_(self.cls_token, std=0.02)
        self.position = PositionalEncoding(transformer_dim, torch, dropout=dropout, max_len=1024)
        encoder_layer = nn_module.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn_module.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn_module.LayerNorm(transformer_dim)
        self.head = nn_module.Linear(transformer_dim, num_classes)

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


def build_temporal_frame_encoder(model_name: str, models_module, nn_module):
    if model_name == "efficientnet_b0":
        model = models_module.efficientnet_b0(weights=None)
        encoder = nn_module.Sequential(
            model.features,
            model.avgpool,
            nn_module.Flatten(),
        )
        return encoder, 1280
    if model_name == "resnet18":
        model = models_module.resnet18(weights=None)
        encoder = nn_module.Sequential(*list(model.children())[:-1], nn_module.Flatten())
        return encoder, model.fc.in_features
    raise ValueError(f"Unsupported temporal backbone model_name: {model_name}")


def load_classifier_bundle(classifier_path: Optional[Path], device, torch_module):
    if classifier_path is None:
        return None

    from PIL import Image
    from torchvision import models, transforms

    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch_module.load(classifier_path, weights_only=True, **load_kwargs)
    except TypeError:
        checkpoint = torch_module.load(classifier_path, **load_kwargs)
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    if "frames_per_clip" in checkpoint and "transformer_dim" in checkpoint:
        model = TemporalTransformerClassifier(
            backbone_name=model_name,
            num_classes=len(class_names),
            transformer_dim=checkpoint["transformer_dim"],
            num_heads=checkpoint["transformer_heads"],
            num_layers=checkpoint["transformer_layers"],
            dropout=checkpoint.get("dropout", 0.1),
            models_module=models,
            nn_module=torch_module.nn,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        model.position.to(device)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return {
            "kind": "temporal",
            "model": model,
            "class_names": class_names,
            "transform": transform,
            "torch": torch_module,
            "pil_image": Image,
            "frames_per_clip": checkpoint["frames_per_clip"],
        }

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
        "kind": "frame",
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "torch": torch_module,
        "pil_image": Image,
    }


def build_fight_classifier_model(model_name: str, num_classes: int):
    from torchvision.models import video as video_models

    if model_name == "r3d_18":
        model = video_models.r3d_18(weights=None)
    elif model_name == "mc3_18":
        model = video_models.mc3_18(weights=None)
    elif model_name == "r2plus1d_18":
        model = video_models.r2plus1d_18(weights=None)
    else:
        raise ValueError(f"Unsupported fight classifier model_name: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_fight_classifier_bundle(classifier_path: Optional[Path], device, torch_module):
    if classifier_path is None:
        return None

    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch_module.load(classifier_path, weights_only=True, **load_kwargs)
    except TypeError:
        checkpoint = torch_module.load(classifier_path, **load_kwargs)

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    num_frames = int(checkpoint["num_frames"])

    model = build_fight_classifier_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return {
        "kind": "fight_video",
        "model": model,
        "class_names": class_names,
        "torch": torch_module,
        "image_size": image_size,
        "num_frames": num_frames,
    }


def build_sound_classifier_model(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def load_sound_classifier_bundle(classifier_path: Optional[Path], device, torch_module):
    if classifier_path is None:
        return None

    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch_module.load(classifier_path, weights_only=True, **load_kwargs)
    except TypeError:
        checkpoint = torch_module.load(classifier_path, **load_kwargs)

    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    num_classes = int(checkpoint["num_classes"])
    dropout = float(checkpoint.get("dropout", 0.0))
    class_names = checkpoint.get("class_names")
    if class_names is None:
        if num_classes == 2:
            class_names = [SOUND_NEGATIVE_LABEL, SOUND_POSITIVE_LABEL]
        else:
            class_names = [f"class_{index}" for index in range(num_classes)]

    model = build_sound_classifier_model(input_dim, hidden_dim, num_classes, dropout)
    state_dict = checkpoint["state_dict"]
    if any(key.startswith("network.") for key in state_dict):
        state_dict = {key.replace("network.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "kind": "sound",
        "model": model,
        "torch": torch_module,
        "input_dim": input_dim,
        "class_names": [str(name) for name in class_names],
        "negative_index": 0,
    }


def extract_audio_to_wav(video_path: Path, wav_path: Path, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or pass --ffmpeg-bin with the executable path."
        ) from exc

    if completed.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"ffmpeg failed while extracting audio from {video_path}.\n{stderr}"
        )


def mux_original_audio(video_without_audio: Path, source_video: Path, output_path: Path, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_without_audio),
        "-i",
        str(source_video),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while muxing audio into {output_path}.\n{completed.stderr.strip()}"
        )


def load_wav_audio_signal(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), sample_rate
    return audio / float(np.iinfo(np.int16).max), sample_rate


def build_sound_feature_vector(audio_chunk: np.ndarray, input_dim: int) -> np.ndarray:
    if input_dim < 2:
        raise ValueError(f"Sound classifier input_dim must be >= 2, got {input_dim}")

    n_fft = (input_dim - 1) * 2
    hop_length = max(n_fft // 2, 1)
    if audio_chunk.size == 0:
        return np.zeros(input_dim, dtype=np.float32)

    if audio_chunk.size < n_fft:
        padded = np.pad(audio_chunk, (0, n_fft - audio_chunk.size))
        frames = padded.reshape(1, n_fft)
    else:
        windows: List[np.ndarray] = []
        max_start = audio_chunk.size - n_fft
        for start in range(0, max_start + 1, hop_length):
            windows.append(audio_chunk[start : start + n_fft])
        if max_start > 0 and max_start % hop_length != 0:
            windows.append(audio_chunk[-n_fft:])
        frames = np.stack(windows, axis=0)

    hann = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * hann[None, :], axis=1)).astype(np.float32)
    feature = np.log1p(spectrum).mean(axis=0)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-6:
        feature = feature / norm
    return feature[:input_dim].astype(np.float32, copy=False)


def load_sound_windows_with_model(
    video_path: Path,
    classifier_bundle,
    window_sec: float,
    threshold: float,
    ffmpeg_bin: str,
) -> List[SoundWindow]:
    if classifier_bundle is None:
        return []

    ffmpeg_path = shutil.which(ffmpeg_bin) or ffmpeg_bin
    analysis_window_sec = max(window_sec, 0.1)
    samples_per_window = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        extract_audio_to_wav(video_path, wav_path, ffmpeg_path)
        audio, sample_rate = load_wav_audio_signal(wav_path)
        samples_per_window = max(1, int(round(analysis_window_sec * sample_rate)))

    if audio.size == 0:
        return []

    windows: List[SoundWindow] = []
    alert_class_indices = set(range(len(classifier_bundle["class_names"]))) - {int(classifier_bundle["negative_index"])}
    device = next(classifier_bundle["model"].parameters()).device
    for start_index in range(0, audio.size, samples_per_window):
        chunk = audio[start_index : start_index + samples_per_window]
        if chunk.size == 0:
            continue

        feature = build_sound_feature_vector(chunk, classifier_bundle["input_dim"])
        tensor = classifier_bundle["torch"].from_numpy(feature).unsqueeze(0).to(device)
        with classifier_bundle["torch"].no_grad():
            logits = classifier_bundle["model"](tensor)
            probabilities = classifier_bundle["torch"].softmax(logits, dim=1)[0]
            confidence, index = probabilities.max(dim=0)

        predicted_index = int(index.item())
        confidence_value = float(confidence.item())
        start_sec = start_index / sample_rate
        end_sec = min((start_index + chunk.size) / sample_rate, audio.size / sample_rate)
        windows.append(
            SoundWindow(
                start_sec=start_sec,
                end_sec=end_sec,
                label=str(classifier_bundle["class_names"][predicted_index]),
                confidence=confidence_value,
                alert=predicted_index in alert_class_indices and confidence_value >= threshold,
            )
        )

    return windows


def get_active_sound_window(windows: List[SoundWindow], cursor: int, timestamp_sec: float) -> Tuple[Optional[SoundWindow], int]:
    if not windows:
        return None, 0

    index = min(max(cursor, 0), len(windows) - 1)
    while index + 1 < len(windows) and timestamp_sec >= windows[index].end_sec:
        index += 1

    current = windows[index]
    if current.start_sec <= timestamp_sec < current.end_sec:
        return current, index
    return None, index


def update_global_clip_state(
    frame: np.ndarray,
    clip_state: GlobalClipState,
    frames_per_clip: int,
) -> Optional[List[np.ndarray]]:
    clip_state.frames.append(frame.copy())
    if len(clip_state.frames) > frames_per_clip:
        clip_state.frames = clip_state.frames[-frames_per_clip:]
    if len(clip_state.frames) < frames_per_clip:
        return None
    return clip_state.frames


def classify_global_clip_with_model(
    frame: np.ndarray,
    classifier_bundle,
    threshold: float,
    clip_state: GlobalClipState,
) -> Tuple[Optional[str], Optional[float]]:
    if classifier_bundle is None:
        return None, None

    clip_frames = update_global_clip_state(
        frame=frame,
        clip_state=clip_state,
        frames_per_clip=classifier_bundle["num_frames"],
    )
    if clip_frames is None:
        return clip_state.last_label, clip_state.last_confidence

    transformed_frames = []
    image_size = classifier_bundle["image_size"]
    for clip_frame in clip_frames:
        rgb_frame = cv2.cvtColor(clip_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = classifier_bundle["torch"].from_numpy(resized).permute(2, 0, 1).float() / 255.0
        transformed_frames.append(tensor)

    clip_tensor = classifier_bundle["torch"].stack(transformed_frames, dim=1).unsqueeze(0)
    clip_tensor = (clip_tensor - KINETICS_MEAN.to(clip_tensor.device)) / KINETICS_STD.to(clip_tensor.device)
    clip_tensor = clip_tensor.to(next(classifier_bundle["model"].parameters()).device)

    with classifier_bundle["torch"].no_grad():
        logits = classifier_bundle["model"](clip_tensor)
        probabilities = classifier_bundle["torch"].softmax(logits, dim=1)[0]
        confidence, index = probabilities.max(dim=0)

    confidence_value = float(confidence.item())
    if confidence_value < threshold:
        clip_state.last_label = None
        clip_state.last_confidence = confidence_value
        return None, confidence_value

    label = str(classifier_bundle["class_names"][int(index.item())])
    clip_state.last_label = label
    clip_state.last_confidence = confidence_value
    return label, confidence_value


def is_fight_label(label: Optional[str]) -> bool:
    if not label:
        return False
    return label.strip().lower() not in NON_FIGHT_LABELS


def localize_fight_participants(
    states: Dict[int, DetectionState],
    proximity_px: float,
    movement_px: float,
) -> Dict[int, List[int]]:
    if len(states) < 2:
        return {}

    def is_aggressive(state: DetectionState) -> bool:
        movement_value = state.movement_px or 0.0
        return bool(
            state.rapid_motion
            or movement_value >= movement_px
            or state.horseplay
            or (state.crowding and state.pose != POSE_SIT)
            or (state.pose == POSE_STAND and movement_value >= movement_px * 0.6)
        )

    aggressive_ids = {person_id for person_id, state in states.items() if is_aggressive(state)}
    if len(aggressive_ids) < 2:
        return {}

    participant_map: Dict[int, set[int]] = {}
    state_items = list(states.items())
    for idx, (person_a, state_a) in enumerate(state_items):
        if person_a not in aggressive_ids:
            continue
        for person_b, state_b in state_items[idx + 1:]:
            if person_b not in aggressive_ids:
                continue
            dist = float(np.hypot(state_a.center[0] - state_b.center[0], state_a.center[1] - state_b.center[1]))
            if dist > proximity_px:
                continue
            participant_map.setdefault(person_a, set()).add(person_b)
            participant_map.setdefault(person_b, set()).add(person_a)

    return {person_id: sorted(partners) for person_id, partners in participant_map.items()}


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
        return None, confidence_value
    return str(classifier_bundle["class_names"][int(index.item())]), confidence_value


def update_track_clip_state(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    person_id: int,
    clip_state_map: Dict[int, TrackClipState],
    frames_per_clip: int,
    padding_ratio: float,
) -> Optional[List[np.ndarray]]:
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = expand_bbox(
        bbox_xyxy=bbox_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=padding_ratio,
    )
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    state = clip_state_map.setdefault(person_id, TrackClipState(frames=[]))
    state.frames.append(crop.copy())
    if len(state.frames) > frames_per_clip:
        state.frames = state.frames[-frames_per_clip:]
    if len(state.frames) < frames_per_clip:
        return None
    return state.frames


def classify_track_with_temporal_model(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    person_id: int,
    classifier_bundle,
    threshold: float,
    padding_ratio: float,
    clip_state_map: Dict[int, TrackClipState],
) -> Tuple[Optional[str], Optional[float]]:
    if classifier_bundle is None:
        return None, None
    if classifier_bundle.get("kind") != "temporal":
        return classify_crop_with_model(frame, bbox_xyxy, classifier_bundle, threshold, padding_ratio)

    clip_frames = update_track_clip_state(
        frame=frame,
        bbox_xyxy=bbox_xyxy,
        person_id=person_id,
        clip_state_map=clip_state_map,
        frames_per_clip=classifier_bundle["frames_per_clip"],
        padding_ratio=padding_ratio,
    )
    state = clip_state_map.setdefault(person_id, TrackClipState(frames=[]))
    if clip_frames is None:
        return state.last_label, state.last_confidence

    transformed_frames = []
    for crop in clip_frames:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = classifier_bundle["pil_image"].fromarray(rgb_crop)
        transformed_frames.append(classifier_bundle["transform"](pil_image))
    clip_tensor = classifier_bundle["torch"].stack(transformed_frames, dim=0).unsqueeze(0)
    clip_tensor = clip_tensor.to(next(classifier_bundle["model"].parameters()).device)

    with classifier_bundle["torch"].no_grad():
        logits = classifier_bundle["model"](clip_tensor)
        probabilities = classifier_bundle["torch"].softmax(logits, dim=1)[0]
        confidence, index = probabilities.max(dim=0)

    confidence_value = float(confidence.item())
    if confidence_value < threshold:
        state.last_label = None
        state.last_confidence = confidence_value
        return None, confidence_value

    label = str(classifier_bundle["class_names"][int(index.item())])
    state.last_label = label
    state.last_confidence = confidence_value
    return label, confidence_value


def prune_track_clip_states(
    clip_state_map: Dict[int, TrackClipState],
    active_person_ids: List[int],
) -> None:
    active = set(active_person_ids)
    stale_ids = [person_id for person_id in clip_state_map if person_id not in active]
    for person_id in stale_ids:
        clip_state_map.pop(person_id, None)


def select_box_color(
    pose: str,
    violation_type: str,
    is_horseplay: bool = False,
    is_red_box: bool = False,
    is_teacher: bool = False,
) -> Tuple[int, int, int]:
    if is_red_box:
        return COLOR_VIOLATION
    if is_horseplay:
        return COLOR_HORSEPLAY
    if violation_type:
        return COLOR_VIOLATION
    if is_teacher:
        return COLOR_TEACHER
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
    sound_alert: bool = False,
    teacher_present: bool = False,
    teacher_track_id: Optional[int] = None,
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
            if bool(row.get("crowding", False)):
                live_stats.crowding_frames += 1
            if bool(row.get("horseplay", False)):
                live_stats.horseplay_frames += 1
            if bool(row.get("fight_detected", False)):
                live_stats.fight_frames += 1
        if sound_alert:
            live_stats.sound_alert_frames += 1
# текущее число людей на экране.
    live_stats.current_people = len(states)
    live_stats.current_sound_alert = sound_alert
    live_stats.current_teacher_present = teacher_present
    live_stats.current_teacher_track_id = teacher_track_id
    if count_only:
        return

    live_stats.current_sit = sum(1 for state in states.values() if state.pose == POSE_SIT)
    live_stats.current_stand = sum(1 for state in states.values() if state.pose == POSE_STAND)
    live_stats.current_unknown = sum(1 for state in states.values() if state.pose == POSE_UNKNOWN)
    live_stats.current_rapid_motion = sum(1 for state in states.values() if state.rapid_motion)
    live_stats.current_crowding = sum(1 for state in states.values() if state.crowding)
    live_stats.current_horseplay = sum(1 for state in states.values() if state.horseplay)
    live_stats.current_red_boxes = sum(1 for state in states.values() if state.red_box_confirmed)
    live_stats.current_fight = any("fight" in str(state.violation_type).lower() for state in states.values())


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
                f"crowding={live_stats.current_crowding}",
                f"red boxes 3s={live_stats.current_red_boxes}",
                f"fight={'yes' if live_stats.current_fight else 'no'}",
                f"teacher={'yes' if live_stats.current_teacher_present else 'no'} id={live_stats.current_teacher_track_id if live_stats.current_teacher_track_id is not None else '-'}",
                f"sound_alert={'yes' if live_stats.current_sound_alert else 'no'} total={live_stats.sound_alert_frames}",
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
    fight_label: Optional[str] = None,
    fight_confidence: Optional[float] = None,
    fight_active: bool = False,
    teacher_present: bool = False,
    teacher_track_id: Optional[int] = None,
    teacher_zone: str = "",
    teacher_board_roi: Optional[np.ndarray] = None,
    teacher_desk_roi: Optional[np.ndarray] = None,
    sound_label: Optional[str] = None,
    sound_confidence: Optional[float] = None,
    sound_active: bool = False,
) -> np.ndarray:
    annotated = frame.copy()

    draw_roi_polygon(annotated, teacher_board_roi, COLOR_TEACHER_ROI, "teacher_board_roi")
    draw_roi_polygon(annotated, teacher_desk_roi, COLOR_TEACHER_DESK_ROI, "teacher_desk_roi")

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
        color = select_box_color(
            state.pose,
            state.violation_type,
            is_horseplay=state.horseplay,
            is_red_box=state.red_box_confirmed,
            is_teacher=state.is_teacher,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        lines = [f"ID {state.person_id}"]
        if not count_only:
            cls_value = state.classifier_pose or state.pose
            cls_line = f"cls={cls_value}"
            if state.classifier_confidence is not None:
                cls_line += f" {state.classifier_confidence:.2f}"
            lines.append(cls_line)
            if state.behavior_label and state.behavior_label.lower() not in NORMAL_BEHAVIOR_LABELS:
                behavior_line = f"beh={state.behavior_label}"
                if state.behavior_confidence is not None:
                    behavior_line += f" {state.behavior_confidence:.2f}"
                lines.append(behavior_line)
            if state.red_box_confirmed:
                lines.append(f"red {state.red_box_duration_sec:.1f}s")
            if state.crowding:
                lines.append(f"crowding {state.crowding_duration_sec:.1f}s")
            if state.fight_detected:
                lines.append("fight")
            if state.is_teacher:
                lines.append(f"teacher {state.teacher_zone or 'roi'}")
        if state.horseplay:
            lines.append(f"!! horseplay {state.horseplay_score:.1f}")
        elif state.violation_type:
            lines.append(f"violation: {state.violation_type}")

        draw_label(annotated, (x1, max(25, y1)), lines, color)

    if fight_label:
        banner = f"fight={fight_label}"
        if fight_confidence is not None:
            banner += f" {fight_confidence:.2f}"
        banner_color = COLOR_VIOLATION if fight_active else (90, 90, 90)
        cv2.rectangle(annotated, (10, 42), (280, 70), COLOR_TEXT_BG, thickness=-1)
        cv2.rectangle(annotated, (10, 42), (280, 70), banner_color, thickness=1)
        cv2.putText(
            annotated,
            banner,
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if sound_label:
        banner = f"sound={sound_label}"
        if sound_confidence is not None:
            banner += f" {sound_confidence:.2f}"
        banner_color = COLOR_SOUND_ALERT if sound_active else (90, 90, 90)
        cv2.rectangle(annotated, (10, 74), (360, 102), COLOR_TEXT_BG, thickness=-1)
        cv2.rectangle(annotated, (10, 74), (360, 102), banner_color, thickness=1)
        cv2.putText(
            annotated,
            banner,
            (18, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    teacher_banner = f"teacher={'yes' if teacher_present else 'no'}"
    if teacher_track_id is not None:
        teacher_banner += f" id={teacher_track_id}"
    if teacher_zone:
        teacher_banner += f" {teacher_zone}"
    teacher_banner_color = COLOR_TEACHER if teacher_present else (90, 90, 90)
    cv2.rectangle(annotated, (10, 106), (320, 134), COLOR_TEXT_BG, thickness=-1)
    cv2.rectangle(annotated, (10, 106), (320, 134), teacher_banner_color, thickness=1)
    cv2.putText(
        annotated,
        teacher_banner,
        (18, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

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
            behavior_label=None,
            behavior_confidence=None,
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
    pose_classifier_bundle,
    behavior_classifier_bundle,
    behavior_clip_states: Optional[Dict[int, TrackClipState]] = None,
    red_box_tracker: Optional["RedBoxTracker"] = None,
    crowding_tracker: Optional["PairDurationTracker"] = None,
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
    if behavior_clip_states is not None:
        prune_track_clip_states(behavior_clip_states, track_ids)

    for idx, person_id in enumerate(track_ids):
        bbox = boxes_xyxy[idx]
        classifier_pose, classifier_confidence = classify_crop_with_model(
            frame=frame,
            bbox_xyxy=bbox,
            classifier_bundle=pose_classifier_bundle,
            threshold=args.classifier_threshold,
            padding_ratio=args.classifier_padding,
        )
        if behavior_classifier_bundle is not None and behavior_classifier_bundle.get("kind") == "temporal":
            behavior_label, behavior_confidence = classify_track_with_temporal_model(
                frame=frame,
                bbox_xyxy=bbox,
                person_id=person_id,
                classifier_bundle=behavior_classifier_bundle,
                threshold=args.behavior_threshold,
                padding_ratio=args.classifier_padding,
                clip_state_map=behavior_clip_states if behavior_clip_states is not None else {},
            )
        else:
            behavior_label, behavior_confidence = classify_crop_with_model(
                frame=frame,
                bbox_xyxy=bbox,
                classifier_bundle=behavior_classifier_bundle,
                threshold=args.behavior_threshold,
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
        normalized_behavior_label = behavior_label.lower() if behavior_label else None
        red_box = bool(
            normalized_behavior_label
            and normalized_behavior_label not in NORMAL_BEHAVIOR_LABELS
        )
        red_box_duration_sec = (
            red_box_tracker.update(person_id, red_box, timestamp_sec)
            if red_box_tracker is not None
            else 0.0
        )
        red_box_confirmed = red_box and red_box_duration_sec >= args.red_box_min_duration_sec
        if red_box_confirmed:
            violation_type = append_violation(
                violation_type,
                normalized_behavior_label or "behavior",
            )

        previous_centers[person_id] = center

        # --- First pass: collect per-person base data, store temporarily ---
        _person_data[person_id] = {
            "idx": idx,
            "bbox": bbox,
            "pose": pose,
            "classifier_pose": classifier_pose,
            "classifier_confidence": classifier_confidence,
            "behavior_label": behavior_label,
            "behavior_confidence": behavior_confidence,
            "violation_type": violation_type,
            "movement_px": movement_px,
            "rapid_motion": rapid_motion,
            "red_box": red_box,
            "red_box_duration_sec": red_box_duration_sec,
            "red_box_confirmed": red_box_confirmed,
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
    crowding_enabled = (
        args.mode == "analyze"
        and crowding_tracker is not None
        and not getattr(args, "disable_crowding", False)
    )
    person_ids_list = list(_person_data.keys())

    # Pre-compute pairwise proximity
    proximity_map: Dict[int, List[int]] = {pid: [] for pid in person_ids_list}
    crowding_candidate_pairs: set[Tuple[int, int]] = set()
    if hp_enabled:
        for i, pid_a in enumerate(person_ids_list):
            ca = _person_data[pid_a]["center"]
            for pid_b in person_ids_list[i + 1:]:
                cb = _person_data[pid_b]["center"]
                dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if dist < args.horseplay_proximity_px:
                    proximity_map[pid_a].append(pid_b)
                    proximity_map[pid_b].append(pid_a)
    if crowding_enabled:
        for i, pid_a in enumerate(person_ids_list):
            da = _person_data[pid_a]
            ca = da["center"]
            for pid_b in person_ids_list[i + 1:]:
                db = _person_data[pid_b]
                cb = db["center"]
                dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if dist >= args.crowding_proximity_px:
                    continue

                movement_a = safe_float(da["movement_px"]) or 0.0
                movement_b = safe_float(db["movement_px"]) or 0.0
                both_seated_and_still = (
                    da["pose"] == POSE_SIT
                    and db["pose"] == POSE_SIT
                    and movement_a <= args.crowding_seated_max_movement_px
                    and movement_b <= args.crowding_seated_max_movement_px
                )
                meaningful_approach = (
                    da["pose"] != POSE_SIT
                    or db["pose"] != POSE_SIT
                    or da["rapid_motion"]
                    or db["rapid_motion"]
                    or movement_a >= args.crowding_approach_movement_px
                    or movement_b >= args.crowding_approach_movement_px
                )
                if both_seated_and_still or not meaningful_approach:
                    continue

                crowding_candidate_pairs.add(PairDurationTracker.normalise_pair(pid_a, pid_b))

    pair_durations: Dict[Tuple[int, int], float] = {}
    confirmed_crowding_pairs: set[Tuple[int, int]] = set()
    confirmed_crowding_groups: List[set[int]] = []
    if crowding_enabled and crowding_tracker is not None:
        for pair_key in crowding_candidate_pairs:
            pair_durations[pair_key] = crowding_tracker.update(pair_key, True, timestamp_sec)
        crowding_tracker.retain_only(crowding_candidate_pairs)
        confirmed_crowding_pairs = {
            pair_key
            for pair_key, duration_sec in pair_durations.items()
            if duration_sec >= args.crowding_min_duration_sec
        }
        if confirmed_crowding_pairs:
            adjacency: Dict[int, set[int]] = {}
            for person_a, person_b in confirmed_crowding_pairs:
                adjacency.setdefault(person_a, set()).add(person_b)
                adjacency.setdefault(person_b, set()).add(person_a)

            visited: set[int] = set()
            for person_id in adjacency:
                if person_id in visited:
                    continue
                stack = [person_id]
                component: set[int] = set()
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    stack.extend(adjacency.get(current, set()) - visited)
                if len(component) >= 3:
                    confirmed_crowding_groups.append(component)

    for person_id in person_ids_list:
        d = _person_data[person_id]
        violation_type = d["violation_type"]

        horseplay_score = 0.0
        horseplay_flag = False
        crowding_flag = False
        crowding_ids_str = ""
        crowding_duration_sec = 0.0
        proximity_ids_str = ""
        oscillation_count = 0
        burst_count = 0

        if crowding_enabled:
            crowding_partners: List[int] = []
            for group in confirmed_crowding_groups:
                if person_id not in group:
                    continue
                crowding_partners.extend(pid for pid in group if pid != person_id)
                for pair_key in confirmed_crowding_pairs:
                    if pair_key[0] in group and pair_key[1] in group:
                        crowding_duration_sec = max(crowding_duration_sec, pair_durations.get(pair_key, 0.0))
            crowding_flag = len(crowding_partners) > 0
            crowding_ids_str = ",".join(str(pid) for pid in sorted(set(crowding_partners)))
            if crowding_flag:
                violation_type = append_violation(violation_type, "crowding")

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
            behavior_label=d["behavior_label"],
            behavior_confidence=safe_float(d["behavior_confidence"]),
            violation_type=violation_type,
            movement_px=safe_float(d["movement_px"]),
            rapid_motion=d["rapid_motion"],
            center=d["center"],
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            red_box=d["red_box"],
            red_box_duration_sec=safe_float(d["red_box_duration_sec"]) or 0.0,
            red_box_confirmed=d["red_box_confirmed"],
            crowding=crowding_flag,
            crowding_ids=crowding_ids_str,
            crowding_duration_sec=round(crowding_duration_sec, 3),
            fight_detected=False,
            fight_ids="",
            is_teacher=False,
            teacher_zone="",
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
                "behavior_label": d["behavior_label"],
                "behavior_confidence": safe_float(d["behavior_confidence"]),
                "movement_px": safe_float(d["movement_px"]),
                "rapid_motion": d["rapid_motion"],
                "crowding": crowding_flag,
                "crowding_ids": crowding_ids_str,
                "crowding_duration_sec": round(crowding_duration_sec, 3),
                "fight_detected": False,
                "fight_ids": "",
                "is_teacher": False,
                "teacher_zone": "",
                "bbox_x1": state.bbox[0],
                "bbox_y1": state.bbox[1],
                "bbox_x2": state.bbox[2],
                "bbox_y2": state.bbox[3],
                "red_box": d["red_box"],
                "red_box_duration_sec": safe_float(d["red_box_duration_sec"]),
                "red_box_confirmed": d["red_box_confirmed"],
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

    if "red_box_confirmed" in ordered.columns:
        for person_id, group in ordered.groupby("person_id", sort=False):
            rows = group.reset_index(drop=True)
            start_idx = None
            for idx, row in rows.iterrows():
                is_distracted = bool(row.get("red_box_confirmed", False))
                if is_distracted and start_idx is None:
                    start_idx = idx
                if not is_distracted and start_idx is not None:
                    start_row = rows.iloc[start_idx]
                    end_row = rows.iloc[idx - 1]
                    duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                    if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                        events.append(
                            {
                                "person_id": int(person_id),
                                "event_type": "distracted_interval",
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
                            "event_type": "distracted_interval",
                            "start_timestamp": start_row["timestamp"],
                            "end_timestamp": end_row["timestamp"],
                            "start_frame": int(start_row["frame"]),
                            "end_frame": int(end_row["frame"]),
                            "duration_sec": round(max(duration_sec, 0.0), 3),
                            "num_observations": int(len(rows) - start_idx),
                        }
                    )

    if "crowding" in ordered.columns:
        for person_id, group in ordered.groupby("person_id", sort=False):
            rows = group.reset_index(drop=True)
            start_idx = None
            partner_ids = ""
            for idx, row in rows.iterrows():
                is_crowding = bool(row.get("crowding", False))
                if is_crowding and start_idx is None:
                    start_idx = idx
                    partner_ids = str(row.get("crowding_ids", "") or "")
                elif is_crowding and start_idx is not None and not partner_ids:
                    partner_ids = str(row.get("crowding_ids", "") or "")
                if not is_crowding and start_idx is not None:
                    start_row = rows.iloc[start_idx]
                    end_row = rows.iloc[idx - 1]
                    duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                    if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                        events.append(
                            {
                                "person_id": int(person_id),
                                "event_type": "crowding_interval",
                                "start_timestamp": start_row["timestamp"],
                                "end_timestamp": end_row["timestamp"],
                                "start_frame": int(start_row["frame"]),
                                "end_frame": int(end_row["frame"]),
                                "duration_sec": round(max(duration_sec, 0.0), 3),
                                "num_observations": int(idx - start_idx),
                                "partner_ids": partner_ids,
                            }
                        )
                    start_idx = None
                    partner_ids = ""

            if start_idx is not None:
                start_row = rows.iloc[start_idx]
                end_row = rows.iloc[len(rows) - 1]
                duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                if duration_sec >= min_event_duration_sec or len(rows) - start_idx >= 1:
                    events.append(
                        {
                            "person_id": int(person_id),
                            "event_type": "crowding_interval",
                            "start_timestamp": start_row["timestamp"],
                            "end_timestamp": end_row["timestamp"],
                            "start_frame": int(start_row["frame"]),
                            "end_frame": int(end_row["frame"]),
                            "duration_sec": round(max(duration_sec, 0.0), 3),
                            "num_observations": int(len(rows) - start_idx),
                            "partner_ids": partner_ids,
                        }
                    )

    if "fight_detected" in ordered.columns:
        for person_id, group in ordered.groupby("person_id", sort=False):
            rows = group.reset_index(drop=True)
            start_idx = None
            for idx, row in rows.iterrows():
                is_fight = bool(row.get("fight_detected", False))
                if is_fight and start_idx is None:
                    start_idx = idx
                if not is_fight and start_idx is not None:
                    start_row = rows.iloc[start_idx]
                    end_row = rows.iloc[idx - 1]
                    duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
                    if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                        events.append(
                            {
                                "person_id": int(person_id),
                                "event_type": "fight_interval",
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
                            "event_type": "fight_interval",
                            "start_timestamp": start_row["timestamp"],
                            "end_timestamp": end_row["timestamp"],
                            "start_frame": int(start_row["frame"]),
                            "end_frame": int(end_row["frame"]),
                            "duration_sec": round(max(duration_sec, 0.0), 3),
                            "num_observations": int(len(rows) - start_idx),
                        }
                    )

    return pd.DataFrame(events)


def build_sound_events(sound_frame_df: pd.DataFrame, min_event_duration_sec: float) -> pd.DataFrame:
    if sound_frame_df.empty or "sound_alert" not in sound_frame_df.columns:
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
                "label",
                "confidence",
            ]
        )

    ordered = sound_frame_df.sort_values("frame").reset_index(drop=True)
    events: List[Dict[str, object]] = []
    start_idx = None
    peak_confidence = 0.0
    label = ""

    for idx, row in ordered.iterrows():
        is_alert = bool(row.get("sound_alert", False))
        if is_alert and start_idx is None:
            start_idx = idx
            peak_confidence = float(row.get("sound_confidence", 0.0) or 0.0)
            label = str(row.get("sound_label", "") or "")
        elif is_alert and start_idx is not None:
            peak_confidence = max(peak_confidence, float(row.get("sound_confidence", 0.0) or 0.0))
            if not label:
                label = str(row.get("sound_label", "") or "")

        if not is_alert and start_idx is not None:
            start_row = ordered.iloc[start_idx]
            end_row = ordered.iloc[idx - 1]
            duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
            if duration_sec >= min_event_duration_sec or idx - start_idx >= 1:
                events.append(
                    {
                        "person_id": None,
                        "event_type": "sound_interval",
                        "start_timestamp": start_row["timestamp"],
                        "end_timestamp": end_row["timestamp"],
                        "start_frame": int(start_row["frame"]),
                        "end_frame": int(end_row["frame"]),
                        "duration_sec": round(max(duration_sec, 0.0), 3),
                        "num_observations": int(idx - start_idx),
                        "label": label,
                        "confidence": round(peak_confidence, 4),
                    }
                )
            start_idx = None
            peak_confidence = 0.0
            label = ""

    if start_idx is not None:
        start_row = ordered.iloc[start_idx]
        end_row = ordered.iloc[len(ordered) - 1]
        duration_sec = float(end_row["timestamp_sec"] - start_row["timestamp_sec"])
        if duration_sec >= min_event_duration_sec or len(ordered) - start_idx >= 1:
            events.append(
                {
                    "person_id": None,
                    "event_type": "sound_interval",
                    "start_timestamp": start_row["timestamp"],
                    "end_timestamp": end_row["timestamp"],
                    "start_frame": int(start_row["frame"]),
                    "end_frame": int(end_row["frame"]),
                    "duration_sec": round(max(duration_sec, 0.0), 3),
                    "num_observations": int(len(ordered) - start_idx),
                    "label": label,
                    "confidence": round(peak_confidence, 4),
                }
            )

    return pd.DataFrame(events)


def build_summary(
    state_df: pd.DataFrame,
    events_df: pd.DataFrame,
    video_path: Path,
    mode: str,
    total_frames_read: int,
    sound_frame_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], str]:
    sound_frame_df = sound_frame_df if sound_frame_df is not None else pd.DataFrame()
    sound_alert_frames = (
        int(sound_frame_df["sound_alert"].fillna(False).astype(bool).sum())
        if not sound_frame_df.empty and "sound_alert" in sound_frame_df.columns
        else 0
    )

    analyzed_frames = 0
    analysis_start = ""
    analysis_end = ""
    if not state_df.empty:
        analyzed_frames = max(analyzed_frames, int(state_df["frame"].nunique()))
        analysis_start = str(state_df["timestamp"].min())
        analysis_end = str(state_df["timestamp"].max())
    if not sound_frame_df.empty:
        analyzed_frames = max(analyzed_frames, int(sound_frame_df["frame"].nunique()))
        sound_start = str(sound_frame_df["timestamp"].min())
        sound_end = str(sound_frame_df["timestamp"].max())
        analysis_start = min([value for value in [analysis_start, sound_start] if value], default="")
        analysis_end = max([value for value in [analysis_end, sound_end] if value], default="")

    def count_events(event_name: str) -> int:
        if events_df.empty or "event_type" not in events_df.columns:
            return 0
        return int((events_df["event_type"] == event_name).sum())

    if state_df.empty:
        summary_row = {
            "video_name": video_path.name,
            "mode": mode,
            "frames_analyzed": analyzed_frames,
            "frames_read": int(total_frames_read),
            "unique_people": 0,
            "pose_sit_frames": 0,
            "pose_stand_frames": 0,
            "pose_unknown_frames": 0,
            "rapid_motion_frames": 0,
            "distracted_frames": 0,
            "crowding_frames": 0,
            "fight_frames": 0,
            "horseplay_frames": 0,
            "sound_alert_frames": sound_alert_frames,
            "standing_intervals": 0,
            "rapid_motion_intervals": 0,
            "distracted_intervals": 0,
            "crowding_intervals": 0,
            "fight_intervals": 0,
            "horseplay_intervals": 0,
            "sound_intervals": count_events("sound_interval") if not events_df.empty else 0,
            "standing_too_long_intervals": 0,
            "analysis_start": analysis_start,
            "analysis_end": analysis_end,
        }
        text = (
            f"Video: {video_path.name}\n"
            f"Mode: {mode}\n"
            f"No person detections were recorded.\n"
            f"Sound alert frames: {sound_alert_frames}\n"
        )
        return pd.DataFrame([summary_row]), [summary_row], text

    pose_counts = state_df["stable_pose"].value_counts(dropna=False).to_dict()
    rapid_motion_frames = int(state_df["rapid_motion"].fillna(False).astype(bool).sum()) if "rapid_motion" in state_df.columns else 0
    distracted_frames = int(state_df["red_box_confirmed"].fillna(False).astype(bool).sum()) if "red_box_confirmed" in state_df.columns else 0
    crowding_frames = int(state_df["crowding"].fillna(False).astype(bool).sum()) if "crowding" in state_df.columns else 0
    fight_frames = int(state_df["fight_detected"].fillna(False).astype(bool).sum()) if "fight_detected" in state_df.columns else 0
    horseplay_frames = int(state_df["horseplay"].fillna(False).astype(bool).sum()) if "horseplay" in state_df.columns else 0

    summary_row = {
        "video_name": video_path.name,
        "mode": mode,
        "frames_analyzed": analyzed_frames,
        "frames_read": int(total_frames_read),
        "unique_people": int(state_df["person_id"].nunique()),
        "pose_sit_frames": int(pose_counts.get(POSE_SIT, 0)),
        "pose_stand_frames": int(pose_counts.get(POSE_STAND, 0)),
        "pose_unknown_frames": int(pose_counts.get(POSE_UNKNOWN, 0)),
        "rapid_motion_frames": rapid_motion_frames,
        "distracted_frames": distracted_frames,
        "crowding_frames": crowding_frames,
        "fight_frames": fight_frames,
        "horseplay_frames": horseplay_frames,
        "sound_alert_frames": sound_alert_frames,
        "standing_intervals": count_events("standing_interval"),
        "rapid_motion_intervals": count_events("rapid_motion_interval"),
        "distracted_intervals": count_events("distracted_interval"),
        "crowding_intervals": count_events("crowding_interval"),
        "fight_intervals": count_events("fight_interval"),
        "horseplay_intervals": count_events("horseplay_interval"),
        "sound_intervals": count_events("sound_interval"),
        "standing_too_long_intervals": count_events("standing_too_long"),
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
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
                "distracted_frames": int(group["red_box_confirmed"].fillna(False).astype(bool).sum()) if "red_box_confirmed" in group.columns else 0,
                "crowding_frames": int(group["crowding"].fillna(False).astype(bool).sum()) if "crowding" in group.columns else 0,
                "fight_frames": int(group["fight_detected"].fillna(False).astype(bool).sum()) if "fight_detected" in group.columns else 0,
                "horseplay_frames": int(group["horseplay"].fillna(False).astype(bool).sum()) if "horseplay" in group.columns else 0,
                "standing_intervals": int((person_events["event_type"] == "standing_interval").sum()) if not person_events.empty else 0,
                "rapid_motion_intervals": int((person_events["event_type"] == "rapid_motion_interval").sum()) if not person_events.empty else 0,
                "distracted_intervals": int((person_events["event_type"] == "distracted_interval").sum()) if not person_events.empty else 0,
                "crowding_intervals": int((person_events["event_type"] == "crowding_interval").sum()) if not person_events.empty else 0,
                "fight_intervals": int((person_events["event_type"] == "fight_interval").sum()) if not person_events.empty else 0,
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
        f"Distracted frames (3s+): {summary_row['distracted_frames']}",
        f"Crowding frames: {summary_row['crowding_frames']}",
        f"Fight frames: {summary_row['fight_frames']}",
        f"Horseplay frames: {summary_row['horseplay_frames']}",
        f"Sound alert frames: {summary_row['sound_alert_frames']}",
        f"Standing intervals: {summary_row['standing_intervals']}",
        f"Rapid-motion intervals: {summary_row['rapid_motion_intervals']}",
        f"Distracted intervals: {summary_row['distracted_intervals']}",
        f"Crowding intervals: {summary_row['crowding_intervals']}",
        f"Fight intervals: {summary_row['fight_intervals']}",
        f"Horseplay intervals: {summary_row['horseplay_intervals']}",
        f"Sound intervals: {summary_row['sound_intervals']}",
        f"Standing-too-long intervals: {summary_row['standing_too_long_intervals']}",
        "",
        "Per-person summary:",
    ]
    for row in per_person_rows:
        text_lines.append(
            f"ID {row['person_id']}: frames={row['frames']}, "
            f"sit={row['sit_frames']}, stand={row['stand_frames']}, unknown={row['unknown_frames']}, "
            f"rapid_motion={row['rapid_motion_frames']}, distracted={row['distracted_frames']}, "
            f"crowding={row['crowding_frames']}, "
            f"fight={row['fight_frames']}, horseplay={row['horseplay_frames']}"
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
    pose_classifier_bundle = None if args.count_only else load_classifier_bundle(args.pose_classifier, torch_device, torch)
    behavior_classifier_bundle = None if args.count_only else load_classifier_bundle(args.behavior_classifier, torch_device, torch)
    fight_classifier_bundle = None if args.count_only else load_fight_classifier_bundle(args.fight_classifier, torch_device, torch)
    sound_classifier_bundle = None if args.count_only else load_sound_classifier_bundle(args.sound_classifier, torch_device, torch)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sound_windows = (
        load_sound_windows_with_model(
            video_path=video_path,
            classifier_bundle=sound_classifier_bundle,
            window_sec=args.sound_window_sec,
            threshold=args.sound_threshold,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        if sound_classifier_bundle is not None
        else []
    )

    output_fps = fps if args.save_every_frame else max(fps / max(args.frame_skip, 1), 1.0)
    writer = create_video_writer(output_paths["video_silent"], output_fps, (frame_width, frame_height))

    report_rows: List[Dict[str, object]] = []
    sound_frame_rows: List[Dict[str, object]] = []
    pose_history: Dict[int, List[str]] = {}
    previous_centers: Dict[int, Tuple[float, float]] = {}
    behavior_clip_states: Dict[int, TrackClipState] = {}
    fight_clip_state = GlobalClipState(frames=[])
    last_states: Dict[int, DetectionState] = {}
    live_stats = LiveReportStats()
    red_box_tracker = None if args.count_only else RedBoxTracker()
    crowding_tracker = None if args.count_only else PairDurationTracker()
    horseplay_tracker = None if args.count_only else HorseplayTracker()
    last_fight_label: Optional[str] = None
    last_fight_confidence: Optional[float] = None
    last_fight_active = False
    sound_window_cursor = 0
    last_sound_label: Optional[str] = None
    last_sound_confidence: Optional[float] = None
    last_sound_active = False

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
            active_sound_window, sound_window_cursor = get_active_sound_window(sound_windows, sound_window_cursor, timestamp_sec)
            last_sound_label = active_sound_window.label if active_sound_window is not None else None
            last_sound_confidence = active_sound_window.confidence if active_sound_window is not None else None
            last_sound_active = bool(active_sound_window.alert) if active_sound_window is not None else False

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
                        pose_classifier_bundle=pose_classifier_bundle,
                        behavior_classifier_bundle=behavior_classifier_bundle,
                        behavior_clip_states=behavior_clip_states,
                        red_box_tracker=red_box_tracker,
                        crowding_tracker=crowding_tracker,
                        horseplay_tracker=horseplay_tracker,
                    )

                    if fight_classifier_bundle is not None:
                        last_fight_label, last_fight_confidence = classify_global_clip_with_model(
                            frame=frame,
                            classifier_bundle=fight_classifier_bundle,
                            threshold=args.fight_threshold,
                            clip_state=fight_clip_state,
                        )
                        last_fight_active = is_fight_label(last_fight_label)
                        fight_participants = (
                            localize_fight_participants(
                                states=last_states,
                                proximity_px=args.fight_participant_proximity_px,
                                movement_px=args.fight_participant_movement_px,
                            )
                            if last_fight_active
                            else {}
                        )
                        for person_id, state in last_states.items():
                            partner_ids = fight_participants.get(person_id, [])
                            state.fight_detected = len(partner_ids) > 0
                            state.fight_ids = ",".join(str(pid) for pid in partner_ids)
                            if state.fight_detected:
                                state.violation_type = append_violation(state.violation_type, "fight")

                    for row in frame_rows:
                        row["fight_label"] = last_fight_label
                        row["fight_confidence"] = safe_float(last_fight_confidence)
                        person_id = int(row["person_id"])
                        state = last_states.get(person_id)
                        row["fight_detected"] = bool(state.fight_detected) if state is not None else False
                        row["fight_ids"] = state.fight_ids if state is not None else ""
                        if bool(row["fight_detected"]):
                            row["violation_type"] = append_violation(str(row.get("violation_type", "")), "fight")
                if sound_classifier_bundle is not None:
                    sound_frame_rows.append(
                        {
                            "timestamp": format_timestamp(timestamp_sec),
                            "timestamp_sec": timestamp_sec,
                            "frame": frame_index,
                            "sound_label": last_sound_label,
                            "sound_confidence": safe_float(last_sound_confidence),
                            "sound_alert": last_sound_active,
                        }
                    )
                report_rows.extend(frame_rows)
                update_live_report_stats(
                    live_stats=live_stats,
                    frame_rows=frame_rows,
                    states=last_states,
                    count_only=args.count_only,
                    sound_alert=last_sound_active,
                )

            annotated = annotate_frame(
                frame=frame,
                states=last_states,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                mode=args.mode,
                live_stats=live_stats,
                count_only=args.count_only,
                fight_label=last_fight_label,
                fight_confidence=last_fight_confidence,
                fight_active=last_fight_active,
                sound_label=last_sound_label,
                sound_confidence=last_sound_confidence,
                sound_active=last_sound_active,
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

    silent_video_path = output_paths["video_silent"]
    final_video_path = output_paths["video"]
    ffmpeg_path = shutil.which(args.ffmpeg_bin) or args.ffmpeg_bin
    if silent_video_path.exists():
        try:
            mux_original_audio(
                video_without_audio=silent_video_path,
                source_video=video_path,
                output_path=final_video_path,
                ffmpeg_bin=ffmpeg_path,
            )
            silent_video_path.unlink(missing_ok=True)
        except (FileNotFoundError, RuntimeError) as exc:
            silent_video_path.replace(final_video_path)
            print(f"Warning: original audio was not preserved: {console_safe(exc)}")

    raw_df = pd.DataFrame(report_rows)
    sound_frame_df = pd.DataFrame(sound_frame_rows)
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
                "behavior_label",
                "behavior_confidence",
                "fight_label",
                "fight_confidence",
                "fight_detected",
                "fight_ids",
                "movement_px",
                "rapid_motion",
                "crowding",
                "crowding_ids",
                "crowding_duration_sec",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "red_box",
                "red_box_duration_sec",
                "red_box_confirmed",
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
        sound_events_df = build_sound_events(sound_frame_df, args.min_event_duration_sec)
        if not sound_events_df.empty:
            events_df = pd.concat([events_df, sound_events_df], ignore_index=True)
        summary_df, per_person_rows, summary_text = build_summary(
            state_df=state_df,
            events_df=events_df,
            video_path=video_path,
            mode=args.mode,
            total_frames_read=frame_index,
            sound_frame_df=sound_frame_df,
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
    if args.behavior_classifier and not args.count_only:
        print(f"Behavior classifier: {console_safe(args.behavior_classifier)}")
    if args.fight_classifier and not args.count_only:
        print(f"Fight classifier: {console_safe(args.fight_classifier)}")
    if args.sound_classifier and not args.count_only:
        print(f"Sound classifier: {console_safe(args.sound_classifier)}")
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
