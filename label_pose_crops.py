import argparse
import csv
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageOps, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ---- Label task configurations ----
LABEL_TASKS = {
    "pose": {
        "title": "Pose Crop Labeler (sit / stand)",
        "source_dir": Path("datasets/pose_classifier/unlabeled"),
        "labeled_dir": Path("datasets/pose_classifier/labeled"),
        "progress_file": Path("datasets/pose_classifier/label_progress.csv"),
        "media_type": "image",
        "labels": [
            {"name": "sit",   "key": "s", "display": "S=sit",   "color": "#4ec9b0"},
            {"name": "stand", "key": "w", "display": "W=stand", "color": "#dcdcaa"},
            {"name": "skip",  "key": "u", "display": "U=skip",  "color": "#808080"},
        ],
    },
    "behavior": {
        "title": "Behavior Crop Labeler (normal / distracted / active)",
        "source_dir": Path("datasets/behavior_classifier/unlabeled"),
        "labeled_dir": Path("datasets/behavior_classifier/labeled"),
        "progress_file": Path("datasets/behavior_classifier/label_progress.csv"),
        "media_type": "image",
        "labels": [
            {"name": "normal",     "key": "n", "display": "N=normal",     "color": "#4ec9b0"},
            {"name": "distracted", "key": "d", "display": "D=distracted", "color": "#dcdcaa"},
            {"name": "active",     "key": "a", "display": "A=active",     "color": "#f44747"},
            {"name": "skip",       "key": "u", "display": "U=skip",       "color": "#808080"},
        ],
    },
    "distracted": {
        "title": "Distracted Clip Labeler (focused / distracted)",
        "source_dir": Path("datasets/distracted_classifier/unlabeled"),
        "labeled_dir": Path("datasets/distracted_classifier/labeled"),
        "progress_file": Path("datasets/distracted_classifier/label_progress.csv"),
        "media_type": "video",
        "labels": [
            {"name": "focused",    "key": "f", "display": "F=focused",    "color": "#4ec9b0"},
            {"name": "distracted", "key": "d", "display": "D=distracted", "color": "#dcdcaa"},
            {"name": "skip",       "key": "u", "display": "U=skip",       "color": "#808080"},
        ],
    },
}


@dataclass
class MediaSample:
    source_path: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual labeling tool for person crops."
    )
    parser.add_argument(
        "--task",
        choices=list(LABEL_TASKS.keys()),
        default="pose",
        help="Labeling task: 'pose', 'behavior', or 'distracted'.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Source dir with unlabeled crops (default depends on --task).",
    )
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=None,
        help="Output dir for labeled crops (default depends on --task).",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="CSV progress file (default depends on --task).",
    )
    parser.add_argument("--window-width", type=int, default=1100)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--video-delay-ms", type=int, default=120, help="Playback delay for video clips.")
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files to labeled folders instead of moving them.",
    )
    args = parser.parse_args()

    task_cfg = LABEL_TASKS[args.task]
    if args.source_dir is None:
        args.source_dir = task_cfg["source_dir"]
    if args.labeled_dir is None:
        args.labeled_dir = task_cfg["labeled_dir"]
    if args.progress_file is None:
        args.progress_file = task_cfg["progress_file"]
    args.task_config = task_cfg
    return args


def collect_samples(source_dir: Path, progress_map: Dict[str, str], media_type: str) -> List[MediaSample]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    samples: List[MediaSample] = []
    allowed_extensions = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        relative_path = path.relative_to(source_dir)
        if str(relative_path) in progress_map:
            continue
        samples.append(MediaSample(source_path=path, relative_path=relative_path))
    return samples


def load_progress(progress_file: Path) -> Dict[str, str]:
    if not progress_file.exists():
        return {}
    progress: Dict[str, str] = {}
    with progress_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            progress[row["relative_path"]] = row["label"]
    return progress


class LabelApp:
    def __init__(self, args: argparse.Namespace, samples: List[MediaSample], progress_map: Dict[str, str]) -> None:
        self.args = args
        self.samples = samples
        self.progress_map = progress_map
        self.history: List[tuple[MediaSample, str, Path]] = []
        self.index = 0
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.current_video_capture: Optional[cv2.VideoCapture] = None
        self.video_job: Optional[str] = None

        self.args.labeled_dir.mkdir(parents=True, exist_ok=True)
        task_cfg = self.args.task_config
        self.media_type = task_cfg.get("media_type", "image")
        self.label_names = [l["name"] for l in task_cfg["labels"]]
        for label_name in self.label_names:
            (self.args.labeled_dir / label_name).mkdir(parents=True, exist_ok=True)
        self.args.progress_file.parent.mkdir(parents=True, exist_ok=True)

        help_parts = [f"{l['display']}" for l in task_cfg["labels"]]
        help_text = "Keys: " + " | ".join(help_parts) + " | Backspace=undo | Q=quit"

        self.root = tk.Tk()
        self.root.title(task_cfg["title"])
        self.root.geometry(f"{self.args.window_width}x{self.args.window_height}")
        self.root.configure(bg="#202124")

        self.title_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.help_var = tk.StringVar(value=help_text)

        self.header_label = tk.Label(
            self.root,
            textvariable=self.title_var,
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#202124",
        )
        self.header_label.pack(pady=(12, 4))

        self.path_label = tk.Label(
            self.root,
            textvariable=self.path_var,
            font=("Consolas", 10),
            fg="#d0d0d0",
            bg="#202124",
        )
        self.path_label.pack(pady=(0, 8))

        self.image_label = tk.Label(self.root, bg="#202124")
        self.image_label.pack(expand=True, fill="both", padx=16, pady=8)

        self.help_label = tk.Label(
            self.root,
            textvariable=self.help_var,
            font=("Segoe UI", 11),
            fg="#9cdcfe",
            bg="#202124",
        )
        self.help_label.pack(pady=(0, 12))

        for label_cfg in task_cfg["labels"]:
            key = label_cfg["key"]
            name = label_cfg["name"]
            self.root.bind(key, lambda event, lbl=name: self.assign_label(lbl))
        self.root.bind("<BackSpace>", lambda event: self.undo())
        self.root.bind("q", lambda event: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.show_current_sample()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.stop_video_playback()
        self.root.destroy()

    def save_progress(self) -> None:
        with self.args.progress_file.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["relative_path", "label", "target_path"])
            writer.writeheader()
            rows = []
            for relative_path, label in sorted(self.progress_map.items()):
                target_path = self.args.labeled_dir / label / Path(relative_path).name
                rows.append(
                    {
                        "relative_path": relative_path,
                        "label": label,
                        "target_path": str(target_path),
                    }
                )
            writer.writerows(rows)

    def build_target_path(self, sample: MediaSample, label: str) -> Path:
        target_dir = self.args.labeled_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        base_name = sample.source_path.name
        target_path = target_dir / base_name
        suffix_index = 1
        while target_path.exists():
            target_path = target_dir / f"{sample.source_path.stem}_{suffix_index}{sample.source_path.suffix}"
            suffix_index += 1
        return target_path

    def assign_label(self, label: str) -> None:
        if self.index >= len(self.samples):
            return
        self.stop_video_playback()
        sample = self.samples[self.index]
        target_path = self.build_target_path(sample, label)

        if self.args.copy_files:
            shutil.copy2(sample.source_path, target_path)
        else:
            shutil.move(sample.source_path, target_path)

        self.progress_map[str(sample.relative_path)] = label
        self.history.append((sample, label, target_path))
        self.save_progress()
        self.index += 1
        self.show_current_sample()

    def undo(self) -> None:
        if not self.history:
            return
        self.stop_video_playback()
        sample, label, target_path = self.history.pop()
        if target_path.exists():
            sample.source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(target_path, sample.source_path)
        self.progress_map.pop(str(sample.relative_path), None)
        self.save_progress()
        self.index = max(0, self.index - 1)
        self.show_current_sample()

    def stop_video_playback(self) -> None:
        if self.video_job is not None:
            self.root.after_cancel(self.video_job)
            self.video_job = None
        if self.current_video_capture is not None:
            self.current_video_capture.release()
            self.current_video_capture = None

    def render_image(self, image: Image.Image) -> None:
        display = image.copy()
        display.thumbnail(
            (self.args.window_width - 80, self.args.window_height - 220),
            Image.Resampling.LANCZOS,
        )
        self.current_photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.current_photo, text="")

    def display_video_frame(self) -> None:
        if self.current_video_capture is None:
            return

        ok, frame = self.current_video_capture.read()
        if not ok:
            self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.current_video_capture.read()
            if not ok:
                self.image_label.configure(
                    image="",
                    text="Unable to read clip",
                    fg="white",
                    font=("Segoe UI", 24, "bold"),
                )
                self.current_photo = None
                return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self.render_image(image)
        self.video_job = self.root.after(self.args.video_delay_ms, self.display_video_frame)

    def show_current_sample(self) -> None:
        self.stop_video_playback()
        if self.index >= len(self.samples):
            done_text = "All clips labeled" if self.media_type == "video" else "All images labeled"
            self.title_var.set(done_text)
            self.path_var.set(f"Progress saved to {self.args.progress_file}")
            self.image_label.configure(image="", text="Done", fg="white", font=("Segoe UI", 28, "bold"))
            self.current_photo = None
            return

        sample = self.samples[self.index]
        sample_label = "Clip" if self.media_type == "video" else "Image"
        self.title_var.set(f"{sample_label} {self.index + 1} / {len(self.samples)}")
        self.path_var.set(str(sample.relative_path))

        if self.media_type == "video":
            self.current_video_capture = cv2.VideoCapture(str(sample.source_path))
            if not self.current_video_capture.isOpened():
                self.image_label.configure(
                    image="",
                    text="Unable to open clip",
                    fg="white",
                    font=("Segoe UI", 24, "bold"),
                )
                self.current_photo = None
                return
            self.display_video_frame()
            return

        with Image.open(sample.source_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            self.render_image(image)


def main() -> None:
    args = parse_args()
    progress_map = load_progress(args.progress_file)
    samples = collect_samples(args.source_dir, progress_map, args.task_config.get("media_type", "image"))
    sample_label = "clips" if args.task_config.get("media_type", "image") == "video" else "images"
    print(f"Pending {sample_label}: {len(samples)}")
    app = LabelApp(args, samples, progress_map)
    app.run()


if __name__ == "__main__":
    main()
