import argparse
import csv
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class CropSample:
    source_path: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual labeling tool for sit/stand person crops."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/pose_classifier/unlabeled"),
    )
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=Path("datasets/pose_classifier/labeled"),
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("datasets/pose_classifier/label_progress.csv"),
    )
    parser.add_argument("--window-width", type=int, default=1100)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy images to labeled folders instead of moving them.",
    )
    return parser.parse_args()


def collect_samples(source_dir: Path, progress_map: Dict[str, str]) -> List[CropSample]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    samples: List[CropSample] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(source_dir)
        if str(relative_path) in progress_map:
            continue
        samples.append(CropSample(source_path=path, relative_path=relative_path))
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
    def __init__(self, args: argparse.Namespace, samples: List[CropSample], progress_map: Dict[str, str]) -> None:
        self.args = args
        self.samples = samples
        self.progress_map = progress_map
        self.history: List[tuple[CropSample, str, Path]] = []
        self.index = 0
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.args.labeled_dir.mkdir(parents=True, exist_ok=True)
        (self.args.labeled_dir / "sit").mkdir(parents=True, exist_ok=True)
        (self.args.labeled_dir / "stand").mkdir(parents=True, exist_ok=True)
        (self.args.labeled_dir / "skip").mkdir(parents=True, exist_ok=True)
        self.args.progress_file.parent.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("Pose Crop Labeler")
        self.root.geometry(f"{self.args.window_width}x{self.args.window_height}")
        self.root.configure(bg="#202124")

        self.title_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.help_var = tk.StringVar(
            value="Keys: S=sit | W=stand | U=skip | Backspace=undo | Q=quit"
        )

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

        self.root.bind("s", lambda event: self.assign_label("sit"))
        self.root.bind("w", lambda event: self.assign_label("stand"))
        self.root.bind("u", lambda event: self.assign_label("skip"))
        self.root.bind("<BackSpace>", lambda event: self.undo())
        self.root.bind("q", lambda event: self.root.destroy())

        self.show_current_sample()

    def run(self) -> None:
        self.root.mainloop()

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

    def build_target_path(self, sample: CropSample, label: str) -> Path:
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
        sample, label, target_path = self.history.pop()
        if target_path.exists():
            sample.source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(target_path, sample.source_path)
        self.progress_map.pop(str(sample.relative_path), None)
        self.save_progress()
        self.index = max(0, self.index - 1)
        self.show_current_sample()

    def show_current_sample(self) -> None:
        if self.index >= len(self.samples):
            self.title_var.set("All images labeled")
            self.path_var.set(f"Progress saved to {self.args.progress_file}")
            self.image_label.configure(image="", text="Done", fg="white", font=("Segoe UI", 28, "bold"))
            self.current_photo = None
            return

        sample = self.samples[self.index]
        self.title_var.set(f"Image {self.index + 1} / {len(self.samples)}")
        self.path_var.set(str(sample.relative_path))

        with Image.open(sample.source_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            display = image.copy()
            display.thumbnail(
                (self.args.window_width - 80, self.args.window_height - 220),
                Image.Resampling.LANCZOS,
            )
            self.current_photo = ImageTk.PhotoImage(display)

        self.image_label.configure(image=self.current_photo, text="")


def main() -> None:
    args = parse_args()
    progress_map = load_progress(args.progress_file)
    samples = collect_samples(args.source_dir, progress_map)
    print(f"Pending images: {len(samples)}")
    app = LabelApp(args, samples, progress_map)
    app.run()


if __name__ == "__main__":
    main()
