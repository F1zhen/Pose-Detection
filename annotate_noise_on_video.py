import argparse
import csv
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class NoiseWindow:
    start_sec: float
    end_sec: float
    db: float
    noisy: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate video when the classroom is noisy according to the audio track."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", type=Path, default=None, help="Output annotated video path.")
    parser.add_argument("--report-csv", type=Path, default=None, help="CSV with noise windows.")
    parser.add_argument("--threshold-db", type=float, default=55.0, help="Noise threshold in dB.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Audio analysis window size in seconds.")
    parser.add_argument("--banner-text", default="NOISY CLASSROOM", help="Overlay text for noisy windows.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg executable.")
    return parser.parse_args()


def ensure_video_exists(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FileNotFoundError(f"Unsupported video extension: {video_path.suffix}")


def default_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_noise_annotated.mp4")


def default_report_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_noise_windows.csv")


def extract_audio_to_wav(video_path: Path, wav_path: Path, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
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

    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while extracting audio from {video_path}.\n{completed.stderr.strip()}"
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


def load_audio_windows(wav_path: Path, window_sec: float, threshold_db: float) -> List[NoiseWindow]:
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
        return []

    samples_per_window = max(1, int(round(window_sec * sample_rate)))
    max_rms = float(np.iinfo(np.int16).max)
    windows: List[NoiseWindow] = []

    for start_index in range(0, audio.size, samples_per_window):
        chunk = audio[start_index : start_index + samples_per_window]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        dbfs = 20.0 * math.log10(max(rms, 1.0) / max_rms)
        estimated_db = dbfs + 100.0
        start_sec = start_index / sample_rate
        end_sec = min((start_index + chunk.size) / sample_rate, audio.size / sample_rate)
        windows.append(
            NoiseWindow(
                start_sec=start_sec,
                end_sec=end_sec,
                db=estimated_db,
                noisy=estimated_db >= threshold_db,
            )
        )

    return windows


def save_noise_report(report_path: Path, windows: List[NoiseWindow]) -> None:
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["start_sec", "end_sec", "db", "is_noisy"])
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "start_sec": round(window.start_sec, 3),
                    "end_sec": round(window.end_sec, 3),
                    "db": round(window.db, 2),
                    "is_noisy": int(window.noisy),
                }
            )


def find_noise_window(windows: List[NoiseWindow], timestamp_sec: float) -> Optional[NoiseWindow]:
    for window in windows:
        if window.start_sec <= timestamp_sec < window.end_sec:
            return window
    return windows[-1] if windows else None


def annotate_video(
    video_path: Path,
    output_path: Path,
    windows: List[NoiseWindow],
    banner_text: str,
    threshold_db: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_sec = frame_index / fps if fps else 0.0
            window = find_noise_window(windows, timestamp_sec)
            if window is not None:
                color = (0, 0, 255) if window.noisy else (0, 180, 0)
                label = banner_text if window.noisy else "SOUND LEVEL NORMAL"
                cv2.rectangle(frame, (20, 20), (width - 20, 110), color, thickness=3)
                cv2.putText(frame, label, (36, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"Audio: {window.db:.1f} dB | threshold: {threshold_db:.1f} dB",
                    (36, 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def main() -> None:
    args = parse_args()
    ensure_video_exists(args.input)

    output_path = args.output or default_output_path(args.input)
    report_path = args.report_csv or default_report_path(args.input)
    ffmpeg_path = shutil.which(args.ffmpeg_bin) or args.ffmpeg_bin

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        silent_video_path = Path(temp_dir) / "annotated_silent.mp4"
        extract_audio_to_wav(args.input, wav_path, ffmpeg_path)
        windows = load_audio_windows(
            wav_path=wav_path,
            window_sec=max(args.window_sec, 0.1),
            threshold_db=args.threshold_db,
        )
        annotate_video(
            video_path=args.input,
            output_path=silent_video_path,
            windows=windows,
            banner_text=args.banner_text,
            threshold_db=args.threshold_db,
        )
        mux_original_audio(silent_video_path, args.input, output_path, ffmpeg_path)

    save_noise_report(report_path, windows)

    noisy_count = sum(1 for window in windows if window.noisy)
    print(f"Input video: {args.input}")
    print(f"Annotated video: {output_path}")
    print(f"Noise report: {report_path}")
    print(f"Threshold: {args.threshold_db:.1f} dB")
    print(f"Noisy windows: {noisy_count} / {len(windows)}")


if __name__ == "__main__":
    main()
