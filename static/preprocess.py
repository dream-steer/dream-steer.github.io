from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "dreamsteer_video"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

# Crop rules are expressed as fractions of frame height: remove rows in [start, end).
CROP_RULES: dict[str, tuple[float, float]] = {
    "pickcup.gif": (0.75, 1.0),
    "pushbox.gif": (0.75, 1.0),
    "pushblock.gif": (0.75, 1.0),
    "foldcloth.gif": (0.75, 1.0),
    "washbottle.gif": (0.25, 0.5),
    "washpan.gif": (0.25, 0.5),
    "movearm.gif": (0.25, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop hardcoded row bands from GIFs and export them as MP4."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing GIF files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write MP4 files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output MP4 frame rate. Default: 10")
    return parser.parse_args()


def load_gif_frames(gif_path: Path) -> list[np.ndarray]:
    with Image.open(gif_path) as image:
        return [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]


def crop_frame(frame: np.ndarray, crop_rule: tuple[float, float] | None) -> np.ndarray:
    if crop_rule is None:
        return frame

    height = frame.shape[0]
    start = int(round(height * crop_rule[0]))
    end = int(round(height * crop_rule[1]))
    start = max(0, min(start, height))
    end = max(start, min(end, height))

    if start == end:
        return frame

    return np.concatenate((frame[:start], frame[end:]), axis=0)


def pad_frame_to_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write MP4 with no frames")

    frames = [pad_frame_to_even(frame) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                process.stdin.close()
                process.kill()
                raise ValueError("All frames must have the same shape before MP4 export")
            process.stdin.write(frame.astype(np.uint8).tobytes())
    except BrokenPipeError:
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        process.wait()
        raise RuntimeError(stderr.decode("utf-8", errors="replace")) from None

    process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def convert_gif(gif_path: Path, output_dir: Path, fps: int) -> None:
    frames = load_gif_frames(gif_path)
    crop_rule = CROP_RULES.get(gif_path.name)
    processed_frames = [crop_frame(frame, crop_rule) for frame in frames]

    output_path = output_dir / f"{gif_path.stem}.mp4"
    write_mp4(processed_frames, output_path, fps=fps)

    if crop_rule is None:
        print(f"{gif_path.name}: no crop rule -> {output_path.name}")
    else:
        print(
            f"{gif_path.name}: removed rows from {crop_rule[0]:.2f}h to {crop_rule[1]:.2f}h "
            f"-> {output_path.name}"
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    gif_paths = sorted(input_dir.glob("*.gif"))
    if not gif_paths:
        raise FileNotFoundError(f"No GIF files found in {input_dir}")

    for gif_path in gif_paths:
        convert_gif(gif_path=gif_path, output_dir=output_dir, fps=args.fps)


if __name__ == "__main__":
    main()
