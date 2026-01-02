"""
cli.py
By: Natiq Ghafoor

Legacy CLI argument parser.

This project currently uses argparse directly in main.py, but this file is kept
as a simple wrapper for older runs and quick testing workflows.

Returned tuple:
(mode, input_path, exercise, debug, output_dir, save_video,
 csv_name, save_summary, calibration, simulate_failures, force_fps)
"""

from __future__ import annotations


def reader():
    """
    Parses command line input and returns config flags used by older main.py versions.
    """
    import argparse
    import os

    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    webcam_keywords = ["webcam", "cam"]
    folder_path = "exercise_modules/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to video file or webcam keyword.")
    parser.add_argument("--exercise", type=str, required=True, help="Exercise module (curl/squat/bench).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save artifacts.")
    parser.add_argument("--save-video", action="store_true", default=False, help="Save annotated video output.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output.")
    parser.add_argument("--csv-name", type=str, default="rep_features_output.csv", help="Custom CSV filename.")
    parser.add_argument("--save-summary", action="store_true", default=False, help="Save summary.json.")
    parser.add_argument("--calibration", type=str, default="calibration_config.json", help="Calibration JSON path.")
    parser.add_argument(
        "--simulate-failures",
        action="store_true",
        help="Enable jitter/occlusion/low-confidence simulations.",
    )
    parser.add_argument(
        "--force-fps",
        type=float,
        default=0.0,
        help="Override FPS if > 0 (useful for videos with missing metadata).",
    )

    args = parser.parse_args()

    if not os.path.exists(folder_path):
        raise ValueError(f"Exercise folder '{folder_path}' not found.")

    valid_exercises = [
        os.path.splitext(entry)[0]
        for entry in os.listdir(folder_path)
        if entry.endswith(".py") and entry != "__init__.py"
    ]

    if args.exercise not in valid_exercises:
        raise ValueError(f"Invalid exercise: '{args.exercise}'. Must be one of: {valid_exercises}")

    if args.input.lower() in webcam_keywords:
        mode = "webcam"
    elif args.input.lower().endswith(tuple(video_extensions)):
        mode = "video_file"
    else:
        raise ValueError("Unsupported input type: must be a video file or webcam keyword.")

    return (
        mode,
        args.input,
        args.exercise,
        args.debug,
        args.output_dir,
        args.save_video,
        args.csv_name,
        args.save_summary,
        args.calibration,
        args.simulate_failures,
        args.force_fps,
    )
