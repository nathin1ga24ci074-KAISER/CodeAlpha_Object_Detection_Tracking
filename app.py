"""
Main Application Entry Point — CodeAlpha Task 4: Object Detection and Tracking.
Provides CLI interface for single-image detection verification, prerecorded video tracking,
interactive live webcam tracking, and desktop GUI dashboard launch.
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, Any, List, Union
import cv2
import numpy as np

from src.detector import YOLODetector, DetectionResult
from src.tracker import ObjectTracker, TrackedObject
from src.video_source import VideoSource
from src.utils import draw_bounding_box, draw_trajectory, draw_hud, ColorPalette, FPSMeter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_image_file(path_str: str) -> bool:
    """Returns True if the path points to an image file based on extension."""
    suffix = Path(path_str).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def is_video_file(path_str: str) -> bool:
    """Returns True if the path points to a video file based on extension."""
    suffix = Path(path_str).suffix.lower()
    return suffix in VIDEO_EXTENSIONS


def run_image_detection(
    image_path: str,
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    device: str = "cpu",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes real YOLO object detection on a single image file and saves annotated output.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    frame = cv2.imread(str(path.resolve()))
    if frame is None or frame.size == 0:
        raise ValueError(f"Failed to read image file or image is empty: {image_path}")

    detector = YOLODetector(
        model_name=model_name,
        confidence_threshold=conf_threshold,
        device=device
    )

    t0_load = time.perf_counter()
    detector.load_model()
    load_time_ms = (time.perf_counter() - t0_load) * 1000.0

    t0_infer = time.perf_counter()
    detections: List[DetectionResult] = detector.detect(frame)
    infer_time_ms = (time.perf_counter() - t0_infer) * 1000.0

    annotated = frame.copy()
    class_counts: Counter = Counter()

    for det in detections:
        class_counts[det.class_name] += 1
        color = ColorPalette.get_color(det.class_id)
        label = f"{det.class_name} {det.confidence:.2f}"
        draw_bounding_box(annotated, det.box, label, color=color, thickness=2)

    if save_path:
        out_p = Path(save_path)
    else:
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / f"detected_{path.stem}{path.suffix}"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_p.resolve()), annotated)
    if not success:
        raise IOError(f"Failed to write annotated image to: {out_p}")

    return {
        "success": True,
        "input_image": str(path.resolve()),
        "output_image": str(out_p.resolve()),
        "model_name": model_name,
        "weights_path": detector.get_model_info()["weights_path"],
        "device": detector.device,
        "confidence_threshold": conf_threshold,
        "total_detections": len(detections),
        "class_counts": dict(class_counts),
        "detections": [d.to_dict() for d in detections],
        "load_time_ms": round(load_time_ms, 2),
        "inference_time_ms": round(infer_time_ms, 2)
    }


def run_video_tracking(
    video_path: str,
    model_name: str = "yolo11n.pt",
    tracker_type: str = "bytetrack.yaml",
    conf_threshold: float = 0.25,
    device: str = "cpu",
    save_path: Optional[Union[str, bool]] = None,
    max_frames: Optional[int] = None,
    show_trajectories: bool = True
) -> Dict[str, Any]:
    """
    Executes real YOLO detection and ByteTrack association on a prerecorded video file.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vs = VideoSource(str(path.resolve()))
    vs.open()

    if vs.width == 0 or vs.height == 0:
        vs.release()
        raise ValueError(f"Unreadable video file or zero-dimension stream: {video_path}")

    tracker = ObjectTracker(
        tracker_type=tracker_type,
        model_name=model_name,
        confidence_threshold=conf_threshold,
        device=device
    )
    tracker.load_model()

    writer: Optional[cv2.VideoWriter] = None
    out_file_path: Optional[str] = None

    if save_path:
        if isinstance(save_path, str):
            out_p = Path(save_path)
        else:
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_p = out_dir / f"tracked_{path.stem}.mp4"

        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p.resolve()), fourcc, vs.fps, (vs.width, vs.height))
        out_file_path = str(out_p.resolve())

    fps_meter = FPSMeter()
    track_history: Dict[int, Dict[str, Any]] = {}
    frames_processed = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                break

            frames_processed += 1
            tracks = tracker.track(frame)
            current_fps = fps_meter.tick()

            annotated = frame.copy()
            active_class_counts: Counter = Counter()

            for t in tracks:
                active_class_counts[t.class_name] += 1
                color = ColorPalette.get_color(t.track_id)
                label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"

                if show_trajectories and len(t.trajectory) >= 2:
                    draw_trajectory(annotated, list(t.trajectory), color, thickness=2)

                draw_bounding_box(annotated, t.box, label, color=color, thickness=2)

                if t.track_id not in track_history:
                    track_history[t.track_id] = {
                        "track_id": t.track_id,
                        "class_name": t.class_name,
                        "first_frame": frames_processed,
                        "last_frame": frames_processed,
                        "frames_seen": 1,
                        "conf_sum": t.confidence
                    }
                else:
                    track_history[t.track_id]["last_frame"] = frames_processed
                    track_history[t.track_id]["frames_seen"] += 1
                    track_history[t.track_id]["conf_sum"] += t.confidence

            draw_hud(
                annotated,
                fps=current_fps,
                active_count=tracker.get_active_count(),
                total_count=tracker.get_total_count(),
                class_counts=dict(active_class_counts)
            )

            if writer is not None:
                writer.write(annotated)

            if max_frames and frames_processed >= max_frames:
                break

    finally:
        vs.release()
        if writer is not None:
            writer.release()

    total_duration = time.perf_counter() - t_start
    avg_fps = (frames_processed / total_duration) if total_duration > 0 else 0.0

    track_summaries = []
    for tid, info in track_history.items():
        avg_c = info["conf_sum"] / max(1, info["frames_seen"])
        track_summaries.append({
            "track_id": tid,
            "class_name": info["class_name"],
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "frames_seen": info["frames_seen"],
            "avg_confidence": round(avg_c, 4)
        })

    return {
        "success": True,
        "input_video": str(path.resolve()),
        "output_video": out_file_path,
        "tracker_type": tracker_type,
        "model_name": model_name,
        "frames_processed": frames_processed,
        "total_unique_tracks": tracker.get_total_count(),
        "total_duration_sec": round(total_duration, 2),
        "average_fps": round(avg_fps, 2),
        "tracks": sorted(track_summaries, key=lambda x: x["track_id"])
    }


def run_webcam_tracking(
    webcam_id: int = 0,
    model_name: str = "yolo11n.pt",
    tracker_type: str = "bytetrack.yaml",
    conf_threshold: float = 0.25,
    device: str = "cpu",
    save_path: Optional[Union[str, bool]] = None,
    max_frames: Optional[int] = None,
    show_trajectories: bool = True,
    headless: bool = False
) -> Dict[str, Any]:
    """
    Executes live YOLO detection and ByteTrack association on an interactive webcam stream.
    Displays OpenCV visualization window with 'Q' or 'ESC' exit handlers.
    """
    vs = VideoSource(webcam_id)
    try:
        vs.open()
    except Exception as err:
        raise IOError(f"Could not open webcam index {webcam_id}: {err}") from err

    tracker = ObjectTracker(
        tracker_type=tracker_type,
        model_name=model_name,
        confidence_threshold=conf_threshold,
        device=device
    )
    tracker.load_model()

    writer: Optional[cv2.VideoWriter] = None
    out_file_path: Optional[str] = None

    if save_path:
        if isinstance(save_path, str):
            out_p = Path(save_path)
        else:
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            out_p = out_dir / f"tracked_webcam_{webcam_id}_{timestamp}.mp4"

        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_val = vs.fps if vs.fps > 0 else 30.0
        writer = cv2.VideoWriter(str(out_p.resolve()), fourcc, fps_val, (vs.width, vs.height))
        out_file_path = str(out_p.resolve())

    fps_meter = FPSMeter()
    track_history: Dict[int, Dict[str, Any]] = {}
    frames_processed = 0
    t_start = time.perf_counter()
    window_name = "CodeAlpha — Live Webcam Tracking (Press 'q' or 'Esc' to exit)"

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                print("[WARNING] Frame grab failed from camera stream.", file=sys.stderr)
                break

            frames_processed += 1
            tracks = tracker.track(frame)
            current_fps = fps_meter.tick()

            annotated = frame.copy()
            active_class_counts: Counter = Counter()

            for t in tracks:
                active_class_counts[t.class_name] += 1
                color = ColorPalette.get_color(t.track_id)
                label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"

                if show_trajectories and len(t.trajectory) >= 2:
                    draw_trajectory(annotated, list(t.trajectory), color, thickness=2)

                draw_bounding_box(annotated, t.box, label, color=color, thickness=2)

                if t.track_id not in track_history:
                    track_history[t.track_id] = {
                        "track_id": t.track_id,
                        "class_name": t.class_name,
                        "first_frame": frames_processed,
                        "last_frame": frames_processed,
                        "frames_seen": 1,
                        "conf_sum": t.confidence
                    }
                else:
                    track_history[t.track_id]["last_frame"] = frames_processed
                    track_history[t.track_id]["frames_seen"] += 1
                    track_history[t.track_id]["conf_sum"] += t.confidence

            draw_hud(
                annotated,
                fps=current_fps,
                active_count=tracker.get_active_count(),
                total_count=tracker.get_total_count(),
                class_counts=dict(active_class_counts)
            )

            if writer is not None:
                writer.write(annotated)

            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    print("\nUser requested exit (quit key pressed).")
                    break

            if max_frames and frames_processed >= max_frames:
                break

    finally:
        vs.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    total_duration = time.perf_counter() - t_start
    avg_fps = (frames_processed / total_duration) if total_duration > 0 else 0.0

    track_summaries = []
    for tid, info in track_history.items():
        avg_c = info["conf_sum"] / max(1, info["frames_seen"])
        track_summaries.append({
            "track_id": tid,
            "class_name": info["class_name"],
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "frames_seen": info["frames_seen"],
            "avg_confidence": round(avg_c, 4)
        })

    return {
        "success": True,
        "webcam_id": webcam_id,
        "output_video": out_file_path,
        "tracker_type": tracker_type,
        "model_name": model_name,
        "frames_processed": frames_processed,
        "total_unique_tracks": tracker.get_total_count(),
        "total_duration_sec": round(total_duration, 2),
        "average_fps": round(avg_fps, 2),
        "tracks": sorted(track_summaries, key=lambda x: x["track_id"])
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeAlpha Task 4: Real-time Object Detection and Tracking with YOLO & ByteTrack"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: '0' for default webcam, or path to an image/video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Multi-object tracker configuration (default: bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.25)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device: 'cpu' or 'cuda' (default: cpu)"
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=True,
        default=None,
        help="Save output: pass flag alone for default in outputs/ or provide custom path"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of video frames to process"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV window (useful for automated testing)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the interactive desktop GUI dashboard"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # GUI Mode Launch
    if args.gui:
        from src.gui import launch_gui
        print("Launching CodeAlpha Object Detection & Tracking GUI Dashboard...")
        launch_gui()
        return

    source_str = args.source.strip()

    save_dest: Optional[Union[str, bool]] = None
    if args.save is True:
        save_dest = True
    elif isinstance(args.save, str):
        save_dest = args.save

    # Mode 1: Static Image Detection Test
    if is_image_file(source_str):
        print("=" * 64)
        print("CodeAlpha AI Internship — Task 4: Single-Image YOLO Detection")
        print("=" * 64)
        print(f"Input image: {source_str}")
        print(f"YOLO model:  {args.model}")
        print(f"Confidence:  {args.conf}")
        print(f"Device:      {args.device}")
        print("Running inference...")

        try:
            res = run_image_detection(
                image_path=source_str,
                model_name=args.model,
                conf_threshold=args.conf,
                device=args.device,
                save_path=save_dest if isinstance(save_dest, str) else None
            )
        except Exception as err:
            print(f"\n[ERROR] Detection failed: {err}", file=sys.stderr)
            sys.exit(1)

        print("\n" + "-" * 64)
        print(f"Inference Completed in {res['inference_time_ms']} ms")
        print(f"Total Detections Found: {res['total_detections']}")
        print(f"Class Distribution:     {res['class_counts']}")
        print("-" * 64)
        for i, d in enumerate(res["detections"], start=1):
            print(f"  {i}. {d['class_name']:<15} Conf: {d['confidence']*100:.1f}%  Box: {d['box']}")
        print("-" * 64)
        print(f"Annotated output saved to:\n  {res['output_image']}")
        print("=" * 64)
        return

    # Mode 2: Prerecorded Video File Tracking
    if is_video_file(source_str):
        print("=" * 64)
        print("CodeAlpha AI Internship — Task 4: Prerecorded Video Tracking")
        print("=" * 64)
        print(f"Input video:  {source_str}")
        print(f"YOLO model:   {args.model}")
        print(f"Tracker:      {args.tracker}")
        print(f"Confidence:   {args.conf}")
        print(f"Device:       {args.device}")
        if args.max_frames:
            print(f"Max frames:   {args.max_frames}")
        print("Processing video stream with ByteTrack...")

        try:
            res = run_video_tracking(
                video_path=source_str,
                model_name=args.model,
                tracker_type=args.tracker,
                conf_threshold=args.conf,
                device=args.device,
                save_path=save_dest,
                max_frames=args.max_frames
            )
        except Exception as err:
            print(f"\n[ERROR] Tracking failed: {err}", file=sys.stderr)
            sys.exit(1)

        print("\n" + "-" * 64)
        print(f"Video Tracking Complete: {res['frames_processed']} frames in {res['total_duration_sec']}s ({res['average_fps']} FPS)")
        print(f"Total Unique Track IDs:   {res['total_unique_tracks']}")
        print("-" * 64)
        print(f"{'Track ID':<10} {'Class':<15} {'Frames Seen':<15} {'Range (First-Last)':<20} {'Avg Conf':<10}")
        print("-" * 64)
        for t in res["tracks"]:
            f_range = f"Frame {t['first_frame']} -> {t['last_frame']}"
            print(f"#{t['track_id']:<9} {t['class_name']:<15} {t['frames_seen']:<15} {f_range:<20} {t['avg_confidence']*100:.1f}%")
        print("-" * 64)
        if res["output_video"]:
            print(f"Annotated video output saved to:\n  {res['output_video']}")
        print("=" * 64)
        return

    # Mode 3: Interactive Live Webcam Stream
    if source_str.isdigit():
        cam_id = int(source_str)
        print("=" * 64)
        print("CodeAlpha AI Internship — Task 4: Interactive Live Webcam Tracking")
        print("=" * 64)
        print(f"Webcam index: {cam_id}")
        print(f"YOLO model:   {args.model}")
        print(f"Tracker:      {args.tracker}")
        print(f"Confidence:   {args.conf}")
        print(f"Device:       {args.device}")
        if not args.headless:
            print("Display:      OpenCV live window (Press 'q' or 'Esc' to quit)")
        print("Opening webcam stream...")

        try:
            res = run_webcam_tracking(
                webcam_id=cam_id,
                model_name=args.model,
                tracker_type=args.tracker,
                conf_threshold=args.conf,
                device=args.device,
                save_path=save_dest,
                max_frames=args.max_frames,
                headless=args.headless
            )
        except Exception as err:
            print(f"\n[ERROR] Webcam tracking failed: {err}", file=sys.stderr)
            sys.exit(1)

        print("\n" + "-" * 64)
        print(f"Webcam Session Complete: {res['frames_processed']} frames in {res['total_duration_sec']}s ({res['average_fps']} FPS)")
        print(f"Total Unique Track IDs:   {res['total_unique_tracks']}")
        print("-" * 64)
        for t in res["tracks"]:
            f_range = f"Frame {t['first_frame']} -> {t['last_frame']}"
            print(f"#{t['track_id']:<9} {t['class_name']:<15} {t['frames_seen']:<15} {f_range:<20} {t['avg_confidence']*100:.1f}%")
        print("-" * 64)
        if res["output_video"]:
            print(f"Annotated webcam video saved to:\n  {res['output_video']}")
        print("=" * 64)
        return

    print(f"[ERROR] Unrecognized input source: {source_str}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
