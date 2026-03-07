"""
gestures/collect_data.py
Upgraded data collector for dynamic gesture recognition.

Features:
    - Continuous recording (press START → perform gesture → press STOP)
    - Automatic motion-onset/offset segmentation
    - Multi-speed prompts (normal, slow, fast)
    - Saves variable-length .npy files (T, 158) — hand + face
    - Captures both hand and face landmarks simultaneously

Usage:
    python -m gestures.collect_data --signs hello thankyou yes no --samples 50
"""

import os
import sys
import argparse
import time
import numpy as np# type: ignore
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .preprocess import (  # type: ignore
    normalize_hand_landmarks,
    extract_face_features,
    build_frame_vector
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect gesture data")
    parser.add_argument("--signs", nargs="+", required=True,
                        help="List of sign names to collect")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples per sign")
    parser.add_argument("--output_dir", type=str, default="gestures/gesture_data",
                        help="Output directory for .npy files")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--min_frames", type=int, default=10,
                        help="Minimum frames for a valid sample")
    parser.add_argument("--max_frames", type=int, default=120,
                        help="Maximum frames per sample (auto-stop)")
    parser.add_argument("--auto_segment", action="store_true",
                        help="Auto-detect motion onset/offset")
    parser.add_argument("--use_face", action="store_true", default=True,
                        help="Also capture face landmarks")
    return parser.parse_args()


def setup_mediapipe(use_face=True):
    """Initialize MediaPipe hand and face landmarkers."""
    from mediapipe import Image, ImageFormat# type: ignore
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision# type: ignore

    # Hand landmarker
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=2
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # Face landmarker (optional)
    face_landmarker = None
    if use_face and os.path.exists('face_landmarker.task'):
        try:
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
                num_faces=1
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            print("  ✓ Face landmarker loaded")
        except Exception as e:
            print(f"  ⚠ Face landmarker failed: {e}")
    
    return hand_landmarker, face_landmarker, Image, ImageFormat


def extract_frame_features(frame_rgb, hand_landmarker, face_landmarker, Image, ImageFormat):
    """
    Extract hand + face features from a single RGB frame.
    Returns: (158-dim vector, bool has_hand)
    """
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    # Hand landmarks
    hand_result = hand_landmarker.detect(mp_image)

    has_hand = bool(hand_result.hand_landmarks)

    # Hand vector (126)
    left = np.zeros(63)
    right = np.zeros(63)
    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            arr = []
            for p in hand:
                arr += [p.x, p.y, p.z]
            if i < len(hand_result.handedness):
                handed = hand_result.handedness[i][0].category_name
                if handed == "Left":
                    left = np.array(arr)
                else:
                    right = np.array(arr)
    hand_126 = np.concatenate([left, right])

    # Face landmarks (32)
    face_vec = np.zeros(32)
    if face_landmarker is not None:
        face_result = face_landmarker.detect(mp_image)
        if face_result.face_landmarks:
            face_vec = extract_face_features(face_result.face_landmarks[0])

    # Combine
    return np.concatenate([hand_126, face_vec]).astype(np.float32), has_hand


def detect_motion(sequence, threshold=0.01):
    """
    Detect motion onset and offset in a sequence.
    Uses the velocity of hand landmarks to find when significant motion starts/stops.
    Returns: (start_idx, end_idx) or (0, len) if can't detect.
    """
    if len(sequence) < 5:
        return 0, len(sequence)

    # Compute per-frame velocity (L2 norm of delta in hand coords)
    hand_only = sequence[:, :126]# type: ignore
    deltas = np.diff(hand_only, axis=0)
    velocities = np.linalg.norm(deltas, axis=1)

    # Smooth velocities
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocities, kernel, mode='same')

    # Find motion onset (first frame above threshold)
    above = smoothed > threshold
    if not above.any():
        return 0, len(sequence)

    start = max(0, np.argmax(above) - 2)  # 2 frame buffer before onset

    # Find motion offset (last frame above threshold)
    end = min(len(sequence), len(above) - np.argmax(above[::-1]) + 2)

    return start, end


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Gesture Data Collector")
    print(f"{'='*60}")
    print(f"  Signs: {args.signs}")
    print(f"  Samples per sign: {args.samples}")
    print(f"  Output: {args.output_dir}")
    print(f"  Auto-segment: {args.auto_segment}")
    print(f"{'='*60}\n")

    # Setup MediaPipe
    hand_lm, face_lm, Image, ImageFormat = setup_mediapipe(args.use_face)

    # Setup camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Camera not found!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    speeds = ["normal", "slow", "fast"]

    for sign in args.signs:
        sign_dir = os.path.join(args.output_dir, sign)
        os.makedirs(sign_dir, exist_ok=True)

        # Count existing samples
        existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
        remaining = max(0, args.samples - existing)

        if remaining == 0:
            print(f"\n  '{sign}' already has {existing} samples — skipping")
            continue

        print(f"\n  === Collecting for: {sign} ({remaining} samples needed) ===")

        for sample_idx in range(existing, args.samples):
            recording = False
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Status display
                status_color = (0, 0, 255) if recording else (0, 255, 255)
                status_text = "● RECORDING" if recording else "Press SPACE to record"

                cv2.putText(frame, f"Sign: {sign}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Sample {sample_idx + 1}/{args.samples}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status_text,
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                if recording:
                    # Extract features
                    vec, has_hand = extract_frame_features(rgb, hand_lm, face_lm, Image, ImageFormat)
                    frames.append(vec)

                    cv2.putText(frame, f"Frames: {len(frames)}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                    # Auto-stop at max frames
                    if len(frames) >= args.max_frames:
                        recording = False
                        break

                cv2.imshow("Gesture Collector", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # SPACE
                    if not recording:
                        # Start recording
                        recording = True
                        frames = []
                    else:
                        # Stop recording
                        recording = False
                        break

                elif key == 27:  # ESC — quit
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n  Aborted by user.")
                    sys.exit(0)

                elif key == ord('s'):  # Skip this sample
                    frames = []
                    break

            # Process and save
            if len(frames) >= args.min_frames:
                sequence = np.array(frames, dtype=np.float32)  # (T, 158)

                # Auto-segment if enabled
                if args.auto_segment:
                    start, end = detect_motion(sequence)
                    sequence = sequence[start:end]

                # Validate
                if len(sequence) >= args.min_frames:
                    save_path = os.path.join(sign_dir, f"{sign}_{sample_idx}.npy")
                    np.save(save_path, sequence)
                    print(f"    Saved {save_path} — {len(sequence)} frames")
                else:
                    print(f"    ✗ Too short after segmentation ({len(sequence)} frames)")
                    # Retry this sample
                    continue
            else:
                if len(frames) > 0:
                    print(f"    ✗ Too short ({len(frames)} frames, need {args.min_frames})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done! Data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
