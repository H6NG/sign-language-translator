# pyre-ignore-all-errors
"""
gestures/test_live.py
Quick live test for the trained gesture model.
Opens your webcam and shows real-time predictions in a window.

Usage:
    python -m gestures.test_live --model gestures/gesture_lstm.pth
"""

import sys
import os
import argparse
import cv2# type: ignore
import numpy as np  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures.inference import GestureInference  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gestures/gesture_lstm.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.75,
                        help="Minimum confidence to display prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load inference engine
    processor = GestureInference(
        model_path=args.model,
        confidence_threshold=args.confidence,
        cooldown_seconds=1.0,
        predict_interval=3
    )

    if not processor.is_loaded:
        print(f"ERROR: Could not load model from {args.model}")
        print("Make sure you have trained the model first:")
        print("  python -m gestures.train --data_dir gestures/gesture_data")
        sys.exit(1)

    print(f"\n  Model loaded: {processor.num_signs} signs")
    print(f"  Signs: {processor.sign_names}")
    print(f"\n  Press Q to quit\n")

    # Setup MediaPipe
    try:
        from mediapipe import Image, ImageFormat  # type: ignore
        from mediapipe.tasks import python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=2
        )
        hand_lm = vision.HandLandmarker.create_from_options(hand_options)

        face_lm = None
        if os.path.exists('face_landmarker.task'):
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
                num_faces=1
            )
            face_lm = vision.FaceLandmarker.create_from_options(face_options)
            print("  Face landmarker loaded")

    except Exception as e:
        print(f"ERROR: MediaPipe setup failed: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_sign = None
    last_confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)# type: ignore

        hand_result = hand_lm.detect(mp_image)# type: ignore

        face_landmarks = None
        if face_lm:
            face_result = face_lm.detect(mp_image)# type: ignore
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]

        # Run inference
        prediction = processor.process_frame(hand_result, face_landmarks)
        if prediction:
            last_sign = prediction['sign']
            last_confidence = prediction['confidence']

        # Draw UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        if last_sign:
            conf_pct = int(last_confidence * 100)# type: ignore
            bar_width = int((frame.shape[1] - 40) * last_confidence)

            cv2.putText(frame, last_sign.upper(),# type: ignore
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(frame, f"{conf_pct}% confidence",
                        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

            # Confidence bar
            cv2.rectangle(frame, (20, 85), (frame.shape[1] - 20, 92), (60, 60, 60), -1)
            bar_color = (0, 220, 100) if last_confidence > 0.85 else (0, 180, 255)
            cv2.rectangle(frame, (20, 85), (20 + bar_width, 92), bar_color, -1)
        else:
            cv2.putText(frame, "Perform a gesture...",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)

        # Top-3 predictions sidebar
        top = processor.get_top_predictions(hand_result, face_landmarks, top_k=3)
        for i, (name, prob) in enumerate(top):
            y = 120 + i * 30
            cv2.putText(frame, f"{i+1}. {name}: {prob:.0%}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220) if i == 0 else (130, 130, 130), 1)

        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
