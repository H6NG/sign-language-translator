"""
Enhanced Sign Language Mode - LSTM + rule-based HELLO detection.
Logic extracted from pytorch/predict.py for use in the Flask server.
"""

import os
import sys
import numpy as np
import torch
from collections import deque

# Add pytorch to path for SignModel import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pytorch'))
from model import SignModel  # noqa: E402

SIGNS = ["hello", "me", "thankyou", "no", "yes"]
ENHANCED_SEQ_LEN = 20


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def are_fingers_extended(hand_landmarks):
    extended_count = 0
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for t_idx, p_idx in zip(tips, pips):
        if hand_landmarks[t_idx].y < hand_landmarks[p_idx].y:
            extended_count += 1
    return extended_count >= 2


def to_vec(hand_result):
    """Convert hand result to 126-d vector (left 63 + right 63)."""
    left = np.zeros(63)
    right = np.zeros(63)
    if not hand_result.hand_landmarks:
        return np.concatenate([left, right])
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
        else:
            if i == 0:
                left = np.array(arr)
            else:
                right = np.array(arr)
    return np.concatenate([left, right])


class EnhancedProcessor:
    """Runs LSTM + HELLO rule-based detection on each frame."""

    def __init__(self, face_landmarker_path, sign_model_path):
        self.seq = deque(maxlen=ENHANCED_SEQ_LEN)
        self.prev_dist = None
        self.hello_counter = 0
        self.hello_cooldown = 0
        self.current_text = "..."
        self.face_detector = None
        self.lstm_model = None
        self.signs = SIGNS

        import mediapipe as mp
        self.mp = mp

        # Face detector (optional)
        if os.path.exists(face_landmarker_path):
            try:
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                BaseOptions = mp.tasks.BaseOptions
                face_options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_landmarker_path),
                    num_faces=5
                )
                self.face_detector = FaceLandmarker.create_from_options(face_options)
            except Exception as e:
                print(f"  ⚠️ Face landmarker failed: {e}")

        # LSTM model (optional)
        if os.path.exists(sign_model_path):
            try:
                self.lstm_model = SignModel(len(self.signs))
                self.lstm_model.load_state_dict(
                    torch.load(sign_model_path, map_location='cpu', weights_only=False)
                )
                self.lstm_model.eval()
            except Exception as e:
                print(f"  ⚠️ LSTM model failed: {e}")

    def process_frame(self, hand_result, face_result, frame_rgb):
        """Process one frame. Returns (current_text, frame_with_overlay optional)."""
        vec = to_vec(hand_result)
        self.seq.append(vec)

        if self.hello_cooldown > 0:
            self.current_text = "HELLO"
            self.hello_cooldown -= 1
        else:
            self.current_text = "..."

        # LSTM prediction
        if self.lstm_model and len(self.seq) == ENHANCED_SEQ_LEN:
            arr = np.array(self.seq)
            if np.sum(np.abs(arr)) >= 1.0:
                x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    try:
                        pred = self.lstm_model(x)
                        probs = torch.softmax(pred, dim=1)
                        if probs.max() > 0.7:
                            idx = pred.argmax(1).item()
                            predicted_sign = self.signs[idx]
                            if predicted_sign.lower() != "hello" and self.hello_cooldown == 0:
                                self.current_text = predicted_sign
                    except Exception:
                        pass

        # HELLO rule-based check (needs face + hand)
        is_hello_frame = False
        if hand_result.hand_landmarks and face_result and face_result.face_landmarks:
            hand_lm = hand_result.hand_landmarks[0]
            for face_lm in face_result.face_landmarks:
                index_tip = hand_lm[8]
                nose = face_lm[1]
                dist = get_dist(index_tip, nose)
                is_moving_away = self.prev_dist is not None and dist > self.prev_dist + 0.002
                is_above = index_tip.y < nose.y
                is_side = abs(index_tip.x - nose.x) > 0.05
                is_close_range = dist < 1.0
                if is_above and is_side and is_close_range and is_moving_away:
                    if are_fingers_extended(hand_lm):
                        is_hello_frame = True
                        self.prev_dist = dist
                        break
            if not is_hello_frame and face_result.face_landmarks:
                nose0 = face_result.face_landmarks[0][1]
                self.prev_dist = get_dist(hand_lm[8], nose0)

        if is_hello_frame:
            self.hello_counter += 1
        else:
            self.hello_counter = 0

        if self.hello_counter >= 5:
            self.current_text = "HELLO"
            self.hello_cooldown = 10

        return self.current_text

    def get_current_text(self):
        return self.current_text
