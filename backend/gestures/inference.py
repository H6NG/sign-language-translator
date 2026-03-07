# pyre-ignore-all-errors
"""
gestures/inference.py
Live inference engine for dynamic gesture recognition.

Uses a multi-window sliding approach:
    - Continuously buffers landmark frames
    - Every N frames, extracts 3 window sizes (short/med/long)
    - Resamples each to fixed length, runs model
    - Takes highest-confidence prediction with cooldown

Integration:
    processor = GestureInference("gestures/gesture_lstm.pth")
    
    # In your frame loop:
    prediction = processor.process_frame(hand_result, face_landmarks)
    if prediction:
        print(f"Detected: {prediction['sign']} ({prediction['confidence']:.0%})")
"""

import os
import sys
import time
import numpy as np  # type: ignore
import torch
from collections import deque

from .preprocess import (  # type: ignore
    normalize_hand_landmarks,
    extract_face_features,
    normalize_sequence_length,
    add_motion_deltas
)
from .model import GestureModel, GestureModelLite  # type: ignore


class GestureInference:
    """
    Multi-window sliding gesture predictor.
    
    Runs alongside the existing static letter classifier — does NOT replace it.
    
    Args:
        model_path: path to trained .pth checkpoint
        confidence_threshold: minimum confidence to emit a prediction
        cooldown_seconds: time to suppress repeated predictions
        predict_interval: run prediction every N frames
    """

    def __init__(
        self,
        model_path,
        confidence_threshold=0.75,
        cooldown_seconds=1.5,
        predict_interval=5,
        device=None
    ):
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.predict_interval = predict_interval
        self.frame_count = 0
        self.last_prediction_time = 0
        self.last_prediction = None

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load model
        self.model = None
        self.sign_names = []
        self.target_length = 40
        self.use_deltas = True
        self.input_size = 316

        self._load_model(model_path)

        # Ring buffer: stores up to 90 frames (~3s at 30fps)
        self.buffer = deque(maxlen=90)

        # Multi-window sizes (frames)
        self.window_sizes = [30, 60, 90]  # short (~1s), med (~2s), long (~3s)

    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        if not os.path.exists(model_path):
            print(f"  ⚠ Gesture model not found at: {model_path}")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            self.sign_names = checkpoint.get('sign_names', [])
            num_classes = checkpoint.get('num_classes', len(self.sign_names))
            self.input_size = checkpoint.get('input_size', 316)
            self.target_length = checkpoint.get('target_length', 40)
            self.use_deltas = checkpoint.get('use_deltas', True)
            is_lite = checkpoint.get('lite', False)

            if is_lite:
                self.model = GestureModelLite(
                    num_classes=num_classes,
                    input_size=self.input_size
                )
            else:
                self.model = GestureModel(
                    num_classes=num_classes,
                    input_size=self.input_size
                )

            self.model.load_state_dict(checkpoint['model_state_dict'])  # type: ignore
            self.model.to(self.device)  # type: ignore
            self.model.eval()  # type: ignore

            print(f"  ✓ Gesture model loaded: {num_classes} signs, "
                  f"{'lite' if is_lite else 'full'}, "
                  f"val_acc={checkpoint.get('val_acc', '?')}")

        except Exception as e:
            print(f"  ✗ Failed to load gesture model: {e}")
            self.model = None

    def _build_frame_vector(self, hand_result, face_landmarks=None):
        """
        Build a 158-dim feature vector from raw MediaPipe results.
        126 hand (normalized) + 32 face.
        """
        # Hand landmarks → 126
        left = np.zeros(63)
        right = np.zeros(63)

        if hand_result and hand_result.hand_landmarks:
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
        hand_normalized = normalize_hand_landmarks(hand_126)

        # Face → 32
        face_vec = extract_face_features(face_landmarks)

        return np.concatenate([hand_normalized, face_vec]).astype(np.float32)

    def process_frame(self, hand_result, face_landmarks=None):
        """
        Process one frame from the camera feed.
        
        Call this every frame. It buffers the data and periodically
        runs gesture prediction.
        
        Args:
            hand_result: MediaPipe HandLandmarkerResult
            face_landmarks: list of 478 face landmarks (or None)
        
        Returns:
            dict with keys {'sign', 'confidence', 'window_size'} if a gesture
            is detected, or None if no gesture detected / still in cooldown.
        """
        if self.model is None:
            return None

        # Build and buffer frame vector
        vec = self._build_frame_vector(hand_result, face_landmarks)
        self.buffer.append(vec)
        self.frame_count += 1

        # Only predict every N frames to save compute
        if self.frame_count % self.predict_interval != 0:
            return None

        # Cooldown check
        now = time.time()
        if now - self.last_prediction_time < self.cooldown_seconds:
            return None

        # Need at least the shortest window
        if len(self.buffer) < min(self.window_sizes):
            return None

        # Multi-window prediction
        best_prediction = None
        best_confidence = 0

        for window_size in self.window_sizes:# type: ignore
            if len(self.buffer) < window_size:
                continue

            # Extract window from buffer
            window = list(self.buffer)[-window_size:]# type: ignore
            window_arr = np.array(window, dtype=np.float32)

            # Check if there's enough hand movement to be a gesture
            hand_only = window_arr[:, :126]
            movement = np.mean(np.abs(np.diff(hand_only, axis=0)))
            if movement < 0.002:  # negligible movement
                continue

            # Resample to target length
            resampled = normalize_sequence_length(window_arr, self.target_length)# type: ignore

            # Add deltas if configured
            if self.use_deltas:# type: ignore
                resampled = add_motion_deltas(resampled)

            # Model inference
            with torch.no_grad():
                x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)
                x = x.to(self.device)# type: ignore
                logits = self.model(x)# type: ignore
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = probs.max(dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()

            if confidence > best_confidence:
                best_confidence = confidence
                best_prediction = {
                    'sign': self.sign_names[pred_idx] if pred_idx < len(self.sign_names) else f"class_{pred_idx}",  # type: ignore
                    'confidence': confidence,
                    'window_size': window_size,
                    'class_idx': pred_idx
                }

        # Emit if above threshold
        if best_prediction and best_confidence >= self.confidence_threshold:# type: ignore
            self.last_prediction_time = now# type: ignore
            self.last_prediction = best_prediction# type: ignore
            return best_prediction

        return None

    def get_top_predictions(self, hand_result, face_landmarks=None, top_k=3):
        """
        Get top-k predictions for the current buffer state.
        Useful for debugging and visualization.
        Returns list of (sign_name, confidence) tuples.
        """
        if self.model is None or len(self.buffer) < min(self.window_sizes):
            return []

        # Use medium window size
        window_size = min(60, len(self.buffer))
        window = list(self.buffer)[-window_size:]# type: ignore
        window_arr = np.array(window, dtype=np.float32)

        resampled = normalize_sequence_length(window_arr, self.target_length)
        if self.use_deltas:
            resampled = add_motion_deltas(resampled)

        with torch.no_grad():
            x = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)# type: ignore
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            name = self.sign_names[idx] if idx < len(self.sign_names) else f"class_{idx}"  # type: ignore
            results.append((name, float(prob)))

        return results

    def reset(self):
        """Clear the buffer and reset state."""
        self.buffer.clear()
        self.frame_count = 0
        self.last_prediction_time = 0
        self.last_prediction = None

    @property
    def is_loaded(self):
        """Whether the model is loaded and ready."""
        return self.model is not None

    @property
    def num_signs(self):
        return len(self.sign_names)
