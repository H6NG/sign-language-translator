"""
Video Transcriber for Sign Language
Analyzes video frames to convert sign language gestures to text.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
import os

# ================= CONFIGURATION =================
MODEL_FILE = 'gesture_model_pytorch.pth'
SCALER_FILE = 'gesture_scaler.pkl'
HAND_LANDMARKER_MODEL = 'hand_landmarker.task'

NUM_LANDMARKS = 21
NUM_COORDS = 3

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ================= NEURAL NETWORK =================
class SignLanguageClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super(SignLanguageClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden1)
        self.layer2 = torch.nn.Linear(hidden1, hidden2)
        self.layer3 = torch.nn.Linear(hidden2, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden1)
        self.bn2 = torch.nn.BatchNorm1d(hidden2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


# ================= MODEL LOADING =================
_ml_model = None
_scaler = None
_classes = None


def load_model():
    """Load the trained PyTorch model and scaler (singleton)."""
    global _ml_model, _scaler, _classes

    if _ml_model is not None:
        return _ml_model, _scaler, _classes

    print("🧠 Loading AI model for video transcription...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILE)
    scaler_path = os.path.join(script_dir, SCALER_FILE)

    print(f"  📁 Model path: {model_path}")
    print(f"  📁 Scaler path: {scaler_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        _ml_model = SignLanguageClassifier(
            checkpoint['input_size'],
            checkpoint['hidden1'],
            checkpoint['hidden2'],
            checkpoint['num_classes']
        )
        _ml_model.load_state_dict(checkpoint['model_state_dict'])
        _ml_model.eval()

        _classes = checkpoint['label_encoder_classes']
        _scaler = joblib.load(scaler_path)

        print(f"  ✅ Model loaded: {list(_classes)}")
        return _ml_model, _scaler, _classes
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        raise


def get_empty_hand():
    """Return zeros for undetected hand."""
    return [0.0] * (NUM_LANDMARKS * NUM_COORDS)


def extract_landmarks(hand_landmarks):
    """Convert landmarks to flat list of 63 features."""
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def predict_gesture(model, scaler, classes, left_features, right_features, mode='both'):
    """Make a prediction from landmark features."""
    features = np.array(left_features + right_features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        if mode == 'letters':
            valid_indices = [i for i, c in enumerate(classes) if c.isalpha()]
        elif mode == 'numbers':
            valid_indices = [i for i, c in enumerate(classes) if c.isdigit()]
        else:
            valid_indices = list(range(len(classes)))

        if not valid_indices:
            return "", 0.0

        filtered_probs = probabilities[0, valid_indices]
        best_idx = filtered_probs.argmax().item()
        confidence_value = filtered_probs[best_idx].item()
        predicted_class = classes[valid_indices[best_idx]]

    return str(predicted_class), float(confidence_value)


def transcribe_video_file(video_path: str, mode: str = 'both', confidence_threshold: float = 0.75, min_stable_duration: float = 0.25, return_segments: bool = False):
    """
    Transcribe a video file of sign language to text.

    Args:
        video_path: Path to the video file.
        mode: Prediction mode ('letters', 'numbers', or 'both').
        confidence_threshold: Minimum confidence to consider a prediction.
        min_stable_duration: Minimum duration (in seconds) a gesture must be held to be transcribed.

    Returns:
        If return_segments=False: transcribed text string.
        If return_segments=True: (text, segments) where segments is list of {start, end, char}.
    """
    print(f"🎥 Transcribing video: {video_path}")

    # Load model
    model, scaler, classes = load_model()

    # Get script directory for model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    landmarker_path = os.path.join(script_dir, HAND_LANDMARKER_MODEL)

    # Create HandLandmarker for VIDEO mode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_tracking_confidence=0.4,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.6
    )

    landmarker = HandLandmarker.create_from_options(options)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default fallback

    # Calculate required stable frames based on duration
    required_stable_frames = max(1, int(fps * min_stable_duration))
    max_grace_frames = max(1, int(fps * 0.15)) # 0.15s grace period

    print(f"  📊 Video FPS: {fps}")
    print(f"  ⏱️ Stability threshold: {min_stable_duration}s ({required_stable_frames} frames)")
    print(f"  🛡️ Grace period: 0.15s ({max_grace_frames} frames)")

    result_text = []
    segments = []  # {start, end, char} in seconds
    last_char = None
    stable_count = 0
    grace_counter = 0
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip frame to match training data (mirror mode)
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calculate timestamp in milliseconds
        timestamp_ms = int((frame_idx / fps) * 1000)

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"  ⚠️ Detection error at frame {frame_idx}: {e}")
            frame_idx += 1
            continue

        # Extract landmarks
        left_features = get_empty_hand()
        right_features = get_empty_hand()
        
        detected_this_frame = False
        predicted_char = None
        confidence = 0.0

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                features = extract_landmarks(hand_lms)
                if idx == 0:
                    left_features = features
                else:
                    right_features = features

            # Make prediction
            predicted_char, confidence = predict_gesture(model, scaler, classes, left_features, right_features, mode)
            
            if confidence >= confidence_threshold:
                detected_this_frame = True

        # Stability Logic with Grace Period
        if detected_this_frame:
            # We have a good prediction
            if predicted_char == last_char:
                # Continuing the same sign
                stable_count += 1
                grace_counter = 0 # Reset grace period
            else:
                # New sign detected (different from potential ongoing stable sign)
                # But is it just noise? Only switch if we've really lost the previous one
                # OR if we were in grace period, maybe the old one is gone.
                last_char = predicted_char
                stable_count = 1
                grace_counter = 0
            
            # Check if we've held it long enough
            if stable_count == required_stable_frames:
                # Only add if it's different from the last *output* character
                if not result_text or result_text[-1] != predicted_char:
                    result_text.append(predicted_char)
                    if return_segments:
                        t_start = (frame_idx - required_stable_frames) / fps if fps else 0
                        t_end = frame_idx / fps if fps else 0
                        segments.append({"start": round(t_start, 2), "end": round(t_end, 2), "char": predicted_char})
                    print(f"  ✅ Detected: '{predicted_char}' (conf: {confidence:.2f})")
        
        else:
            # No detection or low confidence
            if stable_count > 0:
                # We were tracking something. Enter grace period.
                grace_counter += 1
                if grace_counter > max_grace_frames:
                    # Grace period expired. Reset.
                    stable_count = 0
                    grace_counter = 0
                    last_char = None
            else:
                 # Wasn't tracking anything anyway
                 stable_count = 0 
                 last_char = None

        frame_idx += 1

    cap.release()
    landmarker.close()

    final_text = ''.join(result_text)
    print(f"📝 Transcription complete: '{final_text}'")
    if return_segments:
        return final_text, segments
    return final_text


def letter_sequence_to_words(letter_seq: str, dictionary: dict = None) -> list:
    """
    Convert raw letter sequence (e.g. 'HELLOWORLD') to list of words using dictionary.
    Uses greedy longest-match. Dictionary keys should be lowercase.
    """
    if not letter_seq or not letter_seq.strip():
        return []
    if dictionary is None:
        dict_path = os.path.join(os.path.dirname(__file__), 'words_dictionary.json')
        if os.path.exists(dict_path):
            import json
            with open(dict_path, 'r') as f:
                dictionary = json.load(f)
        else:
            return []
    text = letter_seq.strip().upper().replace(' ', '')
    words = []
    i = 0
    while i < len(text):
        found = None
        for wlen in range(min(12, len(text) - i), 0, -1):
            chunk = text[i:i + wlen].lower()
            if chunk in dictionary:
                found = chunk
                break
        if found:
            words.append(found)
            i += len(found)
        else:
            i += 1
    return words


if __name__ == '__main__':
    # Simple test
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = transcribe_video_file(video_path)
        print(f"\n=== RESULT ===\n{result}")
    else:
        print("Usage: python video_transcriber.py <video_path>")
