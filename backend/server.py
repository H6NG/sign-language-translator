"""
Flask server for Sign Language Translator GUI
Provides video streaming, AI predictions, and API endpoints for the React frontend
"""

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import time
import threading
import torch  # type: ignore
import joblib  # type: ignore
from flask import Flask, Response, jsonify, request  # type: ignore
from flask_cors import CORS  # type: ignore
from collections import deque
from dataparser import append_frame  # type: ignore
from detect import get_os  # type: ignore
from video_transcriber import transcribe_video_file, letter_sequence_to_words  # type: ignore
import tempfile
import os as os_module

from gestures.inference import GestureInference  # type: ignore

# Enhanced mode (LSTM + HELLO)
_enhanced_processor = None
_face_landmarker = None

# Gesture mode (BiLSTM Dynamic)
_gesture_engine = None
_gesture_face_landmarker = None

# LSTM full-word model for main tracker (hybrid prediction)
_lstm_model = None
_lstm_signs = ["hello", "me", "thankyou", "no", "yes"]
_lstm_seq = None


app = Flask(__name__)
CORS(app)

# ================= CONFIG =================
BUFFER_SIZE = 200
gesture_buffer = deque(maxlen=BUFFER_SIZE)
capture_count = 0
is_running = True

model_path = os_module.path.join(os_module.path.dirname(__file__), 'hand_landmarker.task')
MODEL_FILE = os_module.path.join(os_module.path.dirname(__file__), 'gesture_model_pytorch.pth')
SCALER_FILE = os_module.path.join(os_module.path.dirname(__file__), 'gesture_scaler.pkl')

# Current settings (configurable via API)
current_settings = {
    'num_hands': 2,
    'min_tracking_confidence': 0.4,
    'min_hand_detection_confidence': 0.4,
    'min_hand_presence_confidence': 0.6,
    'prediction_mode': 'both',  # 'letters', 'numbers', 'both', or 'gestures'
    'min_stable_duration': 0.25, # seconds
    'transcription_confidence': 0.75,
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand skeleton topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

TIP_IDS = [4, 8, 12, 16, 20]
NUM_LANDMARKS = 21
NUM_COORDS = 3

latest_result = None
current_landmarks = []  # For API access
current_prediction = {"letter": "", "confidence": 0.0}  # Store latest prediction
current_word_prediction = {"word": "", "confidence": 0.0}  # LSTM full-word
current_gesture_prediction = {"sign": "", "confidence": 0.0, "top3": []}  # Dynamic gesture
lock = threading.Lock()
landmarker = None
settings_changed = False


# =============== NEURAL NETWORK =================
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


# =============== MODEL LOADING =================
ml_model = None
scaler = None
classes = None
model_loaded = False

def load_ml_model():
    """Load the trained PyTorch model and scaler"""
    global ml_model, scaler, classes, model_loaded
    
    try:
        print("🧠 Loading AI model...")
        
        # Load model checkpoint
        checkpoint = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
        
        # Recreate the model architecture
        ml_model = SignLanguageClassifier(
            checkpoint['input_size'],
            checkpoint['hidden1'],
            checkpoint['hidden2'],
            checkpoint['num_classes']
        )
        ml_model.load_state_dict(checkpoint['model_state_dict'])
        ml_model.eval()  # Set to evaluation mode
        
        # Get class labels
        classes = checkpoint['label_encoder_classes']
        
        # Load scaler
        scaler = joblib.load(SCALER_FILE)
        
        model_loaded = True
        print(f"  ✅ Model loaded successfully")
        print(f"  📚 Classes: {list(classes)}")
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        model_loaded = False


def predict_gesture(left_features, right_features):
    """Make a prediction from landmark features - returns top 3 predictions"""
    global current_prediction
    
    if not model_loaded:
        return "", 0.0, []
    
    try:
        mode = current_settings.get('prediction_mode', 'both')
        
        # Combine features (left hand first, then right - matching training order)
        features = np.array(left_features + right_features).reshape(1, -1)
        
        # Scale features (same as during training)
        features_scaled = scaler.transform(features)  # type: ignore
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        with torch.no_grad():
            outputs = ml_model(features_tensor)  # type: ignore
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Filter by mode
            if mode == 'letters':
                valid_indices = [i for i, c in enumerate(classes) if c.isalpha()]  # type: ignore
            elif mode == 'numbers':
                valid_indices = [i for i, c in enumerate(classes) if c.isdigit()]  # type: ignore
            else:
                valid_indices = list(range(len(classes)))  # type: ignore
            
            if not valid_indices:
                return "", 0.0, []
            
            # Get filtered probabilities
            filtered_probs = probabilities[0, valid_indices]
            
            # Get top 3 predictions
            top_k = min(3, len(valid_indices))
            top_values, top_indices = torch.topk(filtered_probs, top_k)
            
            top_predictions = []
            for i in range(top_k):
                idx = top_indices[i].item()
                conf = top_values[i].item()
                letter = classes[valid_indices[idx]]  # type: ignore
                top_predictions.append({"letter": str(letter), "confidence": float(conf)})
            
            # Best prediction
            best_letter = top_predictions[0]["letter"] if top_predictions else ""
            best_confidence = top_predictions[0]["confidence"] if top_predictions else 0.0
        
        # Update current prediction with top 3
        with lock:
            current_prediction = {
                "letter": best_letter,
                "confidence": best_confidence,
                "top3": top_predictions
            }
        
        return best_letter, best_confidence, top_predictions
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "", 0.0, []


def get_empty_hand():
    """Return zeros for undetected hand"""
    return [0.0] * (NUM_LANDMARKS * NUM_COORDS)


# =============== CALLBACK =================
def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def create_landmarker():
    """Create a new HandLandmarker with current settings"""
    global landmarker
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=current_settings['num_hands'],
        result_callback=result_callback,
        min_tracking_confidence=current_settings['min_tracking_confidence'],
        min_hand_detection_confidence=current_settings['min_hand_detection_confidence'],
        min_hand_presence_confidence=current_settings['min_hand_presence_confidence']
    )
    
    return HandLandmarker.create_from_options(options)


# =============== ENHANCED MODE (LSTM + HELLO) =================
def _init_enhanced():
    """Lazily init enhanced mode components."""
    global _enhanced_processor, _face_landmarker
    if _enhanced_processor is not None:
        return True
    try:
        from enhanced import EnhancedProcessor  # type: ignore
        script_dir = os_module.path.dirname(os_module.path.abspath(__file__))
        face_path = os_module.path.join(script_dir, 'face_landmarker.task')
        sign_path = os_module.path.join(script_dir, '..', 'pytorch', 'sign_model.pth')
        _enhanced_processor = EnhancedProcessor(face_path, sign_path)
        # Face landmarker
        if os_module.path.exists(face_path):
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_path),
                num_faces=5
            )
            _face_landmarker = FaceLandmarker.create_from_options(opts)
        else:
            _face_landmarker = None
        print("  ✅ Enhanced mode (LSTM + HELLO) ready")
        return True
    except Exception as e:
        print(f"  ⚠️ Enhanced mode disabled: {e}")
        return False


def _load_lstm_model():
    """Load LSTM model for full-word prediction (optional)."""
    global _lstm_model, _lstm_seq
    if _lstm_model is not None:
        return
    try:
        import sys
        sys.path.insert(0, os_module.path.join(os_module.path.dirname(__file__), '..', 'pytorch'))
        from model import SignModel  # type: ignore
        sign_path = os_module.path.join(os_module.path.dirname(__file__), '..', 'pytorch', 'sign_model.pth')
        if os_module.path.exists(sign_path):
            _lstm_model = SignModel(len(_lstm_signs))
            _lstm_model.load_state_dict(torch.load(sign_path, map_location='cpu', weights_only=False))
            _lstm_model.eval()
            _lstm_seq = deque(maxlen=20)
            print("  ✅ LSTM word model loaded")
    except Exception as e:
        print(f"  ⚠️ LSTM model not loaded: {e}")


def _init_gesture_engine():
    """Load the dynamic gesture recognition engine and face landmarker."""
    global _gesture_engine, _gesture_face_landmarker
    
    script_dir = os_module.path.dirname(os_module.path.abspath(__file__))
    model_path = os_module.path.join(script_dir, 'gestures', 'gesture_lstm.pth')
    
    if os_module.path.exists(model_path):
        print("🧠 Loading dynamic gesture model...")
        _gesture_engine = GestureInference(
            model_path=model_path,
            confidence_threshold=0.75,
            cooldown_seconds=1.0,
            predict_interval=3
        )
        if _gesture_engine.is_loaded:
            print("  ✅ Dynamic gesture model loaded")
        else:
            _gesture_engine = None
    
    face_path = os_module.path.join(script_dir, 'face_landmarker.task')
    if os_module.path.exists(face_path):
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_path),
            num_faces=1
        )
        _gesture_face_landmarker = FaceLandmarker.create_from_options(opts)
        print("  ✅ Face landmarker loaded for gestures")


# Load AI model at startup
load_ml_model()
_load_lstm_model()
_init_gesture_engine()

# Create initial landmarker
landmarker = create_landmarker()

# Camera setup
def get_default_camera_index():
    return 1 if get_os() == "mac" else 0

cap = cv2.VideoCapture(get_default_camera_index())
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
current_camera_index = get_default_camera_index()


def generate_frames():
    """Generator for MJPEG video stream"""
    global latest_result, current_landmarks, is_running, landmarker, settings_changed, current_prediction
    
    while is_running and cap.isOpened():
        # Check if settings changed and recreate landmarker
        if settings_changed:
            with lock:
                if landmarker:
                    landmarker.close()
                landmarker = create_landmarker()
                settings_changed = False
        
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        timestamp = int(time.time() * 1000)
        
        try:
            landmarker.detect_async(mp_image, timestamp)
        except Exception as e:
            print(f"Detection error: {e}")
            continue
        
        # Process landmarks
        frame_landmarks = []
        left_features = get_empty_hand()
        right_features = get_empty_hand()
        
        # Face landmarks for gesture model
        face_lms = None
        if current_settings.get('prediction_mode') == 'gestures' and _gesture_face_landmarker:
            try:
                face_result = _gesture_face_landmarker.detect(mp_image)
                if face_result and face_result.face_landmarks:
                    face_lms = face_result.face_landmarks[0]
            except Exception:
                pass

        if latest_result and latest_result.hand_landmarks:
            # Use detection order (idx 0 = left, idx 1 = right) to match training data collection
            # This matches data_collector.py which assigned first detected hand to left_features
            for idx, hand_lms in enumerate(latest_result.hand_landmarks):
                features = []
                for lm in hand_lms:
                    features.extend([lm.x, lm.y, lm.z])
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Match data_collector.py: idx 0 = left, idx 1 = right
                if idx == 0:
                    left_features = features
                else:
                    right_features = features
            
            # Pad with zeros if only one hand detected
            if len(latest_result.hand_landmarks) == 1:
                frame_landmarks.extend([0.0] * 63)
            
            # Make prediction if model is loaded
            if current_settings.get('prediction_mode') == 'gestures':
                if _gesture_engine and _gesture_engine.is_loaded:
                    top3_raw = _gesture_engine.get_top_predictions(latest_result, face_lms, top_k=3)
                    pred = _gesture_engine.process_frame(latest_result, face_lms)
                    
                    with lock:
                        # Always update top3 so sidebar feels alive
                        current_gesture_prediction["top3"] = [{"letter": p[0], "confidence": p[1]} for p in top3_raw]
                        if pred:
                            current_gesture_prediction["sign"] = pred['sign']
                            current_gesture_prediction["confidence"] = pred['confidence']
            else:
                if model_loaded:
                    predict_gesture(left_features, right_features)
            
            # Draw on frame
            for hand_landmarks in latest_result.hand_landmarks:
                h, w, _ = frame.shape
                
                # Draw skeleton
                for c in HAND_CONNECTIONS:
                    start = hand_landmarks[c[0]]  # type: ignore
                    end = hand_landmarks[c[1]]  # type: ignore
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                
                # Draw points
                for i, lm in enumerate(hand_landmarks):
                    x, y = int(lm.x * w), int(lm.y * h)
                    color = (0, 255, 100)
                    radius = 5
                    if i in TIP_IDS:
                        color = (255, 100, 100)
                        radius = 8
                    cv2.circle(frame, (x, y), radius, color, -1)
            
            # Prediction is now shown via React AR overlay, no need to draw on frame
        else:
            # No hands - clear prediction
            with lock:
                current_prediction = {"letter": "", "confidence": 0.0}
                current_word_prediction = {"word": "", "confidence": 0.0}
                current_gesture_prediction["sign"] = ""
                current_gesture_prediction["confidence"] = 0.0
                if _gesture_engine:
                    _gesture_engine.reset()
        
        # Update buffer and current landmarks
        if len(frame_landmarks) == 126:
            with lock:
                gesture_buffer.append(frame_landmarks)
                current_landmarks = frame_landmarks.copy()
                # LSTM sequence for full-word prediction
                if _lstm_seq is not None and _lstm_model is not None:
                    _lstm_seq.append(frame_landmarks)
                    if len(_lstm_seq) == 20:
                        arr = np.array(_lstm_seq, dtype=np.float32)
                        if np.sum(np.abs(arr)) >= 1.0:
                            try:
                                x = torch.FloatTensor(arr).unsqueeze(0)
                                with torch.no_grad():
                                    pred = _lstm_model(x)
                                    probs = torch.nn.functional.softmax(pred, dim=1)
                                    conf, idx = probs.max(1)
                                    if conf.item() > 0.7:
                                        current_word_prediction["word"] = _lstm_signs[idx.item()]
                                        current_word_prediction["confidence"] = float(conf.item())
                                    else:
                                        current_word_prediction["word"] = ""
                                        current_word_prediction["confidence"] = 0.0
                            except Exception:
                                current_word_prediction["word"] = ""
                                current_word_prediction["confidence"] = 0.0
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.016)  # ~60fps cap


def generate_enhanced_frames():
    """MJPEG stream with LSTM + HELLO detection overlay."""
    global is_running, _enhanced_processor, _face_landmarker
    if not _init_enhanced() or _enhanced_processor is None:
        # Fallback: serve normal video with "Enhanced unavailable" overlay
        for chunk in generate_frames():
            yield chunk
        return
    # Create sync HandLandmarker for IMAGE mode
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_tracking_confidence=0.4,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.6
    )
    enhanced_hand = HandLandmarker.create_from_options(hand_opts)
    try:
        while is_running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = enhanced_hand.detect(mp_image)
            face_result = _face_landmarker.detect(mp_image) if _face_landmarker else None
            text = _enhanced_processor.process_frame(hand_result, face_result, rgb) # type: ignore
            # Draw overlay
            h, w, _ = frame.shape
            cv2.putText(frame, text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.016)
    finally:
        enhanced_hand.close()


# =============== API ROUTES =================

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/enhanced_feed')
def enhanced_feed():
    """MJPEG video stream with LSTM + HELLO overlay"""
    return Response(generate_enhanced_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/enhanced_output')
def enhanced_output():
    """Current enhanced mode prediction (HELLO, yes, no, etc.)"""
    _init_enhanced()
    text = "..."
    if _enhanced_processor is not None:
        text = _enhanced_processor.get_current_text()
    return jsonify({"success": True, "text": text})


@app.route('/launch_enhanced', methods=['POST'])
def launch_enhanced():
    """Initialize enhanced mode (called by frontend)"""
    ok = _init_enhanced()
    return jsonify({"success": ok, "message": "Enhanced mode ready" if ok else "Enhanced mode unavailable"})


@app.route('/prediction')
def prediction():
    """Get current AI prediction with top 3 and hand position"""
    with lock:
        # Get hand position for AR overlay (use wrist landmark - index 0)
        hand_position = None
        if len(current_landmarks) >= 3:
            # First hand wrist position (normalized 0-1)
            hand_position = {
                "x": current_landmarks[0],  # wrist x
                "y": current_landmarks[1]   # wrist y
            }
        
        return jsonify({
            "letter": current_prediction.get("letter", ""),
            "confidence": current_prediction.get("confidence", 0.0),
            "top3": current_prediction.get("top3", []),
            "word": current_word_prediction.get("word", ""),
            "word_confidence": current_word_prediction.get("confidence", 0.0),
            "gesture": current_gesture_prediction.get("sign", ""),
            "gesture_confidence": current_gesture_prediction.get("confidence", 0.0),
            "gesture_top3": current_gesture_prediction.get("top3", []),
            "hand_position": hand_position,
            "model_loaded": model_loaded,
            "mode": current_settings.get('prediction_mode', 'both')
        })


@app.route('/capture', methods=['POST'])
def capture():
    """Capture current gesture buffer to CSV"""
    global capture_count
    
    with lock:
        if len(gesture_buffer) == BUFFER_SIZE:
            flattened = []
            for landmark_frame in gesture_buffer:
                flattened.extend(landmark_frame)
            gesture_buffer.clear()
            
            # Save async
            def save_async(data):
                append_frame(data)
            
            threading.Thread(target=save_async, args=(flattened,), daemon=True).start()
            capture_count += 1  # type: ignore
            
            return jsonify({
                "success": True,
                "message": "Gesture captured!",
                "capture_count": capture_count
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Buffer not full: {len(gesture_buffer)}/{BUFFER_SIZE}",
                "buffer_level": len(gesture_buffer)
            })


@app.route('/quit', methods=['POST'])
def quit_app():
    """Stop the hand tracker"""
    global is_running
    is_running = False
    
    def shutdown():
        time.sleep(0.5)
        cap.release()
        if landmarker:
            landmarker.close()
    
    threading.Thread(target=shutdown, daemon=True).start()
    
    return jsonify({"success": True, "message": "Shutting down..."})


@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    """Transcribe an uploaded sign language video to text"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    # Save to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    try:
        video_file.save(temp_path)
        
        # Get settings
        mode = current_settings.get('prediction_mode', 'both')
        min_stable_duration = current_settings.get('min_stable_duration', 0.25)
        confidence_threshold = current_settings.get('transcription_confidence', 0.75)
        
        print(f"Using settings: duration={min_stable_duration}s, conf={confidence_threshold}")

        # Transcribe the video (with segments for playback sync)
        result_text, segments = transcribe_video_file(
            temp_path, 
            mode=mode,
            confidence_threshold=confidence_threshold,
            min_stable_duration=min_stable_duration,
            return_segments=True
        )
        # Word-level segmentation
        words = letter_sequence_to_words(result_text)
        return jsonify({
            "text": result_text,
            "words": words,
            "words_text": " ".join(words) if words else result_text,
            "segments": segments,
            "success": True
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Transcription error: {e}")
        return jsonify({"error": str(e), "text": ""}), 500
    finally:
        # Cleanup temp file
        os_module.close(temp_fd)
        if os_module.path.exists(temp_path):
            os_module.unlink(temp_path)


@app.route('/status')
def status():
    """Get current status"""
    with lock:
        return jsonify({
            "buffer_level": len(gesture_buffer),
            "buffer_size": BUFFER_SIZE,
            "capture_count": capture_count,
            "is_running": is_running,
            "has_hands": len(current_landmarks) == 126,
            "model_loaded": model_loaded
        })


@app.route('/feedback')
def hand_feedback():
    """Real-time hand position feedback (too far, tilted, out of frame)."""
    with lock:
        if len(current_landmarks) != 126:
            return jsonify({"status": "no_hands", "hint": "Show your hand to the camera"})
        lm = current_landmarks
        # Hand center (approx wrist average for both hands - use first hand)
        cx = sum(lm[i * 3] for i in range(21)) / 21
        cy = sum(lm[i * 3 + 1] for i in range(21)) / 21
        cz = sum(lm[i * 3 + 2] for i in range(21)) / 21
        # Hand "size" - max span in xy
        xs = [lm[i * 3] for i in range(21)]
        ys = [lm[i * 3 + 1] for i in range(21)]
        span_x = max(xs) - min(xs) if xs else 0
        span_y = max(ys) - min(ys) if ys else 0
        span = max(span_x, span_y)
        status = "ok"
        hint = ""
        if cx < 0.1 or cx > 0.9 or cy < 0.1 or cy > 0.9:
            status = "out_of_frame"
            hint = "Move hand toward center"
        elif span < 0.05:
            status = "too_far"
            hint = "Move hand closer to camera"
        elif span > 0.5:
            status = "too_close"
            hint = "Move hand slightly back"
        return jsonify({"status": status, "hint": hint, "hand_span": round(span, 3)})


@app.route('/landmarks')
def landmarks():
    """Get current landmark data for visualization"""
    with lock:
        if len(current_landmarks) == 126:
            # Extract fingertip positions (indices 4, 8, 12, 16, 20 for right hand)
            fingertips = {
                "thumb": {"x": current_landmarks[4*3], "y": current_landmarks[4*3+1]},
                "index": {"x": current_landmarks[8*3], "y": current_landmarks[8*3+1]},
                "middle": {"x": current_landmarks[12*3], "y": current_landmarks[12*3+1]},
                "ring": {"x": current_landmarks[16*3], "y": current_landmarks[16*3+1]},
                "pinky": {"x": current_landmarks[20*3], "y": current_landmarks[20*3+1]},
            }
            return jsonify({
                "has_data": True,
                "fingertips": fingertips,
                "all_landmarks": current_landmarks
            })
        else:
            return jsonify({"has_data": False})


@app.route('/cameras')
def list_cameras():
    """List available camera indices (0-5)."""
    cameras = []
    for i in range(6):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            cameras.append({"id": i, "name": f"Camera {i}"})
            test.release()
    return jsonify(cameras)


@app.route('/camera', methods=['POST'])
def set_camera():
    """Switch to a different camera index."""
    global cap, current_camera_index
    try:
        data = request.get_json()
        idx = int(data.get('index', 0))
        if idx < 0 or idx > 10:
            return jsonify({"success": False, "message": "Invalid camera index"}), 400
        with lock:
            old_cap = cap
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap = old_cap
                return jsonify({"success": False, "message": "Failed to open camera"}), 400
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            old_cap.release()
            current_camera_index = idx
        return jsonify({"success": True, "camera_index": idx})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current HandLandmarker settings"""
    out = dict(current_settings)
    out['camera_index'] = current_camera_index
    return jsonify(out)


@app.route('/settings', methods=['POST'])
def update_settings():
    """Update HandLandmarker settings"""
    global current_settings, settings_changed
    
    try:
        data = request.get_json()
        
        # Validate and update settings
        if 'num_hands' in data:
            num_hands = int(data['num_hands'])
            if num_hands in [1, 2]:
                current_settings['num_hands'] = num_hands
        
        if 'min_tracking_confidence' in data:
            val = float(data['min_tracking_confidence'])
            if 0 <= val <= 1:
                current_settings['min_tracking_confidence'] = val
        
        if 'min_hand_detection_confidence' in data:
            val = float(data['min_hand_detection_confidence'])
            if 0 <= val <= 1:
                current_settings['min_hand_detection_confidence'] = val
        
        if 'min_hand_presence_confidence' in data:
            val = float(data['min_hand_presence_confidence'])
            if 0 <= val <= 1:
                current_settings['min_hand_presence_confidence'] = val
        
        if 'prediction_mode' in data:
            mode = data['prediction_mode']
            if mode in ['letters', 'numbers', 'both', 'gestures']:
                current_settings['prediction_mode'] = mode

        if 'min_stable_duration' in data:
            val = float(data['min_stable_duration'])
            if 0.1 <= val <= 3.0: # Sensible limits
                current_settings['min_stable_duration'] = val
        
        if 'transcription_confidence' in data:
            val = float(data['transcription_confidence'])
            if 0 <= val <= 1.0:
                current_settings['transcription_confidence'] = val

        # Flag for landmarker recreation
        settings_changed = True
        
        return jsonify({
            "success": True,
            "message": "Settings updated",
            "settings": current_settings
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error updating settings: {str(e)}"
        }), 400


if __name__ == '__main__':
    print("🚀 Starting Sign Language Translator Server...")
    print("📹 Video feed: http://localhost:5000/video_feed")
    print("🎮 API endpoints: /capture, /quit, /status, /landmarks, /settings, /prediction")
    app.run(host='0.0.0.0', port=5001, threaded=True)
