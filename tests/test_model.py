# test_model.py
# ==============
# PURPOSE: Test the trained model with live camera feed
# Shows the predicted letter on screen in real-time
#
# USAGE:
#   python test_model.py

import cv2 # type: ignore       
import mediapipe as mp
import time
import torch # type: ignore     
import numpy as np
import joblib # type: ignore

# ================= CONFIGURATION =================
import os
base_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
model_path = os.path.join(base_dir, 'hand_landmarker.task')
MODEL_FILE = os.path.join(base_dir, 'gesture_model_pytorch.pth')
SCALER_FILE = os.path.join(base_dir, 'gesture_scaler.pkl')

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

NUM_LANDMARKS = 21
NUM_COORDS = 3

# Store latest detection result
latest_result = None


def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def extract_landmarks(hand_landmarks):
    """Convert landmarks to flat list of 63 features"""
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def get_empty_hand():
    """Return zeros for undetected hand"""
    return [0.0] * (NUM_LANDMARKS * NUM_COORDS)


# ================= NEURAL NETWORK (must match training) =================
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


def load_model():
    """Load the trained model and scaler"""
    print("Loading model...")
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    
    # Recreate the model architecture
    model = SignLanguageClassifier(
        checkpoint['input_size'],
        checkpoint['hidden1'],
        checkpoint['hidden2'],
        checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Get class labels
    classes = checkpoint['label_encoder_classes']
    
    # Load scaler
    scaler = joblib.load(SCALER_FILE)
    
    print(f"  - Model loaded successfully")
    print(f"  - Classes: {classes}")
    
    return model, scaler, classes


def predict(model, scaler, classes, left_features, right_features, mode='both'):
    """Make a prediction from landmark features with mode filtering"""
    # Combine features
    features = np.array(left_features + right_features).reshape(1, -1)
    
    # Scale features (same as during training)
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Filter by mode
        if mode == 'letters':
            # Only consider A-Z
            valid_indices = [i for i, c in enumerate(classes) if c.isalpha()]
        elif mode == 'numbers':
            # Only consider 0-9
            valid_indices = [i for i, c in enumerate(classes) if c.isdigit()]
        else:
            # Both - all classes
            valid_indices = list(range(len(classes)))
        
        # Get best prediction among valid classes
        filtered_probs = probabilities[0, valid_indices]
        best_idx = filtered_probs.argmax().item()
        confidence_value = filtered_probs[best_idx].item()
        predicted_class = classes[valid_indices[best_idx]]
    
    return predicted_class, confidence_value


def main():
    print("=" * 50)
    print("SIGN LANGUAGE MODEL TESTER")
    print("=" * 50)
    
    # Load model
    model, scaler, classes = load_model()
    
    # Setup MediaPipe
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=result_callback,
        min_tracking_confidence=0.4,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.6
    )
    
    print("\nControls:")
    print("  L = Letters only (A-Z)")
    print("  N = Numbers only (0-9)")
    print("  B = Both")
    print("  Q = Quit\n")
    
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        prediction = ""
        confidence = 0.0
        mode = 'both'  # Start in both mode
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)
            
            # Draw landmarks and make prediction
            if latest_result and latest_result.hand_landmarks:
                # Get features
                left_features = get_empty_hand()
                right_features = get_empty_hand()
                
                for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                    features = extract_landmarks(hand_landmarks)
                    if idx == 0:
                        left_features = features
                    else:
                        right_features = features
                    
                    # Draw hand skeleton and landmarks (MediaPipe style)
                    h, w, _ = frame.shape
                    
                    # Hand connections for skeleton
                    HAND_CONNECTIONS = [
                        (0,1),(1,2),(2,3),(3,4),        # thumb
                        (0,5),(5,6),(6,7),(7,8),        # index
                        (5,9),(9,10),(10,11),(11,12),   # middle
                        (9,13),(13,14),(14,15),(15,16), # ring
                        (13,17),(17,18),(18,19),(19,20),# pinky
                        (0,17)                          # palm base
                    ]
                    TIP_IDS = [4, 8, 12, 16, 20]  # Fingertips
                    
                    # Draw skeleton lines (blue)
                    for c in HAND_CONNECTIONS:
                        x1 = int(hand_landmarks[c[0]].x * w)
                        y1 = int(hand_landmarks[c[0]].y * h)
                        x2 = int(hand_landmarks[c[1]].x * w)
                        y2 = int(hand_landmarks[c[1]].y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw landmarks (green joints, red fingertips)
                    for i, lm in enumerate(hand_landmarks):
                        x, y = int(lm.x * w), int(lm.y * h)
                        if i in TIP_IDS:
                            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red fingertips
                        else:
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green joints
                
                # Make prediction (filtered by mode)
                prediction, confidence = predict(model, scaler, classes, left_features, right_features, mode)
            
            # Display prediction
            if prediction:
                # Large letter display
                cv2.putText(frame, prediction, (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                
                # Confidence bar
                bar_width = int(confidence * 300)  # type: ignore
                cv2.rectangle(frame, (50, 200), (50 + bar_width, 230), (0, 255, 0), -1)
                cv2.rectangle(frame, (50, 200), (350, 230), (255, 255, 255), 2)
                cv2.putText(frame, f"{confidence:.0%}", (360, 225),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Show hand sign", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display current mode
            mode_text = {'letters': 'MODE: LETTERS (A-Z)', 'numbers': 'MODE: NUMBERS (0-9)', 'both': 'MODE: ALL (A-Z + 0-9)'}[mode]
            mode_color = {'letters': (0, 255, 255), 'numbers': (255, 165, 0), 'both': (255, 255, 255)}[mode]
            cv2.putText(frame, mode_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            cv2.putText(frame, "Press L/N/B to switch mode", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Sign Language Tester', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                mode = 'letters'
                print("Switched to LETTERS mode (A-Z)")
            elif key == ord('n'):
                mode = 'numbers'
                print("Switched to NUMBERS mode (0-9)")
            elif key == ord('b'):
                mode = 'both'
                print("Switched to BOTH mode (A-Z + 0-9)")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
