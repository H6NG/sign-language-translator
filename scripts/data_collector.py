# data_collector.py
# ==================
# PURPOSE: Collect hand landmark data from MediaPipe and save to CSV for training
# 
# HOW IT WORKS:
# 1. Opens your camera (like visionTester.py)
# 2. Detects hand landmarks using MediaPipe
# 3. When you press a letter key (A-Z), it saves the current landmarks + label
# 4. All data goes to landmark_data.csv
#
# USAGE:
# 1. Run: python data_collector.py
# 2. Hold up a hand sign (e.g., ASL letter A)
# 3. Press 'a' on keyboard to save that sample with label 'A'
# 4. Repeat for each letter, collect 50+ samples per letter
# 5. Press 'q' to quit and save

import cv2# type: ignore 
import mediapipe as mp
import time
import csv
import os

# ================= CONFIGURATION =================
import os
model_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')
OUTPUT_FILE = 'landmark_data.csv'

# MediaPipe setup (same as visionTester.py)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Number of landmarks per hand
NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z

# Store latest detection result
latest_result = None
sample_count = {}  # Track samples per label

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Called by MediaPipe when detection is ready"""
    global latest_result
    latest_result = result

def generate_csv_header():
    """Create column names for the CSV"""
    columns = []
    for hand in ['left', 'right']:
        for i in range(NUM_LANDMARKS):
            columns.extend([f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}'])
    columns.append('label')
    return columns

def extract_landmarks(hand_landmarks):
    """
    Convert MediaPipe landmarks to a flat list of numbers.
    Each landmark has x, y, z coordinates (normalized 0-1).
    Returns 63 values (21 landmarks × 3 coords).
    """
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features

def get_empty_hand():
    """Return zeros for a hand that's not detected"""
    return [0.0] * (NUM_LANDMARKS * NUM_COORDS)

def save_sample(left_features, right_features, label, writer):
    """Save one training sample to CSV"""
    row = left_features + right_features + [label.upper()]
    writer.writerow(row)
    
    # Update count
    label_upper = label.upper()
    sample_count[label_upper] = sample_count.get(label_upper, 0) + 1
    print(f"  ✓ Saved '{label_upper}' - Total: {sample_count[label_upper]} samples")

def main():
    print("=" * 50)
    print("SIGN LANGUAGE DATA COLLECTOR")
    print("=" * 50)
    print("\nInstructions:")
    print("  1. Hold up a hand sign in front of camera")
    print("  2. Press the corresponding letter key (a-z)")
    print("  3. Repeat to collect 50+ samples per letter")
    print("  4. Press 'q' to quit and save")
    print("=" * 50 + "\n")
    
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
    
    # Check if CSV exists (append) or create new
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, 'a', newline='')
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(generate_csv_header())
        print(f"Created new file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing file: {OUTPUT_FILE}")
    
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)
            
            # Draw landmarks if detected
            status_text = "No hand detected"
            if latest_result and latest_result.hand_landmarks:
                status_text = f"Hand(s) detected: {len(latest_result.hand_landmarks)}"
                
                # Draw landmarks
                for hand_landmarks in latest_result.hand_landmarks:
                    h, w, _ = frame.shape
                    for lm in hand_landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Display status
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press A-Z to save, Q to quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show sample counts
            y_offset = 110
            for label, count in sorted(sample_count.items()):
                cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow('Data Collector', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif ord('a') <= key <= ord('z'):
                # Save the current landmarks
                if latest_result and latest_result.hand_landmarks:
                    label = chr(key)
                    
                    # Get landmarks for each hand
                    left_features = get_empty_hand()
                    right_features = get_empty_hand()
                    
                    for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                        features = extract_landmarks(hand_landmarks)
                        # Simple assignment (in production, use handedness detection)
                        if idx == 0:
                            left_features = features
                        else:
                            right_features = features
                    
                    save_sample(left_features, right_features, label, writer)
                else:
                    print("  ✗ No hand detected - sample not saved")
        
        cap.release()
        cv2.destroyAllWindows()
    
    csv_file.close()
    print("\n" + "=" * 50)
    print("DATA COLLECTION COMPLETE")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total samples: {sum(sample_count.values())}")
    print("=" * 50)

if __name__ == '__main__':
    main()
