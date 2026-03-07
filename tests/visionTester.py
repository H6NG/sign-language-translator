import cv2# type: ignore # type: ignore 
import mediapipe as mp
import numpy as np# type: ignore 
import time
import threading
from mediapipe.tasks import python# type: ignore 
from mediapipe.tasks.python import vision
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from detect import get_os# type: ignore 
from dataparser import append_frame

# buffer for csv parsing
from collections import deque

#pytorch
import torch# type: ignore 
import torch.nn as nn
import numpy as np# type: ignore 


BUFFER_SIZE = 200
gesture_buffer = deque(maxlen=BUFFER_SIZE)


# ================= CONFIG =================
import os
model_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')
numberOfHands = 2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# ---- Hand skeleton topology (Tasks API compatible) ----
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17)                          # palm base
]

TIP_IDS = [4, 8, 12, 16, 20]
hand_data = []

latest_result = None
prev_key = -1  # For key debouncing


# =============== CALLBACK =================
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


# =============== OPTIONS ==================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=numberOfHands,
    result_callback=print_result,

    min_tracking_confidence=0.4,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.6
)


# =============== MAIN LOOP =================
with HandLandmarker.create_from_options(options) as landmarker:

    if (get_os() == "mac"):
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(0)

    # Optional HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. CAPTURE KEYPRESS ONCE PER LOOP
        key = cv2.waitKey(1) & 0xFF

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # ============ DRAWING ============
        # Always define it first
        current_frame_landmarks = []

        if latest_result and latest_result.hand_landmarks:# type: ignore 

            current_frame_landmarks = []
            hands_found = {"Right": None, "Left": None}

            for idx, hand_lms in enumerate(latest_result.hand_landmarks):# type: ignore 
                # Get the label (Right or Left)
                label = latest_result.handedness[idx][0].category_name# type: ignore 
                hands_found[label] = hand_lms

            # Now build the array in a FIXED order
            # If a hand is missing, we fill it with 0s so the array size is always 126
            for side in ["Right", "Left"]:
                if hands_found[side]:
                    for lm in hands_found[side]:# type: ignore 
                        current_frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    # Fill with 63 zeros if the hand isn't in frame
                    current_frame_landmarks.extend([0.0] * 63)

            for hand_landmarks in latest_result.hand_landmarks:# type: ignore 
                h, w, _ = frame.shape

                # ---- 1. Draw skeleton ----
                for c in HAND_CONNECTIONS:
                    start = hand_landmarks[c[0]]# type: ignore 
                    end = hand_landmarks[c[1]]

                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w),   int(end.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # ---- 2. Draw points ----
                for i, lm in enumerate(hand_landmarks):
                    x, y = int(lm.x * w), int(lm.y * h)

                    color = (0, 255, 0)
                    radius = 4

                    if i in TIP_IDS:
                        color = (0, 0, 255)   # fingertips red
                        radius = 7

                    cv2.circle(frame, (x, y), radius, color, -1)

        if len(current_frame_landmarks) == 126:
            gesture_buffer.append(current_frame_landmarks)

            # 3. SAVE LOGIC
            # 3. SAVE LOGIC
        # Only trigger on key DOWN (not while held)
        if key == ord('d') and prev_key != ord('d'):
            if len(gesture_buffer) == BUFFER_SIZE:
                # Copy data for thread-safe writing
                flattened = []
                for landmark_frame in gesture_buffer:
                    flattened.extend(landmark_frame)
                gesture_buffer.clear()
                
                # Save in background thread (no lag)
                def save_async(data):
                    append_frame(data)
                    print("✅ Gesture saved (buffered)")
                
                threading.Thread(target=save_async, args=(flattened,), daemon=True).start()
            else:
                print(
                    f"⚠️ Buffer not full: {len(gesture_buffer)}/{BUFFER_SIZE}")
        
        prev_key = key  # Track previous key for debouncing

        # Show
        cv2.imshow('Hand Tracking - Tasks API', frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
