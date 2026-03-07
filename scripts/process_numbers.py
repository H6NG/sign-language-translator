# process_numbers.py
# ====================
# PURPOSE: Process ASL NUMBERS dataset through MediaPipe
# Adds to existing landmark_data.csv (letters + numbers)

import cv2 # type: ignore
import mediapipe as mp
import csv
import os
from pathlib import Path

# ================= CONFIGURATION =================
NUMBERS_DATASET_PATH = Path.home() / ".cache/kagglehub/datasets/muhammadkhalid/sign-language-for-numbers/versions/1/Sign Language for Numbers"
OUTPUT_FILE = 'landmark_data.csv'
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')

# Which numbers to process
NUMBERS_TO_PROCESS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# How many images per number
MAX_IMAGES_PER_NUMBER = 500

NUM_LANDMARKS = 21
NUM_COORDS = 3

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def generate_csv_header():
    columns = []
    for hand in ['left', 'right']:
        for i in range(NUM_LANDMARKS):
            columns.extend([f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}'])
    columns.append('label')
    return columns


def extract_landmarks(hand_landmarks):
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def get_empty_hand():
    return [0.0] * (NUM_LANDMARKS * NUM_COORDS)


def main():
    print("=" * 60)
    print("ASL NUMBERS DATASET PROCESSOR")
    print("=" * 60)
    print(f"Dataset: {NUMBERS_DATASET_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    
    if not NUMBERS_DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        return
    
    print(f"Numbers: {NUMBERS_TO_PROCESS}")
    print(f"Max per number: {MAX_IMAGES_PER_NUMBER}")
    print("=" * 60)
    
    # Check if CSV exists (append numbers to existing letters)
    file_exists = os.path.exists(OUTPUT_FILE)
    mode = 'a' if file_exists else 'w'
    
    if file_exists:
        print(f"\nAppending numbers to existing {OUTPUT_FILE}")
    else:
        print(f"\nCreating new {OUTPUT_FILE}")
    
    # Setup MediaPipe
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        with open(OUTPUT_FILE, mode, newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(generate_csv_header())
            
            total_processed = 0
            total_success = 0
            
            for number in NUMBERS_TO_PROCESS:
                number_path = NUMBERS_DATASET_PATH / number
                if not number_path.exists():
                    print(f"  Skipping {number}")
                    continue
                
                image_files = list(number_path.glob("*.jpg")) + list(number_path.glob("*.png")) + list(number_path.glob("*.jpeg"))
                if MAX_IMAGES_PER_NUMBER:
                    image_files = image_files[:MAX_IMAGES_PER_NUMBER] # type: ignore
                
                print(f"\nProcessing '{number}': {len(image_files)} images...")
                
                success_count = 0
                for i, img_path in enumerate(image_files):
                    total_processed += 1 # type: ignore
                    
                    if (i + 1) % 100 == 0:
                        print(f"  {i+1}/{len(image_files)}...")
                    
                    try:
                        image = cv2.imread(str(img_path))
                        if image is None:
                            continue
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        result = landmarker.detect(mp_image)
                        
                        if result.hand_landmarks:
                            left_features = get_empty_hand()
                            right_features = get_empty_hand()
                            
                            for idx, hand_lms in enumerate(result.hand_landmarks):
                                features = extract_landmarks(hand_lms)
                                if idx == 0:
                                    left_features = features
                                else:
                                    right_features = features
                            
                            row = left_features + right_features + [number] # type: ignore
                            writer.writerow(row)
                            success_count += 1 # type: ignore
                            total_success += 1
                    except Exception as e:
                        pass
                
                print(f"  ✓ {number}: {success_count}/{len(image_files)} hands detected")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Processed: {total_processed}")
    print(f"  Detected: {total_success} ({total_success/total_processed*100:.1f}%)") # type: ignore
    print(f"  Added to: {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext: python train_pytorch.py")


if __name__ == '__main__':
    main()
