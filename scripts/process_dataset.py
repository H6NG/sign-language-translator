# process_dataset.py
# ====================
# PURPOSE: Process ASL image dataset through MediaPipe to extract landmarks
# Creates landmark_data.csv for training
#
# USAGE:
#   python process_dataset.py

import cv2 # type: ignore
import mediapipe as mp
import csv
import os
from pathlib import Path

# ================= CONFIGURATION =================
DATASET_PATH = Path.home() / ".cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
OUTPUT_FILE = 'landmark_data.csv'
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')

# Which letters to process
# LETTERS_TO_PROCESS = ['A', 'B', 'C', 'D', 'E']  # Start with 5
LETTERS_TO_PROCESS = None  # Process ALL 26 letters

# How many images per letter
MAX_IMAGES_PER_LETTER = 500
# MAX_IMAGES_PER_LETTER = None  # Uncomment for all

NUM_LANDMARKS = 21
NUM_COORDS = 3

# ================= MEDIAPIPE TASKS API =================
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
    print("ASL DATASET PROCESSOR (MediaPipe Tasks API)")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    
    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        return
    
    letters = LETTERS_TO_PROCESS if LETTERS_TO_PROCESS else sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir() and len(d.name) == 1])
    print(f"Letters: {letters}")
    print(f"Max per letter: {MAX_IMAGES_PER_LETTER or 'all'}")
    print("=" * 60)
    
    # Setup - IMAGE mode for static images
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(generate_csv_header())
            
            total_processed = 0
            total_success = 0
            
            for letter in letters:
                letter_path = DATASET_PATH / letter
                if not letter_path.exists():
                    print(f"  Skipping {letter}")
                    continue
                
                image_files = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
                if MAX_IMAGES_PER_LETTER:
                    image_files = image_files[:MAX_IMAGES_PER_LETTER] # type: ignore
                
                print(f"\nProcessing '{letter}': {len(image_files)} images...")
                
                success_count = 0
                for i, img_path in enumerate(image_files):
                    total_processed += 1 # type: ignore
                    
                    if (i + 1) % 100 == 0:
                        print(f"  {i+1}/{len(image_files)}...")
                    
                    try:
                        # Read image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            continue
                        
                        # Convert and create MP Image
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        # Detect
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
                            
                            row = left_features + right_features + [letter] # type: ignore
                            writer.writerow(row)
                            success_count += 1 # type: ignore
                            total_success += 1 # type: ignore
                    except Exception as e:
                        pass  # Skip problematic images
                
                print(f"  ✓ {letter}: {success_count}/{len(image_files)} hands detected")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Processed: {total_processed}")
    print(f"  Detected: {total_success} ({total_success/total_processed*100:.1f}%)") # type: ignore
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext: python train_pytorch.py")


if __name__ == '__main__':
    main()
