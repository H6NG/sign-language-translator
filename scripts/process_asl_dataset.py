# process_asl_dataset.py
# ====================
# PURPOSE: Download and process ASL Alphabet Dataset (by debashishsau) through MediaPipe
# Creates landmark_data.csv for training
#
# USAGE:
#   python process_asl_dataset.py

import cv2 # type: ignore
import mediapipe as mp
import csv
import os
from pathlib import Path

# ================= CONFIGURATION =================
# Dataset will be auto-downloaded via kagglehub
KAGGLE_DATASET = "debashishsau/aslamerican-sign-language-aplhabet-dataset"
OUTPUT_FILE = 'landmark_data.csv'
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')

# Which letters to process (A-Z only, excluding SPACE, DELETE, NOTHING)
LETTERS_TO_PROCESS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# How many images per letter (None for all ~4000)
MAX_IMAGES_PER_LETTER = 20000

NUM_LANDMARKS = 21
NUM_COORDS = 3

# ================= MEDIAPIPE TASKS API =================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def download_dataset():
    """Download dataset using kagglehub"""
    try:
        import kagglehub # type: ignore
        print(f"Downloading dataset: {KAGGLE_DATASET}")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Dataset downloaded to: {path}")
        return Path(path)
    except ImportError:
        print("ERROR: kagglehub not installed. Install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        return None


def find_training_folder(base_path):
    """Find the training data folder in the downloaded dataset"""
    base = Path(base_path)
    
    # Try common folder names
    possible_names = [
        "Training Data",
        "training_data", 
        "train",
        "asl_alphabet_train",
        "Train"
    ]
    
    for name in possible_names:
        folder = base / name
        if folder.exists():
            return folder
    
    # Check if there's a nested structure
    for child in base.iterdir():
        if child.is_dir():
            for name in possible_names:
                folder = child / name
                if folder.exists():
                    return folder
            # Check if the child itself contains letter folders
            letter_folders = [d for d in child.iterdir() if d.is_dir() and len(d.name) == 1 and d.name.isalpha()]
            if len(letter_folders) >= 20:
                return child
    
    # Last resort: check if base itself has letter folders
    letter_folders = [d for d in base.iterdir() if d.is_dir() and len(d.name) == 1 and d.name.isalpha()]
    if len(letter_folders) >= 20:
        return base
    
    return None


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
    print("ASL ALPHABET DATASET PROCESSOR")
    print("Dataset: debashishsau/aslamerican-sign-language-aplhabet-dataset")
    print("=" * 60)
    
    # Download dataset
    dataset_base = download_dataset()
    if dataset_base is None:
        return
    
    # Find training folder
    dataset_path = find_training_folder(dataset_base)
    if dataset_path is None:
        print(f"\nERROR: Could not find training data folder in {dataset_base}")
        print("Contents:", list(dataset_base.iterdir())[:10]) # type: ignore
        return
    
    print(f"\nTraining data found at: {dataset_path}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: MediaPipe model not found: {MODEL_PATH}")
        print("Make sure hand_landmarker.task is in the current directory")
        return
    
    # Find available letters
    available_letters = sorted([
        d.name for d in dataset_path.iterdir() 
        if d.is_dir() and d.name in LETTERS_TO_PROCESS
    ])
    
    print(f"Letters to process: {available_letters}")
    print(f"Max per letter: {MAX_IMAGES_PER_LETTER or 'all'}")
    print("=" * 60)
    
    # Setup MediaPipe
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
            
            for letter in available_letters:
                letter_path = dataset_path / letter
                if not letter_path.exists():
                    print(f"  Skipping {letter} - folder not found")
                    continue
                
                # Find image files
                image_files = (
                    list(letter_path.glob("*.jpg")) + 
                    list(letter_path.glob("*.jpeg")) + 
                    list(letter_path.glob("*.png"))
                )
                
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
                        
                        # Convert to RGB and create MP Image
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        # Detect hands
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
                            total_success += 1
                    except Exception as e:
                        pass  # Skip problematic images
                
                print(f"  ✓ {letter}: {success_count}/{len(image_files)} hands detected")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Processed: {total_processed}")
    print(f"  Detected: {total_success} ({total_success/max(total_processed,1)*100:.1f}%)") # type: ignore
    print(f"  Saved to: {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext: python train_pytorch.py")


if __name__ == '__main__':
    main()
