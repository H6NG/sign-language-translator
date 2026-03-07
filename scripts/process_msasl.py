
import cv2 # type: ignore
import mediapipe as mp
import csv
import os
import json
import yt_dlp # type: ignore
import numpy as np
import time
from pathlib import Path

# ================= CONFIGURATION =================
MSASL_JSON_PATH = Path('c:/Users/aayus/Desktop/MS-ASL/MS-ASL/MSASL_train.json')
OUTPUT_FILE = 'msasl_landmarks.csv'
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'hand_landmarker.task')
TEMP_VIDEO_DIR = 'temp_videos'

# Target words to process (start small to verify)
TARGET_WORDS = [
    "hello", "nice", "teacher", "eat", "no", 
    "happy", "like", "want", "deaf", "school", 
    "finish", "mother", "father", "friend"
]

# Max videos per word to prevent huge downloads initially
MAX_VIDEOS_PER_WORD = 5 

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

def download_video(url, start_time, end_time, output_path):
    """
    Download a specific segment of a youtube video using yt-dlp
    """
    # yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
        # We can't easily download just a segment with yt-dlp native, 
        # so we download the whole thing and trim later or try to use external downloader args if ffmpeg is present.
        # For simplicity and robustness, we will try to download the whole video if it's not too long, 
        # but MS-ASL links are often specific timestamps. 
        # A better approach for speed is to use `download_sections` if supported or just download and open with opencv seek.
        # Let's try downloading the whole video for now as many might be short clips.
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        return False

def process_video(video_path, start_time, end_time, label, landmarker, writer):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    Error opening video file: {video_path}")
        return 0
    
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Default fallback
    
    # Calculate start and end frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_processed = 0
    frames_with_hands = 0
    
    current_frame = start_frame
    
    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1 # type: ignore
        frames_processed += 1
        
        # Convert to MediaPipe Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks
        # Using IMAGE mode for simplicity frame-by-frame, 
        # VIDEO mode requires timestamps which is better but requires careful timestamp management.
        # Given we are creating a training set of "poses", IMAGE mode is acceptable and less prone to stream errors here.
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            left_features = get_empty_hand()
            right_features = get_empty_hand()
            
            # Simple logic: Assign hands based on index or handedness if available.
            # MediaPipe tasks return handedness in result.handedness
            for idx, hand_lms in enumerate(result.hand_landmarks):
                # We need to check handedness to assign correctly if possible
                # handedness object has category_name 'Left' or 'Right'
                # Note: MediaPipe front camera mirror effect might swap them, but usually for videos it's objective.
                
                # Default fallback if handedness is ambiguous: 0 is Left, 1 is Right (matches previous script logic)
                features = extract_landmarks(hand_lms)
                
                if idx < len(result.handedness):
                    hand_label = result.handedness[idx][0].category_name
                    if hand_label == 'Left':
                        left_features = features
                    else:
                        right_features = features
                else:
                    # Fallback
                    if idx == 0: left_features = features
                    else: right_features = features
            
            row = left_features + right_features + [label] # type: ignore
            writer.writerow(row)
            frames_with_hands += 1
            
    cap.release()
    return frames_with_hands

def main():
    print("=" * 60)
    print("MS-ASL DATSET PROCESSOR")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model {MODEL_PATH} not found.")
        return

    # Create temp dir
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
    
    # Load JSON
    print(f"Loading metadata from {MSASL_JSON_PATH}...")
    try:
        with open(MSASL_JSON_PATH, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
        
    print(f"Total entries in JSON: {len(data)}")
    
    # Filter for target words
    filtered_data = []
    word_counts = {word: 0 for word in TARGET_WORDS}
    
    for entry in data:
        clean_text = entry.get('clean_text', '').lower() # clean_text seems most reliable
        # Also check 'text' field just in case
        text = entry.get('text', '').lower()
        
        # Check if this entry matches one of our target words
        target_match = None
        if clean_text in TARGET_WORDS:
            target_match = clean_text
        elif text in TARGET_WORDS:
            target_match = text
            
        if target_match:
            if word_counts[target_match] < MAX_VIDEOS_PER_WORD:
                filtered_data.append(entry)
                word_counts[target_match] += 1
    
    print(f"Filtered entries to process: {len(filtered_data)}")
    for word, count in word_counts.items():
        print(f"  - {word}: {count}")
    
    # Prepare MediaPipe
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    
    # Open CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(generate_csv_header())
        
        with HandLandmarker.create_from_options(options) as landmarker:
            for i, entry in enumerate(filtered_data):
                url = entry.get('url')
                start_time = entry.get('start_time', 0.0)
                end_time = entry.get('end_time', 0.0)
                label = entry.get('clean_text') or entry.get('text')
                
                # Fix URL format if needed
                if not url.startswith('http'):
                    url = 'https://' + url
                    
                print(f"[{i+1}/{len(filtered_data)}] Processing '{label}' from {url} ({start_time}-{end_time}s)")
                
                # Define temp filename
                video_filename = f"temp_{i}.mp4"
                video_path = os.path.join(TEMP_VIDEO_DIR, video_filename)
                
                # Download
                if download_video(url, start_time, end_time, video_path):
                    # Process
                    hands_found = process_video(video_path, start_time, end_time, label, landmarker, writer)
                    print(f"  -> Extracted {hands_found} frames with hands")
                    
                    # Cleanup
                    try:
                        os.remove(video_path)
                    except:
                        pass
                else:
                    print("  -> Download failed, skipping")
                    
    # Cleanup directory
    try:
        os.rmdir(TEMP_VIDEO_DIR)
    except:
        pass
        
    print("\nProcessing complete!")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
