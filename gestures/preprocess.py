"""
gestures/preprocess.py
Preprocessing utilities for dynamic gesture recognition.
Handles landmark normalization, temporal interpolation, motion deltas,
and facial expression feature extraction.
"""

import numpy as np# type: ignore
from scipy.interpolate import interp1d


# ============================================================
# HAND LANDMARK NORMALIZATION
# ============================================================

def landmarks_to_126(hand_result):
    """
    Convert MediaPipe hand result to a 126-dim vector.
    21 landmarks × 3 coords × 2 hands = 126.
    """
    left = np.zeros(63)
    right = np.zeros(63)

    if not hand_result.hand_landmarks:
        return np.concatenate([left, right])

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
        else:
            if i == 0:
                left = np.array(arr)
            else:
                right = np.array(arr)

    return np.concatenate([left, right])


def normalize_to_wrist(landmarks_63):
    """
    Make coordinates wrist-relative for one hand (63 values).
    Subtracts wrist position from all landmarks so hand position
    on screen doesn't affect the model.
    """
    landmarks = landmarks_63.reshape(21, 3)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    return landmarks.flatten()


def normalize_to_wrist_both(landmarks_126):
    """
    Apply wrist-relative normalization to both hands in a 126-dim vector.
    """
    left = normalize_to_wrist(landmarks_126[:63])
    right = normalize_to_wrist(landmarks_126[63:])
    return np.concatenate([left, right])


def normalize_scale(landmarks_63):
    """
    Scale-normalize one hand by the wrist-to-MCP9 distance.
    Makes the model invariant to hand size and camera distance.
    """
    landmarks = landmarks_63.reshape(21, 3)
    wrist = landmarks[0]
    mcp = landmarks[9]  # middle finger MCP
    scale = np.linalg.norm(mcp - wrist)
    if scale > 1e-6:
        landmarks = landmarks / scale
    return landmarks.flatten()


def normalize_scale_both(landmarks_126):
    """
    Apply scale normalization to both hands in a 126-dim vector.
    """
    left = normalize_scale(landmarks_126[:63])
    right = normalize_scale(landmarks_126[63:])
    return np.concatenate([left, right])


def normalize_hand_landmarks(landmarks_126):
    """
    Full normalization pipeline for a single frame's hand landmarks.
    1. Wrist-relative positioning
    2. Scale normalization
    Returns: 126-dim normalized vector
    """
    wrist_rel = normalize_to_wrist_both(landmarks_126)
    scaled = normalize_scale_both(wrist_rel)
    return scaled


# ============================================================
# FACIAL EXPRESSION FEATURES
# ============================================================

def extract_face_features(face_landmarks):
    """
    Extract 32 expression-relevant features from MediaPipe's 478 face landmarks.
    
    Features capture the ASL-grammatical facial signals:
    - Eyebrow raise/furrow (yes/no questions, wh-questions)
    - Eye openness (emphasis, squinting)
    - Mouth shape (mouth morphemes that distinguish signs)
    - Head pose (tilts, nods, shakes)
    - Cheek/jaw (jaw drop, cheek puff)
    
    Args:
        face_landmarks: list of 478 NormalizedLandmarks from MediaPipe FaceLandmarker,
                        or None if no face detected.
    Returns:
        np.array of shape (32,)
    """
    if face_landmarks is None or len(face_landmarks) == 0:
        return np.zeros(32)

    lm = face_landmarks
    features = []

    # --- Eyebrow raise (4 features) ---
    l_brow_y = np.mean([lm[70].y, lm[63].y, lm[105].y])
    l_eye_y = np.mean([lm[159].y, lm[145].y])
    features.append(l_eye_y - l_brow_y)  # positive = raised

    r_brow_y = np.mean([lm[300].y, lm[293].y, lm[334].y])
    r_eye_y = np.mean([lm[386].y, lm[374].y])
    features.append(r_eye_y - r_brow_y)

    features.append(abs(lm[55].x - lm[285].x))  # brow furrow (inner brow distance)
    features.append((l_eye_y - l_brow_y) - (r_eye_y - r_brow_y))  # asymmetry

    # --- Eye openness (4 features) ---
    l_eye_open = abs(lm[159].y - lm[145].y)
    r_eye_open = abs(lm[386].y - lm[374].y)
    features.extend([l_eye_open, r_eye_open])
    features.extend([
        l_eye_open / max(r_eye_open, 1e-6),
        r_eye_open / max(l_eye_open, 1e-6)
    ])

    # --- Mouth shape (12 features) ---
    mouth_open_v = abs(lm[13].y - lm[14].y)
    mouth_width = abs(lm[61].x - lm[291].x)
    features.extend([mouth_open_v, mouth_width])
    features.append(mouth_open_v / max(mouth_width, 1e-6))  # aspect ratio# type: ignore

    features.extend([lm[13].y, lm[14].y, lm[61].y, lm[291].y])  # lip positions

    mouth_cx = (lm[61].x + lm[291].x) / 2
    features.extend([lm[61].x - mouth_cx, lm[291].x - mouth_cx])  # pucker vs smile

    features.extend([lm[13].z, lm[14].z])  # lip protrusion

    mouth_cy = (lm[13].y + lm[14].y) / 2
    features.append((lm[61].y + lm[291].y) / 2 - mouth_cy)  # smile

    # --- Head pose (6 features) ---
    nose = lm[1]
    features.extend([nose.x, nose.y, nose.z])

    chin = lm[152]
    forehead = lm[10]
    features.extend([
        chin.x - forehead.x,  # lateral tilt
        chin.y - forehead.y,  # nod
        chin.z - forehead.z   # depth tilt
    ])

    # --- Cheek/jaw (6 features) ---
    features.append(abs(lm[152].y - lm[10].y))  # jaw opening
    features.extend([lm[234].x, lm[454].x])     # cheek puff
    features.extend([lm[98].y, lm[327].y])      # nose wrinkle
    features.append(lm[152].z)                    # chin forward

    return np.array(features[:32], dtype=np.float32)# type: ignore


# ============================================================
# FRAME VECTOR CONSTRUCTION
# ============================================================

def build_frame_vector(hand_result, face_landmarks=None):
    """
    Build a single frame's feature vector (158-dim).
    126 hand landmarks (normalized) + 32 face features.
    
    Args:
        hand_result: MediaPipe HandLandmarkerResult
        face_landmarks: list of 478 face landmarks (or None)
    Returns:
        np.array of shape (158,)
    """
    hand_vec = normalize_hand_landmarks(landmarks_to_126(hand_result))  # 126
    face_vec = extract_face_features(face_landmarks)                    # 32
    return np.concatenate([hand_vec, face_vec])


def build_frame_vector_from_raw(hand_126, face_landmarks=None):
    """
    Build frame vector from pre-extracted 126-dim hand vector.
    Used when hand landmarks have already been converted to a flat array.
    """
    hand_vec = normalize_hand_landmarks(hand_126)
    face_vec = extract_face_features(face_landmarks)
    return np.concatenate([hand_vec, face_vec])


# ============================================================
# TEMPORAL PROCESSING
# ============================================================

def normalize_sequence_length(sequence, target_length=40):
    """
    Resample a variable-length sequence to target_length frames
    using linear interpolation. Preserves the motion trajectory
    regardless of original speed.
    
    Args:
        sequence: np.array of shape (T, D) where T varies
        target_length: fixed number of output frames
    Returns:
        np.array of shape (target_length, D)
    """
    T = len(sequence)
    if T == 0:
        return np.zeros((target_length, sequence.shape[1] if sequence.ndim > 1 else 158))
    if T == 1:
        return np.tile(sequence, (target_length, 1))
    if T == target_length:
        return sequence

    original_indices = np.linspace(0, 1, T)
    target_indices = np.linspace(0, 1, target_length)

    interpolator = interp1d(original_indices, sequence, axis=0, kind='linear')
    return interpolator(target_indices)


def add_motion_deltas(sequence):
    """
    Append frame-to-frame differences as velocity features.
    Doubles the feature dimension: the model sees both position and velocity.
    
    Args:
        sequence: np.array of shape (T, D)
    Returns:
        np.array of shape (T, 2*D)
    """
    deltas = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, deltas], axis=1)


def preprocess_sequence(sequence, target_length=40, use_deltas=True):
    """
    Full preprocessing pipeline for a single gesture sequence.
    
    1. Normalize to fixed length via interpolation
    2. Optionally add motion deltas
    
    Args:
        sequence: np.array of shape (T, 158) — hand + face features
        target_length: frame count to resample to
        use_deltas: whether to append velocity features
    Returns:
        np.array of shape (target_length, 316) if use_deltas, else (target_length, 158)
    """
    # Resample to fixed length
    normalized = normalize_sequence_length(sequence, target_length)

    # Add velocity features
    if use_deltas:
        normalized = add_motion_deltas(normalized)

    return normalized.astype(np.float32)
