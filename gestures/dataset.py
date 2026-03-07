"""
gestures/dataset.py
PyTorch Dataset for variable-length gesture sequences with data augmentation.

Loads .npy files from a directory structure:
    data_dir/
        hello/
            hello_0.npy   (T, 126) or (T, 158)
            hello_1.npy
            ...
        thankyou/
            thankyou_0.npy
            ...
"""

import os
import glob
import numpy as np# type: ignore
import torch
from torch.utils.data import Dataset# type: ignore

from .preprocess import (  # type: ignore
    normalize_hand_landmarks,
    normalize_sequence_length,
    add_motion_deltas
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_speed(sequence, speed_range=(0.4, 2.0)):
    """
    Randomly stretch or compress a gesture's timing.
    Simulates fast and slow signers.
    """
    speed = np.random.uniform(*speed_range)
    T = len(sequence)
    new_T = max(10, int(T / speed))  # faster speed = fewer frames

    if new_T == T:
        return sequence

    original_idx = np.linspace(0, 1, T)
    target_idx = np.linspace(0, 1, new_T)

    from scipy.interpolate import interp1d# type: ignore
    interpolator = interp1d(original_idx, sequence, axis=0, kind='linear')
    return interpolator(target_idx)


def augment_spatial_jitter(sequence, noise_std=0.005):
    """
    Add small Gaussian noise to x, y, z coordinates.
    Simulates slight camera position and hand position variations.
    """
    noise = np.random.randn(*sequence.shape) * noise_std
    return sequence + noise


def augment_rotation_z(sequence, max_angle_deg=15):
    """
    Randomly rotate hand landmarks around the z-axis.
    Handles slight hand tilt variations.
    Only affects hand landmarks (first 126 dims), not face features.
    """
    angle = np.random.uniform(-max_angle_deg, max_angle_deg)
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    rotated = sequence.copy()
    # Apply rotation to each hand's x, y coords
    for hand_offset in [0, 63]:  # left hand, right hand
        for point in range(21):
            x_idx = hand_offset + point * 3
            y_idx = hand_offset + point * 3 + 1
            if x_idx < min(126, sequence.shape[1]):
                x = rotated[:, x_idx]# type: ignore
                y = rotated[:, y_idx]
                rotated[:, x_idx] = x * cos_a - y * sin_a
                rotated[:, y_idx] = x * sin_a + y * cos_a

    return rotated


def augment_frame_dropout(sequence, drop_prob=0.1):
    """
    Randomly drop frames and re-interpolate.
    Simulates MediaPipe detection gaps.
    """
    T = len(sequence)
    if T < 5:
        return sequence

    keep_mask = np.random.random(T) > drop_prob
    # Always keep first and last frames
    keep_mask[0] = True
    keep_mask[-1] = True

    if keep_mask.sum() < 3:
        return sequence

    kept = sequence[keep_mask]
    # Re-interpolate back to original length
    return normalize_sequence_length(kept, T)


def augment_mirror(sequence):
    """
    Mirror (x-flip) the hand landmarks.
    Swaps left and right hands and flips x-coordinates.
    Only affects hand landmarks (first 126 dims), not face features.
    """
    mirrored = sequence.copy()
    D = sequence.shape[1]

    if D >= 126:
        # Swap left (0:63) and right (63:126) hands
        left = mirrored[:, :63].copy()
        right = mirrored[:, 63:126].copy()
        mirrored[:, :63] = right
        mirrored[:, 63:126] = left

        # Flip x-coordinates (every 3rd value starting at 0)
        for hand_offset in [0, 63]:
            for point in range(21):
                x_idx = hand_offset + point * 3
                mirrored[:, x_idx] = 1.0 - mirrored[:, x_idx]

    return mirrored


def augment_temporal_shift(sequence, max_shift=5):
    """
    Randomly shift the start of the gesture.
    Simulates imprecise segmentation boundaries.
    """
    T = len(sequence)
    if T < max_shift * 2:
        return sequence

    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift > 0:
        # Trim from start, pad end by repeating last frame
        shifted = np.concatenate([
            sequence[shift:],
            np.tile(sequence[-1:], (shift, 1))
        ])
    elif shift < 0:
        # Pad start by repeating first frame, trim end
        shifted = np.concatenate([
            np.tile(sequence[0:1], (-shift, 1)),
            sequence[:shift]
        ])
    else:
        shifted = sequence

    return shifted


# ============================================================
# DATASET
# ============================================================

class GestureSequenceDataset(Dataset):
    """
    Loads variable-length gesture sequences from .npy files and applies
    preprocessing + augmentation.
    
    Directory structure:
        data_dir/sign_name/*.npy  — each file is (T, D) where T varies
    
    Args:
        data_dir: path to root dataset directory
        target_length: resample all sequences to this many frames
        feature_dim: expected feature dimension per frame (126 for hands-only, 158 for hands+face)
        augment: whether to apply data augmentation
        use_deltas: whether to append motion delta features
    """

    def __init__(
        self,
        data_dir,
        target_length=40,
        feature_dim=158,
        augment=True,
        use_deltas=True
    ):
        self.target_length = target_length
        self.feature_dim = feature_dim
        self.augment = augment
        self.use_deltas = use_deltas

        self.samples = []     # list of (filepath, label_idx)
        self.sign_names = []  # ordered class names

        # Discover classes from directory structure
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        sign_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.sign_names = sign_dirs

        for label_idx, sign_name in enumerate(sign_dirs):
            sign_dir = os.path.join(data_dir, sign_name)
            npy_files = glob.glob(os.path.join(sign_dir, "*.npy"))

            if len(npy_files) == 0:
                print(f"  Warning: No .npy files in {sign_dir}")
                continue

            for f in npy_files:
                self.samples.append((f, label_idx))

        print(f"  Loaded {len(self.samples)} samples across {len(self.sign_names)} classes")
        for i, name in enumerate(self.sign_names):
            count = sum(1 for _, l in self.samples if l == i)
            print(f"    [{i}] {name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load variable-length sequence
        seq = np.load(filepath).astype(np.float32)  # (T, D)

        # If the data is 126-dim (hands-only), pad to 158 with zero face features
        if seq.shape[1] == 126:
            padding = np.zeros((seq.shape[0], 32), dtype=np.float32)
            seq = np.concatenate([seq, padding], axis=1)

        # Normalize hand landmarks per frame
        for i in range(len(seq)):
            seq[i, :126] = normalize_hand_landmarks(seq[i, :126])

        # --- Augmentation ---
        if self.augment:
            # Speed jitter (70% – 140% speed)
            if np.random.random() < 0.8:
                seq = augment_speed(seq, speed_range=(0.7, 1.4))

            # Spatial jitter
            if np.random.random() < 0.5:
                seq = augment_spatial_jitter(seq, noise_std=0.005)

            # Rotation
            if np.random.random() < 0.3:
                seq = augment_rotation_z(seq, max_angle_deg=15)

            # Frame dropout
            if np.random.random() < 0.3:
                seq = augment_frame_dropout(seq, drop_prob=0.1)

            # Mirror
            if np.random.random() < 0.5:
                seq = augment_mirror(seq)

            # Temporal shift
            if np.random.random() < 0.4:
                seq = augment_temporal_shift(seq, max_shift=5)

        # Resample to fixed length
        seq = normalize_sequence_length(seq, self.target_length)

        # Add motion deltas
        if self.use_deltas:
            seq = add_motion_deltas(seq)  # (target_length, 316)

        return torch.tensor(seq, dtype=torch.float32), label

    @property
    def num_classes(self):
        return len(self.sign_names)
