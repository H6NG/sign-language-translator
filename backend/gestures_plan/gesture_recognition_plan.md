# Dynamic Gesture Recognition — Model & Pipeline Plan

## 1. Problem Statement

SignMate currently recognizes **static hand poses** (A–Z, 0–9) using a single-frame feedforward network, and has a basic **LSTM model** for 5 hard-coded dynamic signs ("hello", "me", "thankyou", "no", "yes"). Real ASL communication relies on **hundreds of dynamic gestures** — movements that unfold over time where the motion trajectory is the signal, not a single pose.

### Current Limitations

| Area | Current State | Problem |
|---|---|---|
| **Sequence length** | Fixed 20-frame window | Gestures vary from ~0.3s to ~2s. A fast signer clips early; a slow signer pads with noise |
| **Dataset size** | ~120 samples across 5 signs | Far too small to generalize. No signer diversity |
| **Normalization** | None on landmarks | Model is sensitive to hand distance from camera, hand size, and screen position |
| **Architecture** | 2-layer LSTM, 128 hidden | Too shallow for complex motion patterns; no attention mechanism |
| **Data collection** | Manual SPACE-trigger per sample | Tedious, slow, not scalable to 50+ signs |
| **Augmentation** | None | No speed variation, no spatial jitter, no mirroring |
| **Facial expressions** | Only used for rule-based HELLO | ASL uses eyebrow raises, head tilts, and mouth shapes grammatically — these are ignored by the model |

---

## 2. Recommended Architecture

### Option A — Bidirectional LSTM with Temporal Attention (Recommended Start)

This builds on what already exists in the codebase and is the fastest path to a working system:

```
Input: (batch, T, 158)  →  126 hand landmarks + 32 face features per frame
  ↓
Bidirectional LSTM (2 layers, 256 hidden)
  ↓
Temporal attention pooling (weighted sum over all timesteps)
  ↓
FC → 128 → ReLU → Dropout(0.4)
  ↓
FC → num_classes
```

The 158-dim input vector per frame is: **126** (hand landmarks, 21 pts × 3 coords × 2 hands) + **32** (face expression features — see Section 4d).

**Why bidirectional?** — Some gestures are defined by how they end (e.g., a closing fist), so looking backward matters.

**Why attention pooling instead of last-hidden?** — The current model takes `out[:, -1, :]` which discards everything except the final timestep. Attention lets the model learn which frames are most informative for each gesture.

### Option B — Transformer Encoder (Future Upgrade)

Once the dataset is large enough (500+ samples per class):

```
Input: (batch, T, 158) + positional encoding
  ↓
4-layer Transformer Encoder (d_model=128, 4 heads)
  ↓
CLS token pooling or mean pooling
  ↓
FC → num_classes
```

Transformers need more data but handle long-range dependencies better than LSTMs.

### Recommendation

**Start with Option A.** It reuses the existing LSTM infrastructure, trains on small datasets, and can be swapped to Option B later without changing data collection or preprocessing.

---

## 3. Handling Variable-Length Gestures

This is the core challenge. Different gestures take different amounts of time, and the same gesture performed by different people varies in speed. Three strategies, used together:

### 3a. Temporal Interpolation (Normalize to Fixed Frame Count)

Instead of a raw 20-frame window, **resample every gesture to a fixed number of frames** regardless of original duration:

```python
import numpy as np
from scipy.interpolate import interp1d

def normalize_sequence_length(sequence, target_length=40):
    """
    Resample a variable-length sequence to target_length frames.
    sequence: shape (T, 126) where T varies
    returns: shape (target_length, 126)
    """
    T = len(sequence)
    if T == target_length:
        return sequence
    
    original_indices = np.linspace(0, 1, T)
    target_indices = np.linspace(0, 1, target_length)
    
    interpolator = interp1d(original_indices, sequence, axis=0, kind='linear')
    return interpolator(target_indices)
```

**Why 40 frames?** At 30fps, this covers gestures from ~0.3s (fast) to ~3s (slow) after interpolation. The interpolation preserves the motion trajectory regardless of original speed.

### 3b. Sliding Window with Overlap at Inference Time

During **live prediction**, don't wait for a complete gesture. Use a **sliding window**:

```
Buffer: continuously append landmark vectors
Every N frames (e.g., every 5 frames):
  1. Take the last W frames from the buffer (e.g., W = 60)
  2. Resample to 40 frames via temporal interpolation
  3. Run model prediction
  4. If confidence > threshold → output the sign
  5. Apply cooldown timer to prevent repeated firing
```

Window sizes to support different gesture durations:

- **Primary window**: 60 frames (~2s at 30fps) — covers most gestures
- **Short window**: 30 frames (~1s) — catches quick signs like "yes", "no"
- **Long window**: 90 frames (~3s) — catches slow compound gestures

Run all three in parallel, take the highest-confidence prediction.

### 3c. Speed Augmentation During Training

Artificially vary the speed of training sequences so the model learns to be speed-invariant:

```python
def augment_speed(sequence, speed_range=(0.7, 1.4)):
    """Randomly stretch or compress a gesture's timing."""
    speed = np.random.uniform(*speed_range)
    T = len(sequence)
    new_T = max(10, int(T * speed))  # minimum 10 frames
    return normalize_sequence_length(sequence, new_T)
```

---

## 4. Landmark Preprocessing & Normalization

The current system feeds raw MediaPipe coordinates directly into the model. This makes predictions sensitive to where the hand is on screen and how far it is from the camera.

### 4a. Wrist-Relative Coordinates

Subtract the wrist landmark (index 0) from all other landmarks, making the hand position-invariant:

```python
def normalize_to_wrist(landmarks_63):
    """
    landmarks_63: array of 63 values (21 landmarks × 3 coords)
    Returns: wrist-relative coordinates
    """
    landmarks = landmarks_63.reshape(21, 3)
    wrist = landmarks[0].copy()
    landmarks -= wrist  # now wrist is at origin
    return landmarks.flatten()
```

### 4b. Scale Normalization

Divide by the distance from wrist to middle-finger MCP (landmark 9) to normalize hand size:

```python
def normalize_scale(landmarks_63):
    landmarks = landmarks_63.reshape(21, 3)
    wrist = landmarks[0]
    mcp = landmarks[9]
    scale = np.linalg.norm(mcp - wrist)
    if scale > 1e-6:
        landmarks /= scale
    return landmarks.flatten()
```

### 4c. Motion Delta Features

Append the **frame-to-frame difference** as additional features. This lets the model see velocity directly:

```python
def add_motion_deltas(sequence):
    """
    sequence: shape (T, 158)  # 126 hand + 32 face
    returns: shape (T, 316) — original + delta
    """
    deltas = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, deltas], axis=1)
```

This doubles the feature vector from 158 → 316 per frame, but gives the model explicit velocity information which is critical for gestures where the direction of motion matters.

### 4d. Facial Expression Features

ASL uses facial expressions **grammatically**, not just emotionally. Raised eyebrows mark yes/no questions, furrowed brows mark wh-questions, head shakes negate verbs, and mouth morphemes distinguish between otherwise identical hand signs (e.g., "not yet" vs "late").

The codebase already includes `face_landmarker.task` (MediaPipe FaceLandmarker). Extract a compact **32-dimensional face feature vector** per frame from the 478 face landmarks:

```python
def extract_face_features(face_landmarks):
    """
    Extract 32 expression-relevant features from MediaPipe's 478 face landmarks.
    Returns: np.array of shape (32,)
    """
    if face_landmarks is None:
        return np.zeros(32)
    
    lm = face_landmarks  # list of 478 NormalizedLandmarks
    
    features = []
    
    # --- Eyebrow raise (4 features) ---
    # Left eyebrow height relative to left eye
    l_brow_y = np.mean([lm[70].y, lm[63].y, lm[105].y])   # left brow landmarks
    l_eye_y = np.mean([lm[159].y, lm[145].y])              # left eye landmarks
    features.append(l_eye_y - l_brow_y)                      # positive = raised
    # Right eyebrow
    r_brow_y = np.mean([lm[300].y, lm[293].y, lm[334].y])
    r_eye_y = np.mean([lm[386].y, lm[374].y])
    features.append(r_eye_y - r_brow_y)
    # Brow furrow (inner brow distance)
    features.append(abs(lm[55].x - lm[285].x))
    # Brow asymmetry
    features.append((l_eye_y - l_brow_y) - (r_eye_y - r_brow_y))
    
    # --- Eye openness (4 features) ---
    l_eye_open = abs(lm[159].y - lm[145].y)  # upper - lower lid
    r_eye_open = abs(lm[386].y - lm[374].y)
    features.extend([l_eye_open, r_eye_open])
    features.extend([l_eye_open / max(r_eye_open, 1e-6), r_eye_open / max(l_eye_open, 1e-6)])  # wink detection
    
    # --- Mouth shape (12 features) ---
    mouth_open_v = abs(lm[13].y - lm[14].y)         # upper lip - lower lip
    mouth_width = abs(lm[61].x - lm[291].x)         # left corner - right corner
    features.extend([mouth_open_v, mouth_width])
    features.append(mouth_open_v / max(mouth_width, 1e-6))  # aspect ratio
    # Lip positions (upper, lower, left corner, right corner)
    features.extend([lm[13].y, lm[14].y, lm[61].y, lm[291].y])
    # Mouth pucker vs smile (corners relative to center)
    mouth_cx = (lm[61].x + lm[291].x) / 2
    features.extend([lm[61].x - mouth_cx, lm[291].x - mouth_cx])
    # Upper/lower lip protrusion (z-axis)
    features.extend([lm[13].z, lm[14].z])
    # Smile (corner height relative to center)
    mouth_cy = (lm[13].y + lm[14].y) / 2
    features.append((lm[61].y + lm[291].y) / 2 - mouth_cy)
    
    # --- Head pose (6 features) ---
    # Nose tip position (proxy for head tilt/rotation)
    nose = lm[1]
    features.extend([nose.x, nose.y, nose.z])
    # Head tilt: forehead-to-chin vector
    chin = lm[152]
    forehead = lm[10]
    features.extend([
        chin.x - forehead.x,  # lateral tilt
        chin.y - forehead.y,  # nod (forward/back)
        chin.z - forehead.z   # depth tilt
    ])
    
    # --- Cheek/jaw (6 features) ---
    # Jaw opening (chin drop)
    features.append(abs(lm[152].y - lm[10].y))
    # Cheek puff (left/right cheek x-displacement)
    features.extend([lm[234].x, lm[454].x])
    # Nose wrinkle
    features.extend([lm[98].y, lm[327].y])
    # Chin forward
    features.append(lm[152].z)
    
    return np.array(features[:32])  # ensure exactly 32
```

**Why 32 features and not all 478×3 = 1434?** — Most face landmarks track face mesh geometry (jaw outline, cheeks), not expression. The 32 features above capture the specific expression signals used in ASL grammar while keeping the input compact enough to not overwhelm the hand motion signal.

**Combining hand + face per frame:**

```python
def build_frame_vector(hand_result, face_landmarks):
    hand_vec = normalize_scale(normalize_to_wrist(landmarks_to_126(hand_result)))  # 126
    face_vec = extract_face_features(face_landmarks)                                # 32
    return np.concatenate([hand_vec, face_vec])                                      # 158
```

---

## 5. Data Collection Strategy

### 5a. Upgraded Self-Collection Tool

Rewrite `collect_data.py` to support:

1. **Continuous recording** — press START, perform the gesture naturally, press STOP. No fixed frame count.
2. **Automatic segmentation** — detect motion onset/offset using landmark velocity thresholds (hand starts moving → hand stops moving).
3. **Multi-speed prompts** — for each sign, prompt the user to perform it at "normal", "slow", and "fast" speeds.
4. **Multiple angles** — prompt the user to shift slightly left/right/closer/farther.
5. **Save variable-length .npy files** — each file is shape `(T, 126)` where T varies.

Target: **50+ samples per sign** across at least **3 different "signers"** (can be the same person varying style).

### 5b. External Datasets

For scaling to many more signs without manual collection:

| Dataset | Signs | Samples | Format | How to Use |
|---|---|---|---|---|
| **WLASL** | 2000 glosses | 21,000 clips | RGB video | Run MediaPipe offline → extract landmark sequences → save as .npy |
| **MS-ASL** | 1000 classes | 25,000 clips | RGB video | Same pipeline. Already have `process_msasl.py` in codebase |
| **How2Sign** | Continuous | 35,000 clips | RGB + depth | Advanced — for future continuous recognition |

The existing `process_msasl.py` script already extracts landmarks from videos. Extend it to:

1. Save variable-length sequences (not fixed 20 frames)
2. Apply wrist-relative normalization
3. Filter out sequences where no hand was detected in >30% of frames

### 5c. Gesture Vocabulary — Phase 1 (Target: 30 Signs)

Start with the most common ASL signs that are clearly motion-based:

**Greetings & Social**: hello, goodbye, please, thank you, sorry, nice to meet you
**Conversational**: yes, no, maybe, help, understand, don't understand, again, stop
**Questions**: what, where, when, how, who, why
**Pronouns**: me, you, he/she, we, they
**Common verbs**: want, need, like, know, think, go

---

## 6. Training Pipeline

### 6a. Dataset Class

```python
class GestureSequenceDataset(Dataset):
    def __init__(self, data_dir, target_length=40, augment=True):
        self.samples = []     # list of (sequence, label)
        self.target_length = target_length
        self.augment = augment
        
        for label_idx, sign_name in enumerate(sorted(os.listdir(data_dir))):
            sign_dir = os.path.join(data_dir, sign_name)
            for npy_file in glob.glob(f"{sign_dir}/*.npy"):
                seq = np.load(npy_file)  # (T, 126) variable T
                self.samples.append((seq, label_idx))
    
    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        
        # Apply wrist-relative + scale normalization per frame
        seq = np.array([normalize_scale(normalize_to_wrist(f)) for f in seq])
        
        if self.augment:
            seq = augment_speed(seq)          # random speed change
            seq = augment_spatial(seq)         # random rotation/jitter
            seq = augment_dropout_frames(seq)  # randomly drop 10% of frames
        
        # Resample to fixed length
        seq = normalize_sequence_length(seq, self.target_length)
        
        # Add motion deltas
        seq = add_motion_deltas(seq)  # (40, 316) — 158 features + 158 deltas
        
        return torch.tensor(seq, dtype=torch.float32), label
```

### 6b. Training Hyperparameters

```
Model: BiLSTM (input=316, hidden=256, layers=2, bidirectional=True)
       + Attention pool
       + FC(512 → 128 → num_classes)

Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Scheduler: CosineAnnealingLR, T_max=100
Loss: CrossEntropyLoss with label smoothing (0.1)
Batch size: 32
Epochs: 100–200 with early stopping (patience=15)
Dropout: 0.4 in FC layers, 0.2 in LSTM
```

### 6c. Augmentation Summary

| Augmentation | Parameters | Effect |
|---|---|---|
| Speed jitter | 0.7× – 1.4× | Handles fast/slow signers |
| Spatial jitter | ±2% noise on x,y,z | Camera/position robustness |
| Random rotation | ±15° around z-axis | Slight hand tilt invariance |
| Frame dropout | 10% probability | Handles MediaPipe detection gaps |
| Mirror (x-flip) | 50% probability | Left/right hand invariance |
| Temporal shift | ±5 frames | Handles imprecise segmentation |

---

## 7. Inference Pipeline (Live Camera)

```
┌──────────────┐
│  Camera Feed  │
│  (30+ fps)    │
└──────┬───────┘
       ↓
┌──────────────────────┐
│  MediaPipe Landmarks  │
│  21 pts × 3 coords    │
│  per hand (126-dim)   │
└──────┬───────────────┘
       ↓
┌──────────────────────────┐
│  Ring Buffer (90 frames)  │
│  Continuously appending   │
└──────┬───────────────────┘
       ↓ (every 5 frames)
┌──────────────────────────────────────────┐
│  Multi-window extraction                  │
│  Short: last 30 frames → resample to 40  │
│  Med:   last 60 frames → resample to 40  │
│  Long:  last 90 frames → resample to 40  │
└──────┬───────────────────────────────────┘
       ↓
┌──────────────────────────┐
│  Normalize (wrist-rel,    │
│  scale, add deltas)       │
└──────┬───────────────────┘
       ↓
┌──────────────────────────┐
│  BiLSTM + Attention       │
│  → top prediction per     │
│    window size             │
└──────┬───────────────────┘
       ↓
┌──────────────────────────────────┐
│  Confidence Aggregation           │
│  - Take max-confidence prediction │
│  - If conf > 0.75 → emit sign    │
│  - Cooldown: 1.5s after emission  │
└──────────────────────────────────┘
```

### Integration with Existing System

The gesture model runs **alongside** the existing static letter classifier, not replacing it:

- **Static mode**: current FC network → letters A–Z, 0–9 (single frame)
- **Gesture mode**: new BiLSTM → dynamic signs (multi-frame sequence)
- **Combined mode**: run both, display static letters normally + overlay gesture predictions when confidence is high

This matches the existing `prediction_mode` setting in `server.py` which already supports `'letters'`, `'numbers'`, `'both'` — extend to include `'gestures'` and `'all'`.

---

## 8. Files to Create/Modify

### New Files

| File | Purpose |
|---|---|
| `gestures/model.py` | BiLSTM + Attention model class |
| `gestures/dataset.py` | Variable-length sequence dataset with augmentation |
| `gestures/preprocess.py` | Wrist-relative norm, scale norm, motion deltas, temporal interpolation |
| `gestures/train.py` | Full training script with validation, early stopping, logging |
| `gestures/collect_data.py` | Upgraded data collector with continuous recording |
| `gestures/inference.py` | Multi-window sliding prediction with confidence aggregation |
| `gestures/extract_from_video.py` | Extract landmark sequences from video datasets (WLASL, MS-ASL) |

### Modified Files

| File | Changes |
|---|---|
| `server.py` | Add gesture inference endpoint, load gesture model, add `'gestures'` prediction mode |
| `enhanced.py` | Replace or extend `EnhancedProcessor` to use the new gesture model instead of the basic LSTM |

---

## 9. Implementation Phases

### Phase 1 — Foundation (Week 1–2)

- [ ] Build `preprocess.py` (normalization + interpolation)
- [ ] Build `dataset.py` (variable-length loader + augmentation)
- [ ] Build `model.py` (BiLSTM + Attention)
- [ ] Upgrade `collect_data.py` (continuous recording mode)
- [ ] Collect 50 samples each for 10 initial signs

### Phase 2 — Training & Validation (Week 2–3)

- [ ] Build `train.py` with validation split, early stopping, confusion matrix logging
- [ ] Train on self-collected data, iterate on hyperparameters
- [ ] Add WLASL/MS-ASL data extraction pipeline
- [ ] Train on combined self-collected + external data

### Phase 3 — Live Integration (Week 3–4)

- [ ] Build `inference.py` (multi-window sliding prediction)
- [ ] Integrate into `server.py` as new prediction mode
- [ ] Update frontend to display gesture predictions
- [ ] Add gesture mode toggle in settings

### Phase 4 — Scale (Week 4+)

- [ ] Expand to 30+ signs
- [ ] Collect data from multiple signers
- [ ] Evaluate Transformer architecture (Option B)
- [ ] Add continuous sign language recognition (sentence-level)

---

## 10. Success Metrics

| Metric | Target |
|---|---|
| Accuracy on held-out test set | >85% for 10 classes, >75% for 30 classes |
| Inference latency | <50ms per prediction cycle |
| Speed invariance | <5% accuracy drop between 0.7× and 1.4× speed |
| False positive rate | <5% on non-signing hand movement |
| Min samples needed per new sign | ≤30 (with augmentation) |
