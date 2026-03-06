# How It Works — Expanded Content Draft

> **Purpose**: This is a content draft for the "How It Works" section on the SignMate homepage. It replaces the current 3-card layout with a richer, step-by-step breakdown of the full translation pipeline. No code has been changed — this file exists purely for review before implementation.

---

## Proposed Section Title

**"How It Works"** (kept the same) — with an optional subtitle:

> *From hand gesture to translated text in milliseconds — here's what happens under the hood.*

---

## Step-by-Step Pipeline (6 Steps)

Each step below is intended to become a card, numbered step, or visual block on the homepage. Copy is written to be user-facing — concise, non-technical, but informative.

---

### Step 1 — Camera Capture

**Card Title:** Open Your Camera
**Icon idea:** Video camera / webcam

**Copy:**

> SignMate connects to your webcam and captures a live video feed at up to 60 fps. No external hardware required — any built-in or USB camera works. The feed is mirrored so your movements feel natural, like looking in a mirror.

**Technical detail (for tooltip or expandable):**
OpenCV captures frames at 1280×720. Each frame is flipped horizontally and converted from BGR to RGB color space before being passed into the detection pipeline.

---

### Step 2 — Hand Detection & Landmark Tracking

**Card Title:** Detect & Track Your Hands
**Icon idea:** Hand outline with dots at joints

**Copy:**

> Google's MediaPipe Hand Landmarker identifies your hands in each frame and maps **21 precise landmarks** — from your wrist to every fingertip — in real time. It tracks joint positions in 3D (x, y, and depth), even as your hand moves, rotates, or partially occludes.

**Technical detail:**
MediaPipe runs asynchronously in LIVE_STREAM mode, returning results via callback. It handles up to 2 hands simultaneously. Each landmark provides normalized x, y, z coordinates, yielding 63 features per hand (21 landmarks × 3 coordinates) and 126 total features when both hands are tracked.

---

### Step 3 — Feature Extraction & Normalization

**Card Title:** Extract Hand Geometry
**Icon idea:** Data/graph/waveform icon

**Copy:**

> The raw 3D positions of all 21 landmarks per hand are extracted into a compact feature vector. These coordinates are then **normalized** using the same statistical scaling applied during training, ensuring that hand size, camera distance, and positioning don't skew the results.

**Technical detail:**
A sklearn StandardScaler (saved as `gesture_scaler.pkl`) transforms the 126-dimensional feature vector to zero-mean, unit-variance — matching the distribution the neural network was trained on. Left and right hand features are ordered consistently (left first, right second) to match the training data collection order.

---

### Step 4 — AI Classification

**Card Title:** AI Predicts the Sign
**Icon idea:** Brain / neural network / sparkle

**Copy:**

> Your hand pose is fed into a **deep neural network** that was trained on thousands of labeled sign language examples. The network outputs a probability for every possible sign (A–Z letters and 0–9 numbers) and returns the **top 3 predictions** with confidence scores — so you can see exactly how certain the AI is.

**Technical detail:**
The model is a 3-layer fully-connected PyTorch classifier (`SignLanguageClassifier`) with BatchNorm and Dropout at each layer for stability and generalization:

- **Layer 1:** Input (126 features) → Hidden (512 neurons) → BatchNorm → ReLU → Dropout (30%)
- **Layer 2:** Hidden (512) → Hidden (256 neurons) → BatchNorm → ReLU → Dropout (30%)
- **Layer 3:** Hidden (256) → Output (36 classes: A–Z + 0–9)

A softmax function converts raw scores into probabilities. The prediction mode can be filtered to letters-only, numbers-only, or both.

---

### Step 5 — Word & Sentence Building

**Card Title:** Build Words & Sentences
**Icon idea:** Chat bubble / text builder

**Copy:**

> Individual letter predictions are stitched together into words using a **smart sentence builder** with autocomplete powered by a 370,000-word dictionary. Hold a sign steady to confirm it, pause briefly to add a space, and watch your message assemble in real time. Finished sentences can be read aloud with built-in **text-to-speech**.

**Technical detail:**
A stability timer (`min_stable_duration`, default 0.25s) ensures only intentionally held signs are committed. The dictionary lookup (`words_dictionary.json`, ~370k words) powers autocomplete suggestions. The sentence builder supports backspace gestures, manual corrections, and clipboard copy.

---

### Step 6 — Full-Word Recognition (Enhanced Mode)

**Card Title:** Recognize Whole Words
**Icon idea:** Sparkles / magic wand / LSTM icon

**Copy:**

> Beyond spelling letter-by-letter, SignMate's **Enhanced Mode** uses a second AI model — an LSTM (Long Short-Term Memory) network — that watches a **sequence of 20 frames** to recognize full dynamic signs like "hello," "thank you," "yes," and "no." This captures motion and timing, not just a single pose.

**Technical detail:**
The LSTM model (`SignModel`) takes a sequence of 20 frames × 126 features, processes them through a 2-layer LSTM (hidden size 128), and classifies across the supported word vocabulary. A confidence threshold of 70% is required before a word prediction is surfaced. Face landmark detection is also available in Enhanced Mode for signs that involve facial expressions.

---

## Additional Ideas for Visual/Layout Enhancements

These are optional elements that could accompany the step cards:

1. **Numbered Steps / Vertical Timeline**: Instead of a flat 3-column grid, present the pipeline as a numbered vertical or horizontal timeline, showing the data flowing from step to step.

2. **Animated Diagram**: A simple SVG or Lottie animation showing:
   - Camera → Hand with dots → Feature vector → Neural network → Letter output

3. **Live Stats Strip**: Below the steps, show real-time-looking stats:
   - "21 landmarks tracked per hand"
   - "< 30ms latency"
   - "36 signs recognized (A–Z + 0–9)"
   - "370k word dictionary"

4. **Before/After Visual**: Show a raw camera frame on the left, and the same frame with landmarks + prediction overlay on the right.

---

## Summary of Changes vs. Current

| Aspect | Current | Proposed |
|---|---|---|
| **Number of cards** | 3 | 6 |
| **Depth** | 1 sentence each | Full paragraph + optional technical detail |
| **Pipeline clarity** | Vague ("tracks landmarks", "deep learning model") | Step-by-step with specifics (126 features, 3-layer network, softmax, dictionary size) |
| **Enhanced Mode** | Not mentioned | Dedicated step explaining LSTM word recognition |
| **Feature extraction** | Not mentioned | Dedicated step explaining normalization |
| **Sentence building** | Brief mention | Full explanation with autocomplete, TTS, dictionary |
| **Visual suggestions** | None | Timeline layout, animated diagram, stats strip |
