"""
gestures — Dynamic gesture recognition package for SignMate.

Modules:
    preprocess   - Landmark normalization, face features, temporal interpolation
    model        - BiLSTM + Temporal Attention model
    dataset      - Variable-length PyTorch Dataset with augmentation
    train        - Training script with early stopping
    collect_data - Upgraded data collector 
    inference    - Live multi-window sliding prediction engine
"""
