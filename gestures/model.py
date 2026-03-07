"""
gestures/model.py
Bidirectional LSTM with Temporal Attention for dynamic gesture recognition.

Architecture:
    Input (batch, T, 316) → BiLSTM → Attention Pool → FC → num_classes

    - Input: 158 features per frame (126 hand + 32 face) × 2 (position + velocity)
    - BiLSTM captures forward + backward temporal context
    - Temporal attention learns which frames matter most per gesture
    - FC head with dropout for classification
"""

import torch# type: ignore
import torch.nn as nn
import torch.nn.functional as F# type: ignore


class TemporalAttention(nn.Module):
    """
    Learns a weighted sum over all timesteps.
    Instead of just taking the last hidden state, this lets the model
    focus on the most important frames for each gesture.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted sum
            weights: (batch, seq_len) — attention weights for visualization
        """
        # Compute attention scores
        scores = self.attention(lstm_output)           # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)             # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(lstm_output * weights, dim=1)  # (batch, hidden_size)

        return context, weights.squeeze(-1)


class GestureModel(nn.Module):
    """
    BiLSTM + Temporal Attention for dynamic sign language gesture recognition.
    
    Args:
        input_size:  Feature dimension per frame (default 316 = 158 + 158 deltas)
        hidden_size: LSTM hidden units per direction (total = 2 × hidden_size)
        num_layers:  Stacked LSTM layers
        num_classes: Number of gesture classes to predict
        dropout:     Dropout rate in LSTM and FC layers
    """

    def __init__(
        self,
        num_classes,
        input_size=316,
        hidden_size=256,
        num_layers=2,
        dropout=0.4
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection: smooth the raw features before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # lighter dropout on input
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Temporal attention over BiLSTM outputs
        self.attention = TemporalAttention(hidden_size * 2)  # *2 for bidirectional

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, input_size) — preprocessed gesture sequence
            return_attention: if True, also return attention weights
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len) — only if return_attention=True
        """
        # Project input features
        x = self.input_proj(x)                       # (batch, seq_len, hidden_size)

        # BiLSTM
        lstm_out, _ = self.lstm(x)                   # (batch, seq_len, hidden_size * 2)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_size * 2)

        # Classify
        logits = self.classifier(context)            # (batch, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits


class GestureModelLite(nn.Module):
    """
    Smaller variant for quick prototyping or when GPU is limited.
    Uses standard LSTM (not bidirectional) with simpler attention.
    """

    def __init__(self, num_classes, input_size=316, hidden_size=128, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.attention = TemporalAttention(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits


# ============================================================
# UTILITY: Model summary
# ============================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify shapes
    model = GestureModel(num_classes=30)
    print(f"GestureModel parameters: {count_parameters(model):,}")

    # Simulate a batch of 4 sequences, 40 frames, 316 features
    dummy = torch.randn(4, 40, 316)
    logits, attn = model(dummy, return_attention=True)# type: ignore
    print(f"Input:  {dummy.shape}")
    print(f"Output: {logits.shape}")
    print(f"Attention: {attn.shape}")
    print(f"Attention sum (should be ~1.0): {attn[0].sum().item():.4f}")

    lite = GestureModelLite(num_classes=30)
    print(f"\nGestureModelLite parameters: {count_parameters(lite):,}")
