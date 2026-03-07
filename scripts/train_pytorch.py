# train_pytorch.py
# ==================
# PURPOSE: Train a neural network to classify sign language letters from landmarks
#
# HOW NEURAL NETWORKS WORK (for someone coming from C):
# =====================================================
# A neural network is like a function with learnable parameters (weights).
# 
# Imagine: y = f(x) where x is input (landmarks), y is output (letter A-Z)
# 
# The network has "layers" - each layer transforms the data:
#   Input (126 features) → Hidden Layer 1 (256 neurons) → Hidden Layer 2 (128) → Output (26 letters)
#
# Training process:
#   1. Forward pass: Feed data through network, get prediction
#   2. Calculate loss: How wrong was the prediction?
#   3. Backward pass: Calculate gradient (direction to improve)
#   4. Update weights: Adjust parameters to reduce loss
#   5. Repeat thousands of times
#
# USAGE:
#   python train_pytorch.py

import torch # type: ignore
import torch.nn as nn
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
import os

# ================= CONFIGURATION =================
DATA_FILE = 'landmark_data.csv'
MODEL_FILE = 'gesture_model_pytorch.pth'
SCALER_FILE = 'gesture_scaler.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

# Hyperparameters (settings that control training)
BATCH_SIZE = 128       # Larger batch size for more stable gradients
EPOCHS = 300          # Reduced epochs with early stopping
LEARNING_RATE = 0.0003 # Smaller learning rate for better convergence
HIDDEN_SIZE_1 = 256   # Neurons in first hidden layer
HIDDEN_SIZE_2 = 128   # Neurons in second hidden layer
TEST_SIZE = 0.2       # 20% of data for testing
WEIGHT_DECAY = 1e-4   # L2 regularization to prevent overfitting
DROPOUT_RATE = 0.3    # Moderate dropout for regularization
EARLY_STOP_PATIENCE = 30  # Stop if no improvement for 30 epochs

# Device: Use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= DATASET CLASS =================
# In PyTorch, you create a Dataset class to load your data
class LandmarkDataset(Dataset):
    """
    A Dataset wraps your data so PyTorch can iterate over it.
    Think of it like a fancy array that returns (input, label) pairs.
    """
    
    def __init__(self, features, labels):
        # Convert to PyTorch tensors (like numpy arrays but for PyTorch)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        """How many samples in the dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get one sample by index"""
        return self.features[idx], self.labels[idx]


# ================= NEURAL NETWORK MODEL =================
class SignLanguageClassifier(nn.Module):
    """
    The actual neural network architecture.
    
    nn.Module is the base class for all neural networks in PyTorch.
    You define layers in __init__, and the forward pass in forward().
    """
    
    def __init__(self, input_size, hidden1, hidden2, num_classes, dropout_rate=0.3):
        super(SignLanguageClassifier, self).__init__()
        
        # Define layers
        # nn.Linear = fully connected layer (like matrix multiplication + bias)
        self.layer1 = nn.Linear(input_size, hidden1)   # 126 → 256
        self.layer2 = nn.Linear(hidden1, hidden2)       # 256 → 128
        self.layer3 = nn.Linear(hidden2, num_classes)   # 128 → num_classes
        
        # Activation functions (add non-linearity)
        self.relu = nn.ReLU()      # ReLU: max(0, x) - simple but effective
        self.dropout1 = nn.Dropout(dropout_rate)  # Higher dropout to prevent overfitting
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)  # Slightly less dropout in layer 2
        
        # Batch normalization (stabilizes training)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
    
    def forward(self, x):
        """
        The forward pass: data flows through the network.
        This defines how input x becomes output.
        
        x shape: (batch_size, 126)  # 126 landmark features
        output shape: (batch_size, num_classes)  # One score per letter
        """
        # Layer 1: Linear → BatchNorm → ReLU → Dropout
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: Same pattern
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Output layer: No activation (softmax applied in loss function)
        x = self.layer3(x)
        return x


# ================= TRAINING FUNCTION =================
def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Train for one complete pass through the data (one epoch).
    """
    model.train()  # Set model to training mode (enables dropout)
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        # Move data to device (GPU/CPU)
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass: Get predictions
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass: Compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest score
        correct += (predicted == labels).sum().item() # type: ignore
        total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    """
    Evaluate model on test data (no training, just measure accuracy).
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients (faster)
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item() # type: ignore
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


# ================= MAIN TRAINING SCRIPT =================
def main():
    print("=" * 60)
    print("PYTORCH SIGN LANGUAGE CLASSIFIER - TRAINING")
    print("=" * 60)
    print(f"Using device: {device}")
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: {DATA_FILE} not found!")
        print("Run data_collector.py first to collect training data.")
        return
    
    # ===== STEP 1: Load Data =====
    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Labels: {df['label'].unique()}")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].astype(str).values  # Convert to string to handle mixed types
    
    # ===== STEP 2: Encode Labels =====
    print("\n[2/5] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"  - Classes: {list(label_encoder.classes_)}")
    print(f"  - Num classes: {num_classes}")
    
    # ===== STEP 3: Split and Scale =====
    print("\n[3/5] Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Data augmentation: Add noise to training data to improve generalization
    print("  - Applying data augmentation (adding noise)...")
    noise_factor = 0.02
    X_train_augmented = X_train_scaled + np.random.normal(0, noise_factor, X_train_scaled.shape)
    X_train_augmented = np.vstack([X_train_scaled, X_train_augmented])  # Double the data
    y_train_augmented = np.hstack([y_train, y_train])
    
    print(f"  - Augmented training samples: {len(X_train_augmented)}")
    
    # Create PyTorch datasets and dataloaders
    train_dataset = LandmarkDataset(X_train_augmented, y_train_augmented)
    test_dataset = LandmarkDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    # ===== STEP 4: Create Model =====
    print("\n[4/5] Creating neural network...")
    input_size = X.shape[1]  # 126 features
    model = SignLanguageClassifier(input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, num_classes, DROPOUT_RATE)
    model = model.to(device)
    
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden layers: {HIDDEN_SIZE_1} → {HIDDEN_SIZE_2}")
    print(f"  - Output size: {num_classes}")
    
    # Calculate class weights for imbalanced data
    # Classes with fewer samples get higher weights
    from sklearn.utils.class_weight import compute_class_weight # type: ignore
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"  - Using class weights for imbalanced data")
    print(f"  - Min weight: {class_weights.min():.2f}, Max weight: {class_weights.max():.2f}")
    
    # Loss function: CrossEntropyLoss with class weights for balanced training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer: AdamW with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler (reduce LR when training plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)
    
    # ===== STEP 5: Train =====
    print("\n[5/5] Training...")
    print("-" * 60)
    
    best_accuracy = 0
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        scheduler.step(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
        
        # Early stopping check
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden1': HIDDEN_SIZE_1,
                'hidden2': HIDDEN_SIZE_2,
                'num_classes': num_classes,
                'label_encoder_classes': list(label_encoder.classes_)
            }, MODEL_FILE)
    
    print("-" * 60)
    print(f"\n✓ Training complete!")
    print(f"  - Best test accuracy: {best_accuracy:.2%}")
    print(f"  - Model saved to: {MODEL_FILE}")
    
    # Save scaler for inference
    import joblib # type: ignore
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    print(f"  - Scaler saved to: {SCALER_FILE}")
    print(f"  - Label encoder saved to: {LABEL_ENCODER_FILE}")


if __name__ == '__main__':
    main()
