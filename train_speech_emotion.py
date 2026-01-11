#!/usr/bin/env python3
"""
Train Speech Emotion Recognition Model
CNN + BiLSTM + Attention architecture for 4-class emotion classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing_audio_speech_emotion import load_combined_datasets, CLASS_LABELS
from models.speech_emotion_model import SpeechEmotionCNNBiLSTMAttention


class SpeechEmotionDataset(Dataset):
    """PyTorch Dataset for speech emotion features"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(np.array(features))
        self.labels = torch.LongTensor(np.array(labels))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def calculate_class_weights(labels):
    """Calculate class weights for balanced training."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    class_weights = total / (len(unique) * counts.astype(float))
    return torch.FloatTensor(class_weights)


def train_model(
    crema_dir: str,
    tess_dir: str,
    savee_dir: str,
    model_path: str = "models/speech_emotion_model.pth",
    batch_size: int = 32,
    num_epochs: int = 15,
    learning_rate: float = 0.0001,
    device: str = None,
    use_augmentation: bool = True,
    cache_dir: str = "data/processed"
):
    """
    Train Speech Emotion Recognition model.
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Speech Emotion Recognition Model Training")
    print("=" * 70)
    print(f'Using device: {device}')
    print('=' * 70)
    
    # Load datasets
    print("\n1. Loading combined datasets...")
    X, y = load_combined_datasets(
        crema_dir=crema_dir,
        tess_dir=tess_dir,
        savee_dir=savee_dir,
        apply_augmentation=use_augmentation,
        cache_dir=cache_dir
    )
    
    X = np.array(X)
    y = np.array(y)
    
    print(f'\nTotal samples: {len(X)}')
    print(f'Feature dimension: {X[0].shape[0]}')
    print(f'Number of classes: {len(np.unique(y))}')
    
    # Split into train/validation/test
    print("\n2. Splitting into train/validation/test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Test samples: {len(X_test)}')
    
    # Create datasets
    print("\n3. Creating PyTorch datasets...")
    train_dataset = SpeechEmotionDataset(X_train, y_train)
    val_dataset = SpeechEmotionDataset(X_val, y_val)
    test_dataset = SpeechEmotionDataset(X_test, y_test)
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    # Create weighted sampler for balanced training
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
    
    # Create model
    print("\n4. Creating model...")
    input_dim = X[0].shape[0]
    model = SpeechEmotionCNNBiLSTMAttention(input_dim=input_dim, num_classes=4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_model_state = None
    
    print("\n5. Training model...")
    print('=' * 70)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f'  → Best model saved (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    print('=' * 70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    print(f'Test F1-Score (weighted): {test_f1:.4f}')
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 70)
    class_names = [CLASS_LABELS[i] for i in range(4)]
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("-" * 70)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Speech Emotion Recognition')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = "models/speech_emotion_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    
    # Validation accuracy check
    if test_acc >= 80:
        print("\n✅ SUCCESS: Test accuracy >= 80%!")
    else:
        print(f"\n⚠️  WARNING: Test accuracy ({test_acc:.2f}%) is below 80% target.")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    
    return model, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition model')
    parser.add_argument('--crema-dir', type=str, required=True,
                        help='Path to CREMA-D dataset directory')
    parser.add_argument('--tess-dir', type=str, required=True,
                        help='Path to TESS dataset directory')
    parser.add_argument('--savee-dir', type=str, required=True,
                        help='Path to SAVEE dataset directory')
    parser.add_argument('--model-path', type=str, default='models/speech_emotion_model.pth',
                        help='Path to save trained model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--cache-dir', type=str, default='data/processed',
                        help='Directory for cached preprocessed data')
    
    args = parser.parse_args()
    
    train_model(
        crema_dir=args.crema_dir,
        tess_dir=args.tess_dir,
        savee_dir=args.savee_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        use_augmentation=not args.no_augment,
        cache_dir=args.cache_dir
    )


if __name__ == '__main__':
    main()

