"""
================================================================================
TRAINING SCRIPT FOR KOLAM CLASSIFICATION CNN
================================================================================
Purpose: Train the CNN model on kolam images with proper logging,
         checkpointing, and early stopping
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
from datetime import datetime
import json

# Import project modules
from config import *
from models.kolam_cnn import create_model
from data.dataset import create_dataloaders


class EarlyStopping:
    """
    ============================================================================
    EARLY STOPPING UTILITY
    ============================================================================
    Purpose: Stop training when validation loss stops improving
    ============================================================================
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    ============================================================================
    TRAIN ONE EPOCH
    ============================================================================
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    ============================================================================
    VALIDATE MODEL
    ============================================================================
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """
    ============================================================================
    SAVE MODEL CHECKPOINT
    ============================================================================
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'num_classes': model.num_classes,
        'base_channels': model.base_channels
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def main():
    """
    ============================================================================
    MAIN TRAINING FUNCTION
    ============================================================================
    """
    print("=" * 80)
    print("KOLAM CLASSIFICATION CNN - TRAINING")
    print("=" * 80)
    
    # ============================================================================
    # SETUP DEVICE
    # ============================================================================
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # ============================================================================
    # CREATE DATA LOADERS
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        use_augmentation=USE_AUGMENTATION
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # ============================================================================
    # CREATE MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    model = create_model(
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        dropout_rate=DROPOUT_RATE,
        use_batch_norm=USE_BATCH_NORM,
        feature_dim=FEATURE_DIM
    )
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============================================================================
    # SETUP LOSS FUNCTION AND OPTIMIZER
    # ============================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = None
    if USE_SCHEDULER:
        scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    # Early stopping
    early_stopping = None
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA
        )
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            if SAVE_BEST_MODEL:
                best_model_path = MODEL_DIR / "best_kolam_classifier.pth"
                save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
        
        # Save periodic checkpoints
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = MODEL_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # ============================================================================
    # TRAINING COMPLETE
    # ============================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    history_path = LOGS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # ============================================================================
    # TEST SET EVALUATION (IF AVAILABLE)
    # ============================================================================
    if test_loader:
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST SET")
        print("=" * 80)
        
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

