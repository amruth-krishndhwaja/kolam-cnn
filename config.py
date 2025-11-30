"""
================================================================================
CONFIGURATION FILE
================================================================================
Purpose: Centralized configuration for the Kolam Classification CNN project
         Contains all hyperparameters, paths, and model settings
================================================================================
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
FEATURES_DIR = BASE_DIR / "features"

# Create directories if they don't exist
for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, LOGS_DIR, FEATURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
IMAGE_SIZE = (224, 224)  # Standard input size for CNN
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjust based on your CPU cores
VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# Data augmentation settings
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
NUM_CLASSES = 2  # Start with: [elephant_kolam, not_recognized]
                 # Will scale to: [elephant_kolam, peacock_kolam, geometric_kolam, cow_kolam, not_recognized]

# CNN Architecture parameters
BASE_CHANNELS = 64  # Starting number of channels
DROPOUT_RATE = 0.5
USE_BATCH_NORM = True

# Feature extraction settings
FEATURE_LAYER = "conv_final"  # Layer name for feature extraction
FEATURE_DIM = 512  # Dimension of extracted features

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Checkpointing
SAVE_BEST_MODEL = True
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
MODEL_CHECKPOINT = MODEL_DIR / "best_kolam_classifier.pth"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for classification

# ============================================================================
# CLASS LABELS
# ============================================================================
# Define class labels - easily scalable for future kolam types
CLASS_LABELS = {
    0: "elephant_kolam",
    1: "not_recognized"
}

# Future expansion (uncomment when adding more classes):
# CLASS_LABELS = {
#     0: "elephant_kolam",
#     1: "peacock_kolam",
#     2: "geometric_kolam",
#     3: "cow_kolam",
#     4: "not_recognized"
# }

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Auto-detect CUDA availability (will be checked at runtime)
# For cloud GPU (e.g., Google Colab, AWS), CUDA will be auto-detected
# To force CPU: set DEVICE = "cpu"
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"  # Default to CPU if torch not installed yet

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
SAVE_TRAINING_PLOTS = True

