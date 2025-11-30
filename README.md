# Kolam Classification CNN

A professional, scalable CNN-based image classification system for kolam images. This project is designed to classify different types of kolam patterns (starting with elephant kolam) and can be easily extended to support multiple kolam types.

## 🎯 Features

- **Scalable Architecture**: Easy to add new kolam types (peacock, geometric, cow, etc.)
- **Feature Extraction**: Extract feature maps and vectors for Vision Transformer (ViT) integration
- **Professional Code Structure**: Modular, well-documented, and maintainable
- **Production Ready**: Includes training, validation, inference, and feature extraction pipelines
- **GPU Support**: Automatic GPU detection with CUDA support

## 📁 Project Structure

```
kolam-classification/
├── config.py                 # Configuration file (hyperparameters, paths)
├── train.py                  # Training script
├── inference.py              # Single image classification
├── extract_features.py       # Feature extraction for ViT
├── models/
│   └── kolam_cnn.py         # CNN model architecture
├── data/
│   └── dataset.py           # Data loading and preprocessing
├── data/
│   ├── train/               # Training images (organized by class)
│   │   ├── elephant_kolam/
│   │   └── not_recognized/
│   └── test/                # Test images (optional)
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs and history
├── features/                # Extracted features
└── requirements.txt         # Python dependencies
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU Support (CUDA):**
- Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/)
- Example for CUDA 11.8:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 2. Prepare Your Dataset

Organize your images in the following directory structure:

```
data/
├── train/
│   ├── elephant_kolam/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── not_recognized/
│       ├── image1.jpg
│       └── ...
└── test/                    # Optional
    ├── elephant_kolam/
    └── not_recognized/
```

**Important Notes:**
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- Images will be automatically resized to 224x224
- Ensure balanced classes for better training

### 3. Configure Settings (Optional)

Edit `config.py` to adjust:
- Image size
- Batch size
- Learning rate
- Number of epochs
- Model architecture parameters

**Key Configuration Options:**
- `NUM_CLASSES`: Number of classes (currently 2: elephant_kolam, not_recognized)
- `BATCH_SIZE`: Adjust based on your GPU memory (default: 32)
- `NUM_WORKERS`: Number of data loading workers (adjust based on CPU cores)
- `EPOCHS`: Number of training epochs (default: 50)

## 📖 Usage

### Training the Model

```bash
python train.py
```

**What happens:**
1. Loads images from `data/train/`
2. Splits into train/validation sets (80/20)
3. Trains CNN with data augmentation
4. Saves best model to `models/best_kolam_classifier.pth`
5. Saves training history to `logs/`

**Training Output:**
- Best model checkpoint: `models/best_kolam_classifier.pth`
- Periodic checkpoints: `models/checkpoint_epoch_N.pth`
- Training history: `logs/training_history_TIMESTAMP.json`

### Classifying a Single Image

```bash
python inference.py path/to/image.jpg
```

**Options:**
```bash
# Use specific model checkpoint
python inference.py image.jpg --model models/checkpoint_epoch_10.pth

# Extract features along with classification
python inference.py image.jpg --extract-features

# Adjust confidence threshold
python inference.py image.jpg --threshold 0.7
```

**Output:**
- Predicted class name
- Confidence score
- All class probabilities

### Extracting Features for ViT Integration

```bash
# Extract features from a single image
python extract_features.py --input path/to/image.jpg

# Extract features from all images in a directory
python extract_features.py --input data/train/elephant_kolam/ --output features/elephant_kolam_features/
```

**Output:**
- `extracted_features.pt`: PyTorch tensor format
- `extracted_features.npz`: NumPy format for compatibility

**Feature Formats:**
- `feature_vectors`: Shape `(N, 512)` - Flattened feature vectors
- `feature_maps`: Shape `(N, channels, H, W)` - Spatial feature maps

## 🔧 Scaling to Multiple Kolam Types

### Adding New Kolam Classes

1. **Add Training Data:**
   ```
   data/train/
   ├── elephant_kolam/
   ├── peacock_kolam/      # New class
   ├── geometric_kolam/    # New class
   ├── cow_kolam/          # New class
   └── not_recognized/
   ```

2. **Update Configuration:**
   Edit `config.py`:
   ```python
   NUM_CLASSES = 5  # Update number of classes
   
   CLASS_LABELS = {
       0: "elephant_kolam",
       1: "peacock_kolam",
       2: "geometric_kolam",
       3: "cow_kolam",
       4: "not_recognized"
   }
   ```

3. **Retrain the Model:**
   ```bash
   python train.py
   ```

The model architecture will automatically adjust to the new number of classes.

## 🎨 Model Architecture

The CNN architecture consists of:
- **5 Convolutional Blocks**: Progressive feature extraction
- **Batch Normalization**: For stable training
- **Dropout**: For regularization
- **Adaptive Pooling**: For flexible input sizes
- **Feature Extraction**: Intermediate features available at multiple layers

**Feature Extraction Points:**
- `feature_map`: Spatial features from `conv_final` layer (before pooling)
- `feature_vector`: Flattened features after `fc1` layer (512-dim)

## 📊 Training Tips

1. **Data Quality:**
   - Ensure balanced classes
   - Include diverse lighting conditions
   - Add variations in kolam patterns

2. **Hyperparameter Tuning:**
   - Adjust `LEARNING_RATE` if loss doesn't decrease
   - Increase `BATCH_SIZE` if you have GPU memory
   - Adjust `EPOCHS` based on convergence

3. **Early Stopping:**
   - Enabled by default (patience: 10 epochs)
   - Prevents overfitting
   - Adjust in `config.py` if needed

4. **Data Augmentation:**
   - Enabled by default for training
   - Includes rotation, flipping, color jitter
   - Disable in `config.py` if needed

## 🔌 Cloud GPU Setup

### Google Colab

1. Upload project to Google Drive or GitHub
2. Open Colab notebook
3. Install dependencies:
   ```python
   !pip install torch torchvision Pillow numpy
   ```
4. Set `DEVICE = "cuda"` in `config.py`

### AWS/Azure/GCP

1. Set environment variable:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```
2. Install CUDA-enabled PyTorch
3. Run training as normal

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error:**
   - Reduce `BATCH_SIZE` in `config.py`
   - Reduce `NUM_WORKERS`

2. **Slow Training:**
   - Enable GPU: Check CUDA installation
   - Increase `NUM_WORKERS` (but not more than CPU cores)
   - Use smaller `IMAGE_SIZE`

3. **Poor Accuracy:**
   - Add more training data
   - Ensure balanced classes
   - Adjust learning rate
   - Train for more epochs

4. **Import Errors:**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.7+)

## 📝 Code Structure Details

### Configuration (`config.py`)
- Centralized hyperparameters
- Path management
- Device configuration
- Class label definitions

### Model (`models/kolam_cnn.py`)
- Modular CNN architecture
- Feature extraction capability
- Easy to modify and extend

### Data Loading (`data/dataset.py`)
- Automatic class detection from folders
- Data augmentation
- Train/validation splitting

### Training (`train.py`)
- Full training pipeline
- Early stopping
- Checkpointing
- Training history logging

### Inference (`inference.py`)
- Single image classification
- Batch inference support
- Confidence scoring

### Feature Extraction (`extract_features.py`)
- Batch feature extraction
- Multiple output formats
- ViT-ready features

## 🚀 Future Enhancements

- [ ] Vision Transformer (ViT) integration
- [ ] Web API for real-time classification
- [ ] Grad-CAM visualization
- [ ] Model ensemble support
- [ ] Transfer learning from pre-trained models

## 📄 License

This project is designed for educational and research purposes.

## 🤝 Contributing

This is a scalable foundation. To extend:
1. Add new kolam types to dataset
2. Update `NUM_CLASSES` and `CLASS_LABELS` in `config.py`
3. Retrain the model

---

**Note:** This is a production-ready, scalable implementation. All code includes clear headers and sections for easy understanding and modification.
