# Setup Guide - Getting Your Code Running

This guide provides step-by-step instructions to get your Kolam Classification CNN running on your computer.

## 📋 Prerequisites

1. **Python 3.7 or higher** (Python 3.8+ recommended)
   - Check version: `python --version`
   - Download from: https://www.python.org/downloads/

2. **pip** (Python package installer)
   - Usually comes with Python
   - Check: `pip --version`

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Project Directory

```bash
cd C:\Users\gliit\kolam-classification
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What this installs:**
- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision utilities
- `Pillow`: Image processing library
- `numpy`: Numerical computing

**If installation fails:**
- Try: `pip install --upgrade pip` first
- For Windows issues, try: `python -m pip install -r requirements.txt`

### Step 3: GPU Support (Optional but Recommended)

**For NVIDIA GPU (CUDA):**

1. Check if you have NVIDIA GPU:
   ```bash
   nvidia-smi
   ```

2. If you have GPU, install CUDA-enabled PyTorch:
   - Visit: https://pytorch.org/get-started/locally/
   - Select your CUDA version
   - Example for CUDA 11.8:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

3. Verify GPU is detected:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True` if GPU is available.

**For CPU-only (no GPU):**
- The default installation will work fine
- Training will be slower but functional

### Step 4: Prepare Your Dataset

**Create the following folder structure:**

```
C:\Users\gliit\kolam-classification\
└── data\
    ├── train\
    │   ├── elephant_kolam\
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ... (all your elephant kolam images)
    │   └── not_recognized\
    │       ├── image1.jpg
    │       └── ... (images that are NOT elephant kolam)
    └── test\  (optional)
        ├── elephant_kolam\
        └── not_recognized\
```

**How to organize your images:**

1. **Create folders:**
   ```bash
   mkdir data\train\elephant_kolam
   mkdir data\train\not_recognized
   mkdir data\test  # Optional
   ```

2. **Copy your images:**
   - Put all elephant kolam images in `data\train\elephant_kolam\`
   - Put other kolam images (or non-kolam images) in `data\train\not_recognized\`
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

**Important:**
- Use descriptive folder names exactly as shown (lowercase with underscores)
- Ensure you have at least 50-100 images per class for good results
- More images = better accuracy
- Try to have balanced classes (similar number of images in each folder)

### Step 5: Configure Settings (If Needed)

Open `config.py` and adjust if necessary:

**Common adjustments:**

1. **If you have limited GPU memory:**
   ```python
   BATCH_SIZE = 16  # Reduce from 32
   ```

2. **If you have many CPU cores:**
   ```python
   NUM_WORKERS = 8  # Increase from 4 (but not more than CPU cores)
   ```

3. **For faster training (less accuracy):**
   ```python
   EPOCHS = 30  # Reduce from 50
   ```

4. **For better accuracy (slower training):**
   ```python
   EPOCHS = 100  # Increase from 50
   ```

**Most users can use default settings!**

### Step 6: Train the Model

```bash
python train.py
```

**What to expect:**
- First run will take longer (loading data)
- You'll see progress for each epoch
- Training time depends on:
  - Number of images (more = longer)
  - GPU vs CPU (GPU is 10-50x faster)
  - Number of epochs

**Typical training times:**
- CPU: 2-6 hours for 1000 images, 50 epochs
- GPU: 10-30 minutes for 1000 images, 50 epochs

**Training will:**
- Show loss and accuracy for each epoch
- Save best model automatically
- Stop early if not improving (early stopping)

**Output files:**
- `models\best_kolam_classifier.pth` - Best model (use this for inference)
- `models\checkpoint_epoch_N.pth` - Periodic checkpoints
- `logs\training_history_TIMESTAMP.json` - Training statistics

### Step 7: Test Classification

Once training is complete, test on a new image:

```bash
python inference.py path\to\your\test_image.jpg
```

**Example:**
```bash
python inference.py data\test\elephant_kolam\test1.jpg
```

**Output:**
- Predicted class (elephant_kolam or not_recognized)
- Confidence percentage
- All class probabilities

**With feature extraction:**
```bash
python inference.py image.jpg --extract-features
```

## 🔧 Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision
```

### Issue 2: "Out of memory" or "CUDA out of memory"

**Solution:**
Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
# or
BATCH_SIZE = 8   # Even smaller
```

### Issue 3: "No images found" or "Empty dataset"

**Solution:**
- Check folder structure matches exactly
- Ensure images are in correct folders
- Check image file extensions are supported
- Verify folder names match: `elephant_kolam` and `not_recognized`

### Issue 4: Training is very slow

**Solutions:**
1. **Use GPU:** Install CUDA-enabled PyTorch (see Step 3)
2. **Reduce batch size:** Set `BATCH_SIZE = 16` in `config.py`
3. **Reduce image size:** Set `IMAGE_SIZE = (128, 128)` in `config.py`
4. **Reduce epochs:** Set `EPOCHS = 30` in `config.py`

### Issue 5: Poor accuracy

**Solutions:**
1. **Add more training images** (aim for 200+ per class)
2. **Ensure balanced classes** (similar number of images)
3. **Check image quality** (clear, well-lit images work best)
4. **Train longer:** Increase `EPOCHS` in `config.py`
5. **Adjust learning rate:** Try `LEARNING_RATE = 0.0005` in `config.py`

### Issue 6: "FileNotFoundError" for model checkpoint

**Solution:**
- Train the model first using `python train.py`
- Or specify model path: `python inference.py image.jpg --model models\checkpoint_epoch_10.pth`

## 📊 Monitoring Training

**Watch for:**
- **Training loss decreasing** - Good sign
- **Validation loss decreasing** - Model learning
- **Validation accuracy increasing** - Model improving
- **Early stopping** - Model stopped improving (normal)

**Good training signs:**
- Training loss: Starts high (2-3), decreases to 0.1-0.5
- Validation accuracy: Starts low (50%), increases to 80-95%
- Both train and validation metrics improving together

## 🎯 Next Steps After Training

1. **Test on new images:**
   ```bash
   python inference.py new_image.jpg
   ```

2. **Extract features for ViT:**
   ```bash
   python extract_features.py --input data\train\elephant_kolam\ --output features\
   ```

3. **Scale to more classes:**
   - Add new folders: `peacock_kolam`, `geometric_kolam`, etc.
   - Update `NUM_CLASSES` and `CLASS_LABELS` in `config.py`
   - Retrain: `python train.py`

## 💡 Tips for Best Results

1. **Data Quality:**
   - Use clear, well-lit images
   - Include variety (different angles, lighting, backgrounds)
   - Remove blurry or corrupted images

2. **Data Quantity:**
   - Minimum: 50 images per class
   - Recommended: 200+ images per class
   - More is better!

3. **Training:**
   - Let training complete (don't stop early)
   - Monitor validation accuracy
   - Use GPU if available

4. **Testing:**
   - Test on images NOT in training set
   - Test on various conditions
   - Check confidence scores

## 🆘 Getting Help

If you encounter issues:

1. **Check error messages** - They usually tell you what's wrong
2. **Verify folder structure** - Must match exactly
3. **Check Python version** - Must be 3.7+
4. **Verify dependencies** - Run `pip list` to see installed packages
5. **Check file paths** - Use forward slashes or raw strings in Python

## ✅ Quick Checklist

Before training, ensure:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset organized correctly (`data\train\elephant_kolam\` and `data\train\not_recognized\`)
- [ ] At least 50 images per class
- [ ] GPU setup (optional but recommended)
- [ ] `config.py` settings adjusted (if needed)

---

**You're all set!** Start with `python train.py` and let the model learn from your kolam images.

