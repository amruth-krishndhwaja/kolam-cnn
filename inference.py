"""
================================================================================
INFERENCE SCRIPT FOR KOLAM CLASSIFICATION
================================================================================
Purpose: Classify a single kolam image using the trained model
         Supports both single image and batch inference
================================================================================
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import argparse
import sys

# Import project modules
from config import *
from models.kolam_cnn import create_model
from data.dataset import get_transforms


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
    ============================================================================
    LOAD TRAINED MODEL FROM CHECKPOINT
    ============================================================================
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved configuration
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    base_channels = checkpoint.get('base_channels', BASE_CHANNELS)
    
    model = create_model(
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_rate=DROPOUT_RATE,
        use_batch_norm=USE_BATCH_NORM,
        feature_dim=FEATURE_DIM
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  - Classes: {num_classes}")
    print(f"  - Base channels: {base_channels}")
    if 'accuracy' in checkpoint:
        print(f"  - Training accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    ============================================================================
    PREPROCESS IMAGE FOR INFERENCE
    ============================================================================
    """
    # Load and preprocess image
    transform = get_transforms(IMAGE_SIZE, is_training=False, use_augmentation=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


def classify_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_labels: dict,
    confidence_threshold: float = 0.5
) -> dict:
    """
    ============================================================================
    CLASSIFY IMAGE
    ============================================================================
    
    Returns:
        Dictionary with:
            - predicted_class: Class name
            - confidence: Confidence score
            - probabilities: All class probabilities
    ============================================================================
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        # Get class name
        predicted_class = class_labels.get(predicted_idx, f"class_{predicted_idx}")
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        prob_dict = {class_labels.get(i, f"class_{i}"): float(prob) 
                    for i, prob in enumerate(all_probs)}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict
        }


def main():
    """
    ============================================================================
    MAIN INFERENCE FUNCTION
    ============================================================================
    """
    parser = argparse.ArgumentParser(description='Classify kolam image')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default=str(MODEL_CHECKPOINT),
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for classification')
    parser.add_argument('--extract-features', action='store_true',
                       help='Extract and save feature maps')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KOLAM CLASSIFICATION - INFERENCE")
    print("=" * 80)
    
    # ============================================================================
    # SETUP DEVICE
    # ============================================================================
    device = torch.device(DEVICE)
    print(f"Using device: {device}\n")
    
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    model = load_model(model_path, device)
    print()
    
    # ============================================================================
    # LOAD AND PREPROCESS IMAGE
    # ============================================================================
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    image_tensor = preprocess_image(image_path, device)
    print()
    
    # ============================================================================
    # CLASSIFY IMAGE
    # ============================================================================
    print("=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    
    result = classify_image(model, image_tensor, CLASS_LABELS, args.threshold)
    
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nAll Class Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob * 100:.2f}%")
    
    # ============================================================================
    # FEATURE EXTRACTION (IF REQUESTED)
    # ============================================================================
    if args.extract_features:
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES")
        print("=" * 80)
        
        features = model.extract_features(image_tensor)
        
        # Save feature map
        feature_map = features['feature_map']
        feature_vector = features['feature_vector']
        
        print(f"Feature map shape: {feature_map.shape}")
        print(f"Feature vector shape: {feature_vector.shape}")
        
        # Save features
        feature_file = FEATURES_DIR / f"{image_path.stem}_features.pt"
        torch.save({
            'feature_map': feature_map.cpu(),
            'feature_vector': feature_vector.cpu(),
            'image_path': str(image_path)
        }, feature_file)
        print(f"Features saved to {feature_file}")
    
    # ============================================================================
    # FINAL RESULT
    # ============================================================================
    print("\n" + "=" * 80)
    if result['confidence'] >= args.threshold:
        print(f"✓ Image classified as: {result['predicted_class']}")
    else:
        print(f"⚠ Low confidence ({result['confidence'] * 100:.2f}%). Classification uncertain.")
    print("=" * 80)


if __name__ == "__main__":
    main()

