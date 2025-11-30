"""
================================================================================
FEATURE EXTRACTION UTILITY
================================================================================
Purpose: Extract feature maps and feature vectors from kolam images
         for use with Vision Transformer (ViT) or other downstream tasks
================================================================================
"""

import torch
from pathlib import Path
from PIL import Image
import argparse
import numpy as np

# Import project modules
from config import *
from models.kolam_cnn import create_model
from data.dataset import get_transforms
from inference import load_model


def extract_features_batch(
    model: torch.nn.Module,
    image_paths: list,
    device: torch.device,
    output_dir: Path
):
    """
    ============================================================================
    EXTRACT FEATURES FROM MULTIPLE IMAGES
    ============================================================================
    
    Purpose: Batch process images to extract features for ViT integration
    
    Args:
        model: Trained CNN model
        image_paths: List of image file paths
        device: Torch device
        output_dir: Directory to save extracted features
    ============================================================================
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = get_transforms(IMAGE_SIZE, is_training=False, use_augmentation=False)
    
    print(f"Processing {len(image_paths)} images...")
    
    all_features = []
    all_feature_maps = []
    all_image_paths = []
    
    model.eval()
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Extract features
                features = model.extract_features(image_tensor)
                
                # Store features
                all_features.append(features['feature_vector'].cpu())
                all_feature_maps.append(features['feature_map'].cpu())
                all_image_paths.append(str(image_path))
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    # ============================================================================
    # SAVE FEATURES
    # ============================================================================
    # Stack all features
    feature_vectors = torch.cat(all_features, dim=0)
    feature_maps = torch.cat(all_feature_maps, dim=0)
    
    # Save as PyTorch tensors
    features_file = output_dir / "extracted_features.pt"
    torch.save({
        'feature_vectors': feature_vectors,  # Shape: (N, feature_dim)
        'feature_maps': feature_maps,        # Shape: (N, channels, H, W)
        'image_paths': all_image_paths,
        'feature_dim': FEATURE_DIM,
        'num_images': len(all_image_paths)
    }, features_file)
    
    # Also save as numpy arrays for compatibility
    np_features_file = output_dir / "extracted_features.npz"
    np.savez(
        np_features_file,
        feature_vectors=feature_vectors.numpy(),
        feature_maps=feature_maps.numpy(),
        image_paths=all_image_paths
    )
    
    print(f"\nFeatures saved:")
    print(f"  - PyTorch format: {features_file}")
    print(f"  - NumPy format: {np_features_file}")
    print(f"  - Feature vectors shape: {feature_vectors.shape}")
    print(f"  - Feature maps shape: {feature_maps.shape}")


def main():
    """
    ============================================================================
    MAIN FEATURE EXTRACTION FUNCTION
    ============================================================================
    """
    parser = argparse.ArgumentParser(description='Extract features from kolam images')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image file or directory containing images')
    parser.add_argument('--model', type=str, default=str(MODEL_CHECKPOINT),
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=str(FEATURES_DIR),
                       help='Output directory for extracted features')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FEATURE EXTRACTION FOR ViT INTEGRATION")
    print("=" * 80)
    
    # ============================================================================
    # SETUP
    # ============================================================================
    device = torch.device(DEVICE)
    print(f"Using device: {device}\n")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    model = load_model(model_path, device)
    print()
    
    # ============================================================================
    # GET IMAGE PATHS
    # ============================================================================
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = [p for p in input_path.rglob('*') 
                      if p.suffix.lower() in image_extensions]
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    if not image_paths:
        print(f"Error: No images found in {input_path}")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    
    # ============================================================================
    # EXTRACT FEATURES
    # ============================================================================
    output_dir = Path(args.output)
    extract_features_batch(model, image_paths, device, output_dir)
    
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nThese features can now be used for:")
    print("  - Vision Transformer (ViT) integration")
    print("  - Transfer learning")
    print("  - Feature visualization")
    print("  - Similarity search")
    print("=" * 80)


if __name__ == "__main__":
    main()

