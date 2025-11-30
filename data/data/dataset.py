"""
================================================================================
DATA LOADING AND PREPROCESSING
================================================================================
Purpose: Custom dataset class and data augmentation for kolam images
         Handles image loading, preprocessing, and augmentation
================================================================================
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class KolamDataset(Dataset):
    """
    ============================================================================
    KOLAM IMAGE DATASET CLASS
    ============================================================================
    
    Purpose:
    - Load kolam images from directory structure
    - Apply data augmentation for better generalization
    - Handle class labels automatically from folder names
    
    Directory Structure Expected:
    data/
        train/
            elephant_kolam/
                image1.jpg
                image2.jpg
                ...
            not_recognized/
                image1.jpg
                ...
        test/
            elephant_kolam/
                ...
            not_recognized/
                ...
    ============================================================================
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[Callable] = None,
        is_training: bool = True
    ):
        """
        Initialize KolamDataset
        
        Args:
            data_dir: Path to directory containing class subdirectories
            transform: Optional transform to be applied on images
            is_training: Whether this is training data (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # ========================================================================
        # LOAD IMAGE PATHS AND LABELS
        # ========================================================================
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get class names from subdirectories
        class_dirs = [d for d in self.data_dir.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        class_dirs.sort()  # Ensure consistent ordering
        
        # Create class mappings
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Load all images from this class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.images.append(img_path)
                    self.labels.append(idx)
        
        print(f"Loaded {len(self.images)} images from {len(class_dirs)} classes")
        print(f"Classes: {self.class_to_idx}")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # ========================================================================
        # LOAD AND PREPROCESS IMAGE
        # ========================================================================
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                image = transforms.ToTensor()(Image.new('RGB', (224, 224), color='black'))
            return image, label


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    ============================================================================
    GET DATA TRANSFORMS
    ============================================================================
    
    Purpose: Create appropriate transforms for training/validation
    
    Training transforms include:
    - Random horizontal flip
    - Random rotation
    - Color jitter
    - Random affine transformations
    
    Validation transforms:
    - Only resizing and normalization
    ============================================================================
    """
    
    if is_training and use_augmentation:
        # ========================================================================
        # TRAINING TRANSFORMS WITH AUGMENTATION
        # ========================================================================
        transform = transforms.Compose([
            transforms.Resize((image_size[0] + 32, image_size[1] + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # ========================================================================
        # VALIDATION/TEST TRANSFORMS (NO AUGMENTATION)
        # ========================================================================
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(
    train_dir: Path,
    test_dir: Optional[Path] = None,
    validation_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    ============================================================================
    CREATE DATA LOADERS
    ============================================================================
    
    Purpose: Create train, validation, and test data loaders
    
    Args:
        train_dir: Directory containing training data
        test_dir: Optional directory containing test data
        validation_split: Fraction of training data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        use_augmentation: Whether to use data augmentation for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    ============================================================================
    """
    
    # ========================================================================
    # CREATE TRAINING DATASET
    # ========================================================================
    train_transform = get_transforms(image_size, is_training=True, use_augmentation=use_augmentation)
    train_dataset = KolamDataset(train_dir, transform=train_transform, is_training=True)
    
    # ========================================================================
    # SPLIT TRAINING DATA INTO TRAIN AND VALIDATION
    # ========================================================================
    dataset_size = len(train_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # ========================================================================
    # CREATE VALIDATION TRANSFORMS (NO AUGMENTATION)
    # ========================================================================
    val_transform = get_transforms(image_size, is_training=False, use_augmentation=False)
    val_subset.dataset.transform = val_transform
    
    # ========================================================================
    # CREATE DATA LOADERS
    # ========================================================================
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ========================================================================
    # CREATE TEST DATA LOADER (IF PROVIDED)
    # ========================================================================
    test_loader = None
    if test_dir and test_dir.exists():
        test_transform = get_transforms(image_size, is_training=False, use_augmentation=False)
        test_dataset = KolamDataset(test_dir, transform=test_transform, is_training=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

