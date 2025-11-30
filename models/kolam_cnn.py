"""
================================================================================
KOLAM CNN MODEL ARCHITECTURE
================================================================================
Purpose: Scalable CNN architecture for kolam image classification
         Designed for easy expansion to multiple kolam types
         Includes feature extraction capability for ViT integration
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class KolamCNN(nn.Module):
    """
    ============================================================================
    KOLAM CLASSIFICATION CNN MODEL
    ============================================================================
    
    Architecture:
    - Input: RGB images (224x224)
    - Feature extraction: Multiple convolutional blocks with batch normalization
    - Classification head: Fully connected layers with dropout
    - Feature extraction: Intermediate features available for ViT integration
    
    Scalability:
    - Easy to modify num_classes for additional kolam types
    - Feature extraction at multiple layers
    - Modular design for architecture modifications
    ============================================================================
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 64,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        feature_dim: int = 512
    ):
        """
        Initialize KolamCNN model
        
        Args:
            num_classes: Number of output classes (default: 2 for elephant_kolam/not_recognized)
            base_channels: Starting number of channels in first conv layer
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
            feature_dim: Dimension of feature vector for extraction
        """
        super(KolamCNN, self).__init__()
        
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.feature_dim = feature_dim
        
        # ========================================================================
        # FEATURE EXTRACTION LAYERS (Convolutional Blocks)
        # ========================================================================
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 2: Deeper features
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Mid-level features
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4) if use_batch_norm else nn.Identity()
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: High-level features
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels * 8) if use_batch_norm else nn.Identity()
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5: Final feature extraction (for feature map extraction)
        self.conv_final = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1)
        self.bn_final = nn.BatchNorm2d(base_channels * 8) if use_batch_norm else nn.Identity()
        self.relu_final = nn.ReLU(inplace=True)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========================================================================
        # CLASSIFICATION HEAD
        # ========================================================================
        
        # Calculate flattened size after convolutions
        # For 224x224 input: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1 (after adaptive pool)
        self.fc1 = nn.Linear(base_channels * 8, feature_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(feature_dim, feature_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(feature_dim // 2, num_classes)
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            return_features: If True, return intermediate features for ViT integration
            
        Returns:
            If return_features=False: Logits tensor (batch_size, num_classes)
            If return_features=True: Tuple of (logits, feature_map, feature_vector)
        """
        # ========================================================================
        # FEATURE EXTRACTION
        # ========================================================================
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Block 5 - Final feature extraction
        feature_map = self.conv_final(x)  # Save feature map before pooling
        feature_map = self.bn_final(feature_map)
        feature_map = self.relu_final(feature_map)
        
        # Global average pooling for classification
        x = self.pool_final(feature_map)
        x = x.view(x.size(0), -1)  # Flatten
        
        # ========================================================================
        # FEATURE VECTOR (for ViT integration)
        # ========================================================================
        feature_vector = self.fc1(x)
        feature_vector = F.relu(feature_vector)
        
        # ========================================================================
        # CLASSIFICATION
        # ========================================================================
        x = self.dropout1(feature_vector)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc3(x)
        
        if return_features:
            return logits, feature_map, feature_vector
        return logits
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at multiple layers for analysis or ViT integration
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Dictionary containing:
                - 'feature_map': Spatial feature map from conv_final (before pooling)
                - 'feature_vector': Flattened feature vector (after fc1)
                - 'logits': Final classification logits
        """
        self.eval()
        with torch.no_grad():
            logits, feature_map, feature_vector = self.forward(x, return_features=True)
            
        return {
            'feature_map': feature_map,      # Shape: (batch, channels, H, W)
            'feature_vector': feature_vector, # Shape: (batch, feature_dim)
            'logits': logits                  # Shape: (batch, num_classes)
        }


def create_model(num_classes: int = 2, **kwargs) -> KolamCNN:
    """
    Factory function to create KolamCNN model
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to KolamCNN
        
    Returns:
        Initialized KolamCNN model
    """
    return KolamCNN(num_classes=num_classes, **kwargs)

