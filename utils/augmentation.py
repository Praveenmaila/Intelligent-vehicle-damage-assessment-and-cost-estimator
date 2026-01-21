"""
Data Augmentation Pipeline
Extensive augmentations as specified in research paper
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_training_augmentation(image_size=512):
    """
    Training augmentation pipeline.
    
    Includes all augmentations mentioned in research paper:
    - Random cropping
    - Horizontal flips
    - Perspective transforms
    - Gaussian noise
    - Blur/sharpen
    - Brightness/contrast
    - Hue/saturation adjustments
    - Gamma correction
    
    Args:
        image_size: Target image size (default 512)
        
    Returns:
        Albumentations composition
    """
    train_transform = A.Compose([
        # Geometric transforms
        A.RandomResizedCrop(height=image_size, width=image_size, 
                           scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                          rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, 
                          value=0, p=0.5),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, 
                     pad_mode=cv2.BORDER_CONSTANT, p=0.3),
        
        # Color/intensity transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, 
                                      contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, 
                                val_shift_limit=20, p=1.0)
        ], p=0.8),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0)
        ], p=0.3),
        
        # Quality degradation (simulate real-world conditions)
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.7, scale_max=0.95, p=1.0)
        ], p=0.2),
        
        # Sharpen
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
        
        # Normalize using ImageNet stats
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform


def get_validation_augmentation(image_size=512):
    """
    Validation augmentation (only resize and normalize).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return val_transform


def get_test_time_augmentation(image_size=512):
    """
    Test time augmentation for improved inference.
    
    Returns multiple augmented versions of same image,
    predictions are then averaged.
    
    Args:
        image_size: Target image size
        
    Returns:
        List of augmentation compositions
    """
    # Original
    tta_transforms = [
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Horizontal flip
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Brightness adjustment
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Scale variation
        A.Compose([
            A.RandomScale(scale_limit=0.1, p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    
    return tta_transforms


# For classification tasks (vehicle detection)
def get_classification_augmentation(image_size=224):
    """
    Augmentation for vehicle detection (classification) task.
    
    Args:
        image_size: Target image size (224 for MobileNet)
        
    Returns:
        Training and validation transforms
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, 
                           scale=(0.8, 1.0), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


if __name__ == '__main__':
    # Test augmentations
    print("Data Augmentation Module")
    print("=" * 60)
    
    train_aug = get_training_augmentation()
    val_aug = get_validation_augmentation()
    
    print("Training augmentation pipeline:")
    print(train_aug)
    print("\nValidation augmentation pipeline:")
    print(val_aug)
    
    print("\n✓ Augmentation module ready")
