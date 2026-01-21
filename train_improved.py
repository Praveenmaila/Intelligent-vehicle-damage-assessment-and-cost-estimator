# -------------------------------
# Enhanced Vehicle Damage Detection Training
# Works with real datasets and internet images
# Includes data augmentation and transfer learning
# -------------------------------

import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# -------------------------------
# Enhanced Dataset Class with Augmentation
# -------------------------------
class VehicleDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_training=True):
        # Filter existing files
        self.dataframe = dataframe.copy()
        self.dataframe['exists'] = self.dataframe['image'].apply(
            lambda x: os.path.exists(x) or os.path.exists(os.path.join('datasets', x))
        )
        
        missing = (~self.dataframe['exists']).sum()
        if missing > 0:
            print(f"⚠️  Warning: {missing} images not found, skipping...")
        
        self.dataframe = self.dataframe[self.dataframe['exists']].reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training
        
        # Advanced augmentation for training
        if is_training:
            self.augmentation = A.Compose([
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.GaussNoise(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.augmentation = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image']
        
        # Try multiple path combinations
        if not os.path.exists(img_path):
            img_path = os.path.join('datasets', img_path)
        
        label = self.dataframe.iloc[idx]['label']
        
        # Load image as numpy array for albumentations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        augmented = self.augmentation(image=image)
        image = augmented['image']
        
        return image, label

# -------------------------------
# Class Mapping - Comprehensive Damage Types
# -------------------------------
CLASS_MAPPING = {
    'unknown': 0,
    'minor_dent': 1,
    'major_dent': 2,
    'minor_scratch': 3,
    'major_scratch': 4,
    'glass_shatter': 5,
    'lamp_broken': 6,
    'bumper_dent': 7,
    
    # Extended mapping for detailed classification
    'head_lamp': 6,
    'rear_lamp': 6,
    'tail_lamp': 6,
    'front_bumper_dent': 7,
    'rear_bumper_dent': 7,
    'front_bumper_scratch': 3,
    'rear_bumper_scratch': 3,
    'door_dent': 2,
    'door_scratch': 3,
    'hood_dent': 2,
    'hood_scratch': 3,
    'trunk_dent': 1,
    'trunk_scratch': 3,
    'fender_dent': 1,
    'fender_scratch': 3,
    'windshield_crack': 5,
    'windshield_shatter': 5,
    'side_window_crack': 5,
    'side_window_shatter': 5,
    'rear_window_crack': 5,
    'rear_window_shatter': 5,
    'side_mirror_crack': 6,
    'side_mirror_broken': 6,
    'wheel_rim_scratch': 3,
    'wheel_rim_bent': 2,
    'tire_damage': 1,
    'paint_peel': 4,
    'rust_damage': 4,
    'panel_misalignment': 2,
    'grille_damage': 1,
}

# Cost mapping based on damage severity (INR)
COST_MAPPING = {
    0: 0,       # unknown
    1: 8000,    # minor_dent
    2: 25000,   # major_dent
    3: 6000,    # minor_scratch
    4: 18000,   # major_scratch
    5: 30000,   # glass_shatter
    6: 12000,   # lamp_broken
    7: 20000,   # bumper_dent
}

# -------------------------------
# Model Creation with Transfer Learning
# -------------------------------
def create_model(num_classes=8, pretrained=True, model_type='resnet50'):
    """Create model with various backbones"""
    
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_type == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    else:  # resnet18 - faster
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=20, device='cuda'):
    """Training loop with validation"""
    
    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            train_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                val_bar.set_postfix({'loss': loss.item()})
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            print(f"✓ New best model! Accuracy: {best_acc:.4f}")
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

# -------------------------------
# Main Training Script
# -------------------------------
def main():
    print("="*70)
    print("🚗 Vehicle Damage Detection Training")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 8
    MODEL_TYPE = 'resnet50'  # Options: resnet18, resnet50, efficientnet_b0, mobilenet_v3
    
    # Check for dataset
    csv_path = 'datasets/data.csv'
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: Dataset not found at {csv_path}")
        print("\n📥 Please run: python download_dataset.py")
        print("   Or manually create datasets/data.csv with your images")
        return
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} images")
    
    # Map classes to labels
    df['label'] = df['classes'].map(CLASS_MAPPING).fillna(0).astype(int)
    
    print("\n📊 Class distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['label'] if len(df) > NUM_CLASSES else None
    )
    
    print(f"\n✓ Train: {len(train_df)} | Validation: {len(val_df)}")
    
    # Create datasets
    train_dataset = VehicleDamageDataset(train_df, is_training=True)
    val_dataset = VehicleDamageDataset(val_df, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # Create model
    print(f"\n🏗️  Creating {MODEL_TYPE} model...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=True, model_type=MODEL_TYPE)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train model
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    model, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Save model
    save_path = 'vehicle_damage_model.pth'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'model_type': MODEL_TYPE,
        'num_classes': NUM_CLASSES,
        'best_accuracy': best_acc.item(),
        'class_mapping': CLASS_MAPPING,
        'cost_mapping': COST_MAPPING,
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    
    print("\n" + "="*70)
    print(f"✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best_acc:.2%}")
    print(f"   Model saved to: {save_path}")
    print("="*70)
    print("\n📝 Next steps:")
    print("   1. Run: python app.py")
    print("   2. Open browser: http://localhost:5000")
    print("   3. Upload vehicle damage images")
    print("="*70)

if __name__ == '__main__':
    main()
