"""
Damage Localization Module
Semantic segmentation for damage type classification
Architecture: DeepLabV3+ with EfficientNet-b5 encoder (joint model as per research paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Optional
import segmentation_models_pytorch as smp


# Damage type taxonomy (categorical - 3 damage types + no damage)
DAMAGE_TYPE_CLASSES = {
    0: 'no_damage',
    1: 'body_damage',       # Merged: dent, missing
    2: 'surface_damage',    # Merged: scratch, paint chips, corrosion
    3: 'deformity'          # Merged: crack, shatter
}

NUM_DAMAGE_CLASSES = len(DAMAGE_TYPE_CLASSES)


class DamageLocalizer(nn.Module):
    """
    Semantic segmentation model for damage localization.
    Uses DeepLabV3+ with EfficientNet-b5 encoder (joint model).
    
    Achieves 0.463 mIoU on OE Fleet test set and 0.392 mIoU on OEM test set.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize damage localizer.
        
        Args:
            model_path: Path to pretrained weights
            device: Device for inference
        """
        super(DamageLocalizer, self).__init__()
        
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Build DeepLabV3+ with EfficientNet-b5 encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b5',
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_DAMAGE_CLASSES,
            activation=None
        )
        
        # Load pretrained weights if provided
        if model_path:
            self.load_weights(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        self.class_names = DAMAGE_TYPE_CLASSES
    
    def load_weights(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"✓ Damage localizer weights loaded from {path}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for segmentation."""
        original_size = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, original_size
    
    def predict(self, image: Image.Image, return_mask=True, 
                return_probabilities=True) -> Dict:
        """
        Predict damage types in image.
        
        Args:
            image: PIL Image
            return_mask: Whether to return segmentation mask
            return_probabilities: Whether to return per-pixel probabilities
            
        Returns:
            Dictionary with:
                - mask: Segmentation mask (H x W) with class IDs
                - probabilities: Per-pixel probabilities (C x H x W)
                - damage_counts: Dictionary of detected damage types and pixel counts
                - has_damage: Boolean indicating if any damage detected
                - average_confidence: Average confidence across damaged pixels
        """
        original_size = image.size
        image_tensor, _ = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.model(image_tensor)
            else:
                logits = self.model(image_tensor)
            
            # Get probabilities and predictions
            probabilities = F.softmax(logits, dim=1)
            mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
            
            # Resize mask back to original size
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_resized = mask_pil.resize(original_size, Image.NEAREST)
            mask = np.array(mask_resized)
            
            # Resize probabilities
            probs_resized = F.interpolate(
                probabilities, 
                size=original_size[::-1],
                mode='bilinear',
                align_corners=False
            )
            probs_np = probs_resized.squeeze(0).cpu().numpy()
        
        # Count damage types detected
        unique_damages = np.unique(mask)
        damage_counts = {}
        damaged_probs = []
        
        for damage_id in unique_damages:
            if damage_id > 0:  # Exclude no_damage
                damage_name = self.class_names[damage_id]
                pixel_count = np.sum(mask == damage_id)
                damage_counts[damage_name] = int(pixel_count)
                
                # Get average confidence for this damage type
                damage_mask_bool = (mask == damage_id)
                damage_confidences = probs_np[damage_id][damage_mask_bool]
                damaged_probs.extend(damage_confidences.tolist())
        
        has_damage = len(damage_counts) > 0
        avg_confidence = np.mean(damaged_probs) * 100 if damaged_probs else 0.0
        
        result = {
            'has_damage': has_damage,
            'damage_counts': damage_counts,
            'average_confidence': round(avg_confidence, 2),
            'unique_damage_ids': unique_damages.tolist()
        }
        
        if return_mask:
            result['mask'] = mask
        
        if return_probabilities:
            result['probabilities'] = probs_np
        
        return result
    
    def get_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert damage mask to colored visualization.
        
        Args:
            mask: Damage segmentation mask (H x W)
            
        Returns:
            RGB colored mask
        """
        colors = {
            0: [0, 0, 0],         # no_damage - black
            1: [255, 0, 0],       # body_damage - blue
            2: [0, 255, 0],       # surface_damage - green
            3: [0, 0, 255]        # deformity - red
        }
        
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'architecture': 'DeepLabV3Plus',
            'encoder': 'EfficientNet-b5',
            'num_classes': NUM_DAMAGE_CLASSES,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'device': str(self.device),
            'input_size': (512, 512),
            'class_names': self.class_names
        }


class JointLocalizer(nn.Module):
    """
    Joint model for simultaneous vehicle part and damage localization.
    Shares encoder-decoder with separate segmentation heads.
    
    According to paper: joint training improves damage detection but 
    independent training is better for part localization.
    """
    
    def __init__(self, device: Optional[str] = None):
        super(JointLocalizer, self).__init__()
        
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Shared encoder-decoder
        self.shared_model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b5',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,  # Dummy, will use custom heads
            activation=None
        )
        
        # Get decoder output channels
        decoder_channels = 256  # DeepLabV3+ decoder output
        
        # Separate segmentation heads
        self.part_head = nn.Conv2d(decoder_channels, 14, kernel_size=1)  # 14 part classes
        self.damage_head = nn.Conv2d(decoder_channels, 4, kernel_size=1)  # 4 damage classes
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass returns both part and damage predictions.
        
        Returns:
            Tuple of (part_logits, damage_logits)
        """
        # Get shared features
        features = self.shared_model.encoder(x)
        decoder_output = self.shared_model.decoder(*features)
        
        # Separate heads
        part_logits = self.part_head(decoder_output)
        damage_logits = self.damage_head(decoder_output)
        
        # Upsample to input size
        part_logits = F.interpolate(part_logits, size=x.shape[2:], 
                                    mode='bilinear', align_corners=False)
        damage_logits = F.interpolate(damage_logits, size=x.shape[2:], 
                                      mode='bilinear', align_corners=False)
        
        return part_logits, damage_logits


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        num_classes = predictions.shape[1]
        predictions = F.softmax(predictions, dim=1)
        
        targets_one_hot = F.one_hot(targets.long(), num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        predictions = predictions.reshape(predictions.shape[0], num_classes, -1)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0], num_classes, -1)
        
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


def train_damage_localizer(train_loader, val_loader, epochs=50, lr=0.0001,
                           save_path='vehicle_damage_model.pth', joint=True):
    """
    Training function for damage localizer.
    
    Args:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation  
        epochs: Number of epochs
        lr: Learning rate
        save_path: Path to save model
        joint: Whether to use joint training with part localization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if joint:
        model = JointLocalizer().to(device)
    else:
        model = DamageLocalizer().to(device)
    
    model.train()
    
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs-warmup_epochs
    )
    
    best_val_iou = 0.0
    
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup_epochs
        
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            if joint:
                images, part_masks, damage_masks = batch
                images = images.to(device)
                part_masks = part_masks.to(device)
                damage_masks = damage_masks.to(device)
                
                optimizer.zero_grad()
                part_logits, damage_logits = model(images)
                
                loss_part = criterion(part_logits, part_masks)
                loss_damage = criterion(damage_logits, damage_masks)
                loss = loss_part + loss_damage
            else:
                images, damage_masks = batch
                images, damage_masks = images.to(device), damage_masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, damage_masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_iou = compute_damage_iou(model, val_loader, device, joint)
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Val mIoU: {val_iou:.4f}')
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'joint': joint
            }, save_path)
            print(f'✓ Best model saved with mIoU: {val_iou:.4f}')
    
    print(f'\nTraining completed. Best mIoU: {best_val_iou:.4f}')
    return model


def compute_damage_iou(model, data_loader, device, joint=False):
    """Compute mean IoU for damage classes."""
    model.eval()
    ious = []
    
    with torch.no_grad():
        for batch in data_loader:
            if joint:
                images, _, damage_masks = batch
                images = images.to(device)
                damage_masks = damage_masks.to(device)
                _, outputs = model(images)
            else:
                images, damage_masks = batch
                images, damage_masks = images.to(device), damage_masks.to(device)
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            for class_id in range(1, NUM_DAMAGE_CLASSES):
                pred_mask = (preds == class_id)
                true_mask = (damage_masks == class_id)
                
                intersection = (pred_mask & true_mask).float().sum()
                union = (pred_mask | true_mask).float().sum()
                
                if union > 0:
                    iou = intersection / union
                    ious.append(iou.item())
    
    return np.mean(ious) if ious else 0.0


if __name__ == '__main__':
    # Example usage
    localizer = DamageLocalizer()
    print("Damage Localizer initialized successfully")
    print(localizer.get_model_info())
