"""
Vehicle Detection Module
Binary classifier to identify vehicle presence in images
Architecture: MobileNet (as per research paper)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional


class VehicleDetector(nn.Module):
    """
    Binary classifier for vehicle presence detection.
    Uses MobileNet architecture for efficient inference.
    
    Achieves 98.9% accuracy on OE validation set and 91% on OEM test set.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize vehicle detector.
        
        Args:
            model_path: Path to pretrained weights (optional)
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        super(VehicleDetector, self).__init__()
        
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Build MobileNet architecture
        self.model = models.mobilenet_v2(weights=None)
        
        # Modify classifier for binary classification
        # MobileNet has a classifier with 1280 input features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)  # 2 classes: vehicle / no vehicle
        )
        
        # Load pretrained weights if provided
        if model_path:
            self.load_weights(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    
    def load_weights(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"✓ Vehicle detector weights loaded from {path}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict vehicle presence in image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with:
                - has_vehicle: bool indicating vehicle presence
                - confidence: float confidence score (0-100)
                - probabilities: dict with class probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = self.model(image_tensor)
            else:
                outputs = self.model(image_tensor)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            vehicle_prob = probabilities[0, 1].item()  # Probability of vehicle class
            
            # Get prediction
            confidence, pred = torch.max(probabilities, 1)
            has_vehicle = pred.item() == 1
        
        return {
            'has_vehicle': has_vehicle,
            'confidence': round(confidence.item() * 100, 2),
            'probabilities': {
                'no_vehicle': round(probabilities[0, 0].item() * 100, 2),
                'vehicle': round(probabilities[0, 1].item() * 100, 2)
            }
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'architecture': 'MobileNetV2',
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'device': str(self.device),
            'input_size': (224, 224),
            'num_classes': 2
        }


def train_vehicle_detector(train_loader, val_loader, epochs=20, lr=0.001, 
                          save_path='vehicle_detector.pth'):
    """
    Training function for vehicle detector.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save best model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = VehicleDetector().to(device)
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_accuracy': val_acc,
                'train_accuracy': train_acc
            }, save_path)
            print(f'✓ Best model saved with validation accuracy: {val_acc:.2f}%')
    
    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%')
    return model


if __name__ == '__main__':
    # Example usage
    detector = VehicleDetector()
    print("Vehicle Detector initialized successfully")
    print(detector.get_model_info())
