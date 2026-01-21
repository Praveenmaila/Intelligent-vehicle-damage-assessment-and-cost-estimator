# -------------------------------
# Vehicle Damage Detection Inference Module
# Integrates ResNet50 for classification and YOLO for bounding box detection
# Based on Kaggle implementations
# -------------------------------

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import os
from typing import Dict, List, Tuple, Optional
import cv2

class VehicleDamageDetector:
    """
    Hybrid vehicle damage detection system combining:
    - ResNet50: Damage type classification and severity assessment
    - YOLO11: Bounding box detection for damaged regions
    """
    
    def __init__(self, 
                 resnet_model_path: str = 'vehicle_damage_model.pth',
                 yolo_model_path: str = 'yolo_vehicle_damage.pt',
                 device: Optional[str] = None):
        """
        Initialize the detector with pretrained models.
        
        Args:
            resnet_model_path: Path to ResNet50 checkpoint
            yolo_model_path: Path to YOLO model weights
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = torch.device(device if device else 
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"🔧 Initializing Vehicle Damage Detector on {self.device}")
        
        # Initialize models
        self.resnet_model = None
        self.yolo_model = None
        self.class_mapping = None
        self.cost_mapping = None
        self.resnet_loaded = False
        self.yolo_loaded = False
        
        # Load ResNet50 for classification
        if os.path.exists(resnet_model_path):
            self._load_resnet_model(resnet_model_path)
        else:
            print(f"⚠️  ResNet model not found at {resnet_model_path}")
            print("   Classification will be unavailable.")
        
        # Load YOLO for object detection
        if os.path.exists(yolo_model_path):
            self._load_yolo_model(yolo_model_path)
        else:
            print(f"⚠️  YOLO model not found at {yolo_model_path}")
            print("   Using pretrained YOLOv11 as fallback...")
            try:
                self.yolo_model = YOLO('yolo11n.pt')  # Nano model for speed
                self.yolo_loaded = True
                print("✓ YOLOv11n loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load YOLO: {e}")
        
        # Define image transformations for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # Damage type to color mapping for visualization (BGR format for OpenCV)
        self.damage_colors = {
            'unknown': (128, 128, 128),              # Gray
            'head_lamp': (0, 0, 255),                # Red
            'rear_lamp': (0, 50, 255),               # Orange-Red
            'tail_lamp': (0, 100, 255),              # Light Red
            'front_bumper_dent': (255, 0, 0),        # Blue
            'rear_bumper_dent': (255, 100, 0),       # Light Blue
            'front_bumper_scratch': (255, 128, 0),   # Sky Blue
            'rear_bumper_scratch': (255, 200, 0),    # Light Sky Blue
            'door_dent': (0, 255, 0),                # Green
            'door_scratch': (100, 255, 100),         # Light Green
            'hood_dent': (0, 255, 255),              # Yellow
            'hood_scratch': (100, 255, 255),         # Light Yellow
            'trunk_dent': (0, 200, 255),             # Gold
            'trunk_scratch': (100, 200, 255),        # Light Gold
            'fender_dent': (255, 0, 255),            # Magenta
            'fender_scratch': (255, 100, 255),       # Light Magenta
            'windshield_crack': (128, 0, 128),       # Purple
            'windshield_shatter': (200, 0, 200),     # Light Purple
            'side_window_crack': (255, 0, 128),      # Pink
            'side_window_shatter': (255, 100, 180),  # Light Pink
            'rear_window_crack': (128, 0, 255),      # Violet
            'rear_window_shatter': (180, 100, 255),  # Light Violet
            'side_mirror_crack': (0, 128, 255),      # Orange
            'side_mirror_broken': (0, 180, 255),     # Light Orange
            'wheel_rim_scratch': (128, 255, 0),      # Lime
            'wheel_rim_bent': (180, 255, 100),       # Light Lime
            'tire_damage': (64, 64, 64),             # Dark Gray
            'paint_peel': (200, 150, 100),           # Brown
            'rust_damage': (0, 100, 150),            # Rust Color
            'panel_misalignment': (255, 255, 0),     # Cyan
            'grille_damage': (128, 128, 255)         # Light Red
        }
    
    def _load_resnet_model(self, model_path: str):
        """Load ResNet50 model with pretrained weights."""
        try:
            print(f"📥 Loading ResNet50 from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            self.class_mapping = checkpoint.get('class_mapping', {})
            self.cost_mapping = checkpoint.get('cost_mapping', {})
            num_classes = len(self.class_mapping)
            
            # Build ResNet18 architecture (matching the trained model)
            self.resnet_model = models.resnet18(weights=None)
            # Modify final layer for custom classes
            self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, num_classes)
            
            # Load trained weights
            self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
            self.resnet_model = self.resnet_model.to(self.device)
            self.resnet_model.eval()
            
            self.resnet_loaded = True
            val_acc = checkpoint.get('validation_accuracy', 0)
            print(f"✓ ResNet18 loaded | Classes: {num_classes} | Val Acc: {val_acc:.2%}")
            
        except Exception as e:
            print(f"❌ Failed to load ResNet50: {e}")
            self.resnet_loaded = False
    
    def _load_yolo_model(self, model_path: str):
        """Load YOLO model for bounding box detection."""
        try:
            print(f"📥 Loading YOLO from {model_path}...")
            self.yolo_model = YOLO(model_path)
            self.yolo_loaded = True
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
            self.yolo_loaded = False
    
    def _classify_damage(self, image: Image.Image) -> Dict:
        """
        Classify damage type using ResNet50.
        
        Args:
            image: PIL Image
            
        Returns:
            Dict with damage_type, confidence, cost, class_id
        """
        if not self.resnet_loaded:
            return {
                'damage_type': 'unknown',
                'confidence': 0.0,
                'estimated_cost': 0,
                'class_id': -1
            }
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = self.resnet_model(image_tensor)
            else:
                outputs = self.resnet_model(image_tensor)
            
            # Get prediction
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            
            predicted_class = pred.item()
            confidence_score = confidence.item()
        
        # Map to class name
        class_name = self._get_class_name(predicted_class)
        estimated_cost = self.cost_mapping.get(predicted_class, 0)
        
        return {
            'damage_type': class_name,
            'confidence': round(confidence_score * 100, 2),
            'estimated_cost': estimated_cost,
            'class_id': predicted_class
        }
    
    def _detect_bounding_boxes(self, image: Image.Image, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Detect bounding boxes using YOLO.
        
        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection dictionaries with bbox, confidence, class
        """
        if not self.yolo_loaded:
            return []
        
        # Convert PIL to numpy array for YOLO
        img_array = np.array(image)
        
        # Run YOLO inference
        results = self.yolo_model(img_array, conf=confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(confidence * 100, 2),
                    'class': class_name,
                    'class_id': class_id
                })
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        for name, cid in self.class_mapping.items():
            if cid == class_id:
                return name
        return "unknown"
    
    def draw_detections(self, image: Image.Image, detections: List[Dict], 
                       damage_type: str = None) -> Image.Image:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: PIL Image
            detections: List of detection dictionaries
            damage_type: Overall damage type from classification
            
        Returns:
            Annotated PIL Image
        """
        # Create a copy to draw on
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Get color based on damage type
        color = self.damage_colors.get(damage_type, (0, 255, 0))
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class']
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            label = f"{class_name} {confidence:.1f}%"
            bbox = draw.textbbox((x1, y1 - 25), label, font=small_font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], 
                         fill=color)
            
            # Draw label text
            draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=small_font)
        
        # Add overall damage type at top if provided
        if damage_type and damage_type != 'unknown':
            header = f"Damage Type: {damage_type.replace('_', ' ').title()}"
            header_bbox = draw.textbbox((10, 10), header, font=font)
            draw.rectangle([header_bbox[0] - 5, header_bbox[1] - 5, 
                          header_bbox[2] + 5, header_bbox[3] + 5], 
                         fill=(0, 0, 0))
            draw.text((10, 10), header, fill=(255, 255, 255), font=font)
        
        return img_draw
    
    def predict(self, image: Image.Image, return_annotated: bool = True,
               confidence_threshold: float = 0.25) -> Dict:
        """
        Complete inference pipeline: classification + detection.
        
        Args:
            image: PIL Image
            return_annotated: Whether to return annotated image
            confidence_threshold: Minimum confidence for YOLO detections
            
        Returns:
            Dictionary containing:
                - damage_type: Classification result
                - confidence: Classification confidence
                - estimated_cost: Repair cost estimate
                - detections: List of bounding boxes
                - annotated_image: PIL Image with annotations (if requested)
        """
        # Step 1: Classify damage type
        classification = self._classify_damage(image)
        
        # Step 2: Detect bounding boxes
        detections = self._detect_bounding_boxes(image, confidence_threshold)
        
        # Step 3: Draw annotations if requested
        annotated_image = None
        if return_annotated:
            annotated_image = self.draw_detections(
                image, 
                detections, 
                classification['damage_type']
            )
        
        # Combine results
        result = {
            'damage_type': classification['damage_type'],
            'confidence': classification['confidence'],
            'estimated_cost': classification['estimated_cost'],
            'class_id': classification['class_id'],
            'detections': detections,
            'detection_count': len(detections)
        }
        
        if return_annotated:
            result['annotated_image'] = annotated_image
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'device': str(self.device),
            'resnet_loaded': self.resnet_loaded,
            'yolo_loaded': self.yolo_loaded,
            'num_classes': len(self.class_mapping) if self.class_mapping else 0,
            'cuda_available': torch.cuda.is_available(),
            'class_mapping': self.class_mapping,
            'cost_mapping': self.cost_mapping
        }


# -------------------------------
# Standalone Inference Function for Quick Use
# -------------------------------
def predict_damage(image_path: str, 
                  resnet_model: str = 'vehicle_damage_model.pth',
                  yolo_model: str = 'yolo_vehicle_damage.pt',
                  save_output: bool = True,
                  output_path: str = None) -> Dict:
    """
    Convenience function for standalone inference.
    
    Args:
        image_path: Path to input image
        resnet_model: Path to ResNet model
        yolo_model: Path to YOLO model
        save_output: Whether to save annotated image
        output_path: Where to save output (auto-generated if None)
        
    Returns:
        Prediction results dictionary
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Initialize detector
    detector = VehicleDamageDetector(resnet_model, yolo_model)
    
    # Run inference
    result = detector.predict(image, return_annotated=True)
    
    # Save annotated image
    if save_output and result.get('annotated_image'):
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_detected.jpg"
        
        result['annotated_image'].save(output_path)
        print(f"✓ Annotated image saved to: {output_path}")
        result['output_path'] = output_path
    
    return result


if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("Vehicle Damage Detector - Standalone Test")
    print("="*60)
    
    # Test with a sample image
    test_image = "test_vehicle.jpg"
    
    if os.path.exists(test_image):
        result = predict_damage(test_image)
        
        print(f"\n🔍 Detection Results:")
        print(f"   Damage Type: {result['damage_type']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Estimated Cost: ${result['estimated_cost']}")
        print(f"   Detections Found: {result['detection_count']}")
        
        for i, det in enumerate(result['detections'], 1):
            print(f"\n   Detection {i}:")
            print(f"      Class: {det['class']}")
            print(f"      Confidence: {det['confidence']:.2f}%")
            print(f"      BBox: {det['bbox']}")
    else:
        print(f"ℹ️  Test image not found. Showing model info only.")
        detector = VehicleDamageDetector()
        info = detector.get_model_info()
        print(f"\n📊 Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
