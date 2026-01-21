#!/usr/bin/env python3
"""
Model Setup and Download Helper
Downloads pretrained YOLO models and provides instructions for ResNet50 weights
"""

import os
import sys
from pathlib import Path

def download_yolo_model():
    """Download pretrained YOLOv11 model for vehicle detection."""
    try:
        from ultralytics import YOLO
        
        print("="*60)
        print("Downloading YOLOv11 Pretrained Model")
        print("="*60)
        
        # Download YOLOv11n (nano - fastest, good for CPU)
        print("\n📥 Downloading YOLOv11n (Nano) model...")
        print("   This is a lightweight model suitable for CPU inference.")
        
        model = YOLO('yolo11n.pt')
        print("✓ YOLOv11n downloaded successfully!")
        
        # Also download YOLOv11s (small) as an alternative
        print("\n📥 Downloading YOLOv11s (Small) model...")
        print("   Better accuracy, slightly slower.")
        
        model_s = YOLO('yolo11s.pt')
        print("✓ YOLOv11s downloaded successfully!")
        
        print("\n✓ YOLO models ready!")
        print("\nℹ️  Available models:")
        print("   - yolo11n.pt (Fastest, CPU-friendly)")
        print("   - yolo11s.pt (Better accuracy)")
        
        return True
        
    except ImportError:
        print("❌ ultralytics not installed!")
        print("   Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error downloading YOLO models: {e}")
        return False

def check_resnet_model():
    """Check if ResNet50 model exists and provide instructions."""
    print("\n" + "="*60)
    print("Checking ResNet50 Model")
    print("="*60)
    
    model_path = "vehicle_damage_model.pth"
    
    if os.path.exists(model_path):
        print(f"✓ Found ResNet model: {model_path}")
        
        # Try to load and display info
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print("\n📊 Model Information:")
            print(f"   Classes: {len(checkpoint.get('class_mapping', {}))}")
            print(f"   Validation Accuracy: {checkpoint.get('validation_accuracy', 0):.2%}")
            
            if 'class_mapping' in checkpoint:
                print("\n   Class Mapping:")
                for name, idx in sorted(checkpoint['class_mapping'].items(), key=lambda x: x[1]):
                    cost = checkpoint.get('cost_mapping', {}).get(idx, 0)
                    print(f"      {idx}: {name} (${cost})")
            
            return True
        except Exception as e:
            print(f"⚠️  Model file exists but couldn't load: {e}")
            return False
    else:
        print(f"❌ ResNet model not found: {model_path}")
        print("\n📋 To use custom ResNet50 model:")
        print("   1. Train your model using train.py")
        print("   2. Or download pretrained weights from Kaggle:")
        print("      - https://www.kaggle.com/code/sumanthvuppu/vehicle-damage-detection-with-resnet50-yolo11")
        print("      - https://www.kaggle.com/code/rsainivas/vehicle-damage-detection/notebook")
        print("   3. Place the model file as 'vehicle_damage_model.pth' in this directory")
        print("\n   The application will work with YOLO-only detection if ResNet is missing.")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'ultralytics': 'Ultralytics YOLO',
        'cv2': 'OpenCV',
        'flask': 'Flask',
        'numpy': 'NumPy'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} package(s)")
        print("\n📦 Install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def create_sample_test_image():
    """Create a simple test image for verification."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a simple car-like shape
        img = Image.new('RGB', (640, 480), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple car shape
        # Body
        draw.rectangle([150, 200, 490, 320], fill=(30, 30, 200), outline=(0, 0, 0), width=3)
        # Roof
        draw.rectangle([200, 150, 440, 200], fill=(30, 30, 200), outline=(0, 0, 0), width=3)
        # Windows
        draw.rectangle([220, 160, 310, 195], fill=(100, 150, 200), outline=(0, 0, 0), width=2)
        draw.rectangle([330, 160, 420, 195], fill=(100, 150, 200), outline=(0, 0, 0), width=2)
        # Wheels
        draw.ellipse([170, 300, 230, 360], fill=(20, 20, 20), outline=(0, 0, 0), width=2)
        draw.ellipse([410, 300, 470, 360], fill=(20, 20, 20), outline=(0, 0, 0), width=2)
        # Headlights
        draw.ellipse([140, 230, 160, 250], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
        
        # Add damage indicator (scratch)
        draw.line([300, 250, 400, 280], fill=(255, 0, 0), width=5)
        
        # Save
        img.save('test_vehicle.jpg')
        print("✓ Created test image: test_vehicle.jpg")
        return True
    except Exception as e:
        print(f"⚠️  Couldn't create test image: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n" + "="*60)
    print("Setting Up Directories")
    print("="*60)
    
    dirs = ['uploads', 'datasets', 'templates', 'static']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")
    
    return True

def main():
    """Main setup routine."""
    print("\n")
    print("="*70)
    print(" "*15 + "VEHICLE DAMAGE DETECTION SETUP")
    print("="*70)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check ResNet model
    resnet_ok = check_resnet_model()
    
    # Download YOLO models
    if deps_ok:
        yolo_ok = download_yolo_model()
    else:
        yolo_ok = False
        print("\n⚠️  Skipping YOLO download - install dependencies first")
    
    # Create test image
    create_sample_test_image()
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    print(f"Dependencies:  {'✓ OK' if deps_ok else '❌ MISSING'}")
    print(f"ResNet Model:  {'✓ OK' if resnet_ok else '❌ NOT FOUND (optional)'}")
    print(f"YOLO Models:   {'✓ OK' if yolo_ok else '❌ FAILED (install ultralytics)'}")
    
    if deps_ok and (resnet_ok or yolo_ok):
        print("\n✓ Setup complete! You can now run:")
        print("   python app.py")
        print("\nℹ️  Note: At least one model (ResNet or YOLO) must be available.")
    elif deps_ok:
        print("\n⚠️  Setup incomplete:")
        print("   - Install dependencies: pip install -r requirements.txt")
        if not resnet_ok:
            print("   - Get ResNet model: Train or download from Kaggle")
        if not yolo_ok:
            print("   - YOLO will download automatically on first run")
    else:
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        print("\n   Then run this script again: python setup_models.py")
    
    print("="*70)
    print("\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
