#!/usr/bin/env python3
# -------------------------------
# One-Click Setup and Training
# Vehicle Damage Detection System
# -------------------------------

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("📦 Checking Dependencies")
    
    required = [
        'torch', 'torchvision', 'PIL', 'flask', 
        'cv2', 'albumentations', 'ultralytics', 'pandas'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} (missing)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ").strip().lower()
        if response == 'y':
            print("\n📥 Installing requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Installation complete!")
        else:
            print("❌ Cannot proceed without dependencies.")
            return False
    else:
        print("\n✅ All dependencies installed!")
    
    return True

def check_dataset():
    """Check if dataset exists"""
    print_header("📂 Checking Dataset")
    
    csv_path = Path("datasets/data.csv")
    
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset CSV found: {len(df)} images")
        
        # Check if images exist
        existing = 0
        for img_path in df['image'][:10]:  # Check first 10
            full_path = Path("datasets") / img_path if not Path(img_path).exists() else Path(img_path)
            if full_path.exists():
                existing += 1
        
        if existing > 0:
            print(f"✓ Images found in dataset folder")
            return True
        else:
            print(f"⚠️  CSV exists but images not found")
            return False
    else:
        print("❌ No dataset found")
        return False

def download_dataset_interactive():
    """Interactive dataset download"""
    print_header("📥 Dataset Setup")
    
    print("Choose dataset source:")
    print("1. Create sample dataset (for testing)")
    print("2. Manual setup (I have my own images)")
    print("3. Skip (download dataset manually)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n🔨 Creating sample dataset...")
        subprocess.run([sys.executable, "download_dataset.py"])
        return True
        
    elif choice == '2':
        print("\n📖 Manual Setup Instructions:")
        print("="*70)
        print("\n1. Create folder: datasets/images/")
        print("2. Add your vehicle damage images there")
        print("3. Create datasets/data.csv with this format:")
        print("\n   image,classes")
        print("   images/img1.jpg,front_bumper_dent")
        print("   images/img2.jpg,door_scratch")
        print("   images/img3.jpg,hood_dent")
        print("\n4. Supported damage types:")
        print("   front_bumper_dent, rear_bumper_scratch, door_dent,")
        print("   hood_scratch, head_lamp, windshield_crack, etc.")
        print("\n" + "="*70)
        
        input("\nPress Enter when you've completed the setup...")
        return check_dataset()
        
    else:
        print("\n📖 Download dataset manually from:")
        print("1. Kaggle: https://www.kaggle.com/datasets")
        print("   Search: 'car damage detection'")
        print("2. Roboflow: https://universe.roboflow.com/")
        print("   Search: 'vehicle damage detection'")
        print("\nPlace images in: datasets/images/")
        print("Create CSV: datasets/data.csv")
        return False

def check_trained_model():
    """Check if model is already trained"""
    print_header("🤖 Checking Model")
    
    model_path = Path("vehicle_damage_model.pth")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Trained model found: {size_mb:.2f} MB")
        
        response = input("\nRetrain model? (y/n): ").strip().lower()
        return response == 'y'
    else:
        print("❌ No trained model found")
        return True

def train_model():
    """Train the model"""
    print_header("🚀 Training Model")
    
    print("Choose training script:")
    print("1. Enhanced training (recommended) - with data augmentation")
    print("2. Basic training - faster but less accurate")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        script = "train_improved.py"
        print("\n📊 Starting enhanced training...")
        print("This includes data augmentation for better accuracy")
    else:
        script = "train.py"
        print("\n📊 Starting basic training...")
    
    print("\nTraining in progress... This may take 10-30 minutes.")
    print("You'll see progress updates for each epoch.\n")
    
    try:
        subprocess.run([sys.executable, script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Training failed!")
        return False
    except FileNotFoundError:
        print(f"\n❌ Training script not found: {script}")
        return False

def test_model():
    """Test the trained model"""
    print_header("🧪 Testing Model")
    
    print("Testing model inference...")
    
    try:
        from model_inference import VehicleDamageDetector
        
        detector = VehicleDamageDetector()
        info = detector.get_model_info()
        
        print(f"\n📊 Model Information:")
        print(f"   Device: {info['device']}")
        print(f"   ResNet Loaded: {'✓' if info['resnet_loaded'] else '✗'}")
        print(f"   YOLO Loaded: {'✓' if info['yolo_loaded'] else '✗'}")
        print(f"   Number of Classes: {info['num_classes']}")
        
        if info['resnet_loaded']:
            print("\n✅ Model is ready for inference!")
            return True
        else:
            print("\n⚠️  Model loaded but may not work properly")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        return False

def start_app():
    """Start the Flask application"""
    print_header("🌐 Starting Web Application")
    
    print("Starting Flask server...")
    print("Open your browser at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

def main():
    """Main setup wizard"""
    print_header("🚗 Vehicle Damage Detection System - Setup Wizard")
    
    print("This wizard will guide you through:")
    print("1. Dependency installation")
    print("2. Dataset setup")
    print("3. Model training")
    print("4. Application launch")
    
    input("\nPress Enter to begin...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed: Missing dependencies")
        return
    
    # Step 2: Check dataset
    dataset_ready = check_dataset()
    
    if not dataset_ready:
        response = input("\nSetup dataset now? (y/n): ").strip().lower()
        if response == 'y':
            dataset_ready = download_dataset_interactive()
        
        if not dataset_ready:
            print("\n⚠️  Cannot train without dataset")
            print("Please setup dataset and run this script again")
            return
    
    # Step 3: Check/train model
    need_training = check_trained_model()
    
    if need_training:
        if not check_dataset():
            print("\n❌ Cannot train: Dataset not ready")
            return
        
        response = input("\nStart training now? (y/n): ").strip().lower()
        if response == 'y':
            if not train_model():
                print("\n❌ Setup failed: Training error")
                return
            
            # Test the model
            test_model()
        else:
            print("\n⚠️  Model not trained. System will show 'unknown' for all predictions.")
    
    # Step 4: Launch app
    print_header("✅ Setup Complete!")
    
    print("Your vehicle damage detection system is ready!")
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Upload vehicle damage images")
    print("4. View detection results and cost estimates")
    
    response = input("\nStart application now? (y/n): ").strip().lower()
    if response == 'y':
        start_app()
    else:
        print("\n👍 Run 'python app.py' when ready!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
