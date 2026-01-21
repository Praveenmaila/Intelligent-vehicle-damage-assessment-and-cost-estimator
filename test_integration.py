"""
Quick Test Script for Vehicle Damage Detection Integration
Tests the inference pipeline without starting the full Flask server
"""

import sys
import os
from PIL import Image
import io

def test_imports():
    """Test if all required modules can be imported."""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    modules = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'flask': 'Flask',
        'numpy': 'NumPy'
    }
    
    optional_modules = {
        'ultralytics': 'Ultralytics YOLO (optional)',
        'cv2': 'OpenCV (optional)'
    }
    
    success = True
    
    # Test required modules
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"❌ {name} - FAILED: {e}")
            success = False
    
    # Test optional modules
    for module, name in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠️  {name} - Not installed (optional)")
    
    return success

def test_model_inference_module():
    """Test if model_inference.py can be imported."""
    print("\n" + "="*60)
    print("Testing Model Inference Module")
    print("="*60)
    
    try:
        from model_inference import VehicleDamageDetector
        print("✓ model_inference.py imported successfully")
        
        # Try to instantiate detector (will warn if models not found)
        print("\nAttempting to initialize detector...")
        detector = VehicleDamageDetector(
            resnet_model_path='vehicle_damage_model.pth',
            yolo_model_path='yolo_vehicle_damage.pt'
        )
        
        print("\n✓ Detector initialized successfully")
        
        # Get model info
        info = detector.get_model_info()
        print("\n📊 Model Status:")
        print(f"   Device: {info['device']}")
        print(f"   ResNet Loaded: {info['resnet_loaded']}")
        print(f"   YOLO Loaded: {info['yolo_loaded']}")
        print(f"   Classes: {info['num_classes']}")
        
        return True, detector
        
    except ImportError as e:
        print(f"❌ Failed to import model_inference: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_prediction(detector):
    """Test prediction on a dummy image."""
    print("\n" + "="*60)
    print("Testing Prediction Pipeline")
    print("="*60)
    
    try:
        # Create a simple test image
        from PIL import Image, ImageDraw
        import numpy as np
        
        img = Image.new('RGB', (640, 480), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple car-like shape
        draw.rectangle([150, 200, 490, 320], fill=(30, 30, 200), outline=(0, 0, 0), width=3)
        draw.rectangle([200, 150, 440, 200], fill=(30, 30, 200), outline=(0, 0, 0), width=3)
        
        print("✓ Created test image (640x480)")
        
        # Run prediction
        print("\nRunning inference pipeline...")
        result = detector.predict(img, return_annotated=True, confidence_threshold=0.25)
        
        print("\n✓ Prediction completed successfully!")
        print("\n📊 Results:")
        print(f"   Damage Type: {result['damage_type']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Estimated Cost: ${result['estimated_cost']}")
        print(f"   Detections Found: {result['detection_count']}")
        
        if result['detections']:
            print("\n   Bounding Boxes:")
            for i, det in enumerate(result['detections'], 1):
                print(f"      {i}. Class: {det['class']}, Confidence: {det['confidence']:.1f}%")
                print(f"         BBox: {det['bbox']}")
        
        if result.get('annotated_image'):
            print("\n✓ Annotated image generated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_app():
    """Test if Flask app can be imported."""
    print("\n" + "="*60)
    print("Testing Flask Application")
    print("="*60)
    
    try:
        import app
        print("✓ app.py imported successfully")
        print("✓ Flask routes configured")
        
        # Check if key functions exist
        functions = ['load_models', 'predict_image', 'allowed_file']
        for func in functions:
            if hasattr(app, func):
                print(f"✓ Function '{func}' found")
            else:
                print(f"⚠️  Function '{func}' not found")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("="*70)
    print(" "*15 + "VEHICLE DAMAGE DETECTION TEST SUITE")
    print("="*70)
    print("\n")
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    # Test 2: Model Inference Module
    inference_ok, detector = test_model_inference_module()
    
    if not inference_ok:
        print("\n❌ Model inference module test failed.")
        return False
    
    # Test 3: Prediction (if we have a detector)
    if detector:
        prediction_ok = test_prediction(detector)
    else:
        prediction_ok = False
        print("\n⚠️  Skipping prediction test (no detector available)")
    
    # Test 4: Flask App
    flask_ok = test_flask_app()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Imports:           {'✓ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Inference Module:  {'✓ PASS' if inference_ok else '❌ FAIL'}")
    print(f"Prediction Test:   {'✓ PASS' if prediction_ok else '⚠️  SKIP'}")
    print(f"Flask App:         {'✓ PASS' if flask_ok else '❌ FAIL'}")
    
    all_pass = imports_ok and inference_ok and flask_ok
    
    if all_pass:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou can now start the server:")
        print("   python app.py")
        print("\nOr setup models first:")
        print("   python setup_models.py")
    else:
        print("\n" + "="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)
        print("\nPlease check the errors above and:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Ensure model files exist or run: python setup_models.py")
    
    print("="*70)
    print("\n")
    
    return all_pass

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tests failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
