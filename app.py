# -------------------------------
# Flask Web Application for Vehicle Damage Detection
# Real-time Damage Assessment with ResNet50 + YOLO
# Integrated Detection and Cost Estimation
# -------------------------------

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import base64
from io import BytesIO
import json

# Import our custom inference module
from model_inference import VehicleDamageDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Global Variables for Model
# -------------------------------
detector = None

# -------------------------------
# Load Models on Startup
# -------------------------------
def load_models():
    global detector
    
    resnet_path = 'vehicle_damage_model.pth'
    yolo_path = 'yolo_vehicle_damage.pt'
    
    print("="*60)
    print("Initializing Vehicle Damage Detection System")
    print("="*60)
    
    # Initialize the hybrid detector (ResNet50 + YOLO)
    detector = VehicleDamageDetector(
        resnet_model_path=resnet_path,
        yolo_model_path=yolo_path
    )
    
    # Get model info
    info = detector.get_model_info()
    
    print("\n📊 Model Status:")
    print(f"   Device: {info['device']}")
    print(f"   ResNet50 Loaded: {'✓' if info['resnet_loaded'] else '✗'}")
    print(f"   YOLO Loaded: {'✓' if info['yolo_loaded'] else '✗'}")
    print(f"   Number of Classes: {info['num_classes']}")
    print(f"   CUDA Available: {info['cuda_available']}")
    
    if not info['resnet_loaded'] and not info['yolo_loaded']:
        print("\n⚠️  WARNING: No models loaded! Detection will not work.")
        print("   Please ensure model files exist:")
        print(f"   - {resnet_path}")
        print(f"   - {yolo_path}")
    
    print("="*60)

# -------------------------------
# Helper Functions
# -------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image, return_annotated=True):
    """
    Predict damage type, cost, and detect bounding boxes from PIL Image.
    Uses hybrid ResNet50 + YOLO detection.
    
    Args:
        image: PIL Image
        return_annotated: Whether to return image with bounding boxes drawn
        
    Returns:
        Dictionary with prediction results and optional annotated image
    """
    global detector
    
    if detector is None:
        raise RuntimeError("Detector not initialized. Please restart the server.")
    
    # Run complete inference pipeline
    result = detector.predict(
        image, 
        return_annotated=return_annotated,
        confidence_threshold=0.25
    )
    
    return result

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction with bounding boxes"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Open and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction with bounding boxes
        result = predict_image(image, return_annotated=True)
        
        # Convert original image to base64
        buffered_orig = BytesIO()
        image.save(buffered_orig, format="JPEG")
        img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode()
        result['original_image'] = f"data:image/jpeg;base64,{img_str_orig}"
        
        # Convert annotated image to base64 (if available)
        if 'annotated_image' in result and result['annotated_image']:
            buffered_annotated = BytesIO()
            result['annotated_image'].save(buffered_annotated, format="JPEG")
            img_str_annotated = base64.b64encode(buffered_annotated.getvalue()).decode()
            result['annotated_image'] = f"data:image/jpeg;base64,{img_str_annotated}"
        else:
            # Fallback to original if no annotation
            result['annotated_image'] = result['original_image']
        
        # For backward compatibility, keep 'image' field
        result['image'] = result['annotated_image']
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle multiple image uploads with detection"""
    try:
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        total_cost = 0
        total_detections = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                image = Image.open(file.stream).convert('RGB')
                prediction = predict_image(image, return_annotated=False)
                
                results.append({
                    'filename': secure_filename(file.filename),
                    'damage_type': prediction['damage_type'],
                    'cost': prediction['estimated_cost'],
                    'confidence': prediction['confidence'],
                    'detections': prediction.get('detection_count', 0),
                    'bounding_boxes': prediction.get('detections', [])
                })
                
                total_cost += prediction['estimated_cost']
                total_detections += prediction.get('detection_count', 0)
        
        return jsonify({
            'results': results,
            'total_cost': total_cost,
            'total_images': len(results),
            'total_detections': total_detections
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if detector:
        info = detector.get_model_info()
        return jsonify({
            'status': 'healthy',
            'detector_loaded': detector is not None,
            'resnet_loaded': info['resnet_loaded'],
            'yolo_loaded': info['yolo_loaded'],
            'device': info['device'],
            'cuda_available': info['cuda_available'],
            'num_classes': info['num_classes']
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'detector_loaded': False,
            'error': 'Detector not initialized'
        }), 503

@app.route('/model-info')
def model_info():
    """Get detailed model information"""
    if detector:
        info = detector.get_model_info()
        return jsonify(info)
    else:
        return jsonify({'error': 'Detector not initialized'}), 503

# -------------------------------
# Run Application
# -------------------------------
if __name__ == '__main__':
    print("="*60)
    print("Vehicle Damage Detection - Flask Web Application")
    print("ResNet50 + YOLO Hybrid Detection System")
    print("="*60)
    
    # Load models before starting server
    load_models()
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("="*60)
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("\n📋 API Endpoints:")
    print("   POST /predict        - Single image detection with bounding boxes")
    print("   POST /batch-predict  - Multiple images batch processing")
    print("   GET  /health         - Health check and model status")
    print("   GET  /model-info     - Detailed model information")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
