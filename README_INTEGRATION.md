# Vehicle Damage Detection - ResNet50 + YOLO Integration

## 🚀 Overview

This Flask application integrates **ResNet50** for damage classification and **YOLO11** for bounding box detection to provide comprehensive vehicle damage assessment with cost estimation.

## 🏗️ Architecture

### Hybrid Detection System

1. **ResNet50 Classifier**
   - Classifies damage types (bumper_dent, door_scratch, glass_shatter, etc.)
   - Provides confidence scores
   - Estimates repair costs

2. **YOLO11 Detector**
   - Detects damaged regions with bounding boxes
   - Locates multiple damage areas in a single image
   - Provides precise spatial information

3. **Flask Web Interface**
   - User-friendly upload interface
   - Real-time predictions
   - Visual annotations with bounding boxes
   - Batch processing support

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Models

Run the setup script to download YOLO models and check configuration:

```bash
python setup_models.py
```

This will:
- Check all dependencies
- Download YOLOv11 pretrained models
- Verify ResNet50 model (if available)
- Create necessary directories
- Generate test images

### 3. Model Files Required

#### ResNet50 Model (Optional but Recommended)
- **File**: `vehicle_damage_model.pth`
- **Source**: Train using `train.py` or download from Kaggle
- **Contains**: 
  - Model weights
  - Class mappings
  - Cost mappings
  - Validation metrics

#### YOLO Model (Auto-downloaded)
- **File**: `yolo_vehicle_damage.pt` (custom trained) OR `yolo11n.pt` (pretrained)
- **Source**: Auto-downloaded by setup script
- **Alternatives**: YOLOv11n, YOLOv11s, YOLOv11m

## 🎯 Usage

### Start the Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

### API Endpoints

#### 1. Single Image Detection
```bash
POST /predict
Content-Type: multipart/form-data

Request:
- file: image file

Response:
{
    "damage_type": "bumper_dent",
    "confidence": 95.42,
    "estimated_cost": 500,
    "detection_count": 2,
    "detections": [
        {
            "bbox": [100, 150, 300, 350],
            "confidence": 89.5,
            "class": "vehicle",
            "class_id": 0
        }
    ],
    "original_image": "data:image/jpeg;base64,...",
    "annotated_image": "data:image/jpeg;base64,..."
}
```

#### 2. Batch Processing
```bash
POST /batch-predict
Content-Type: multipart/form-data

Request:
- files: multiple image files

Response:
{
    "results": [
        {
            "filename": "car1.jpg",
            "damage_type": "door_scratch",
            "cost": 300,
            "confidence": 92.1,
            "detections": 1,
            "bounding_boxes": [...]
        }
    ],
    "total_cost": 1500,
    "total_images": 5,
    "total_detections": 8
}
```

#### 3. Health Check
```bash
GET /health

Response:
{
    "status": "healthy",
    "detector_loaded": true,
    "resnet_loaded": true,
    "yolo_loaded": true,
    "device": "cpu",
    "cuda_available": false,
    "num_classes": 8
}
```

#### 4. Model Information
```bash
GET /model-info

Response:
{
    "device": "cpu",
    "resnet_loaded": true,
    "yolo_loaded": true,
    "num_classes": 8,
    "cuda_available": false,
    "class_mapping": {
        "unknown": 0,
        "head_lamp": 1,
        "door_scratch": 2,
        ...
    },
    "cost_mapping": {
        "0": 0,
        "1": 800,
        "2": 300,
        ...
    }
}
```

## 🔧 Configuration

### Model Paths
Edit `app.py` to customize model paths:
```python
resnet_path = 'vehicle_damage_model.pth'
yolo_path = 'yolo_vehicle_damage.pt'
```

### Detection Parameters
Adjust confidence threshold in predictions:
```python
result = detector.predict(
    image, 
    return_annotated=True,
    confidence_threshold=0.25  # Adjust this value
)
```

### Damage Type Colors
Customize bounding box colors in `model_inference.py`:
```python
self.damage_colors = {
    'bumper_dent': (255, 0, 0),      # Red
    'door_scratch': (0, 255, 0),     # Green
    # ... add more
}
```

## 📊 Damage Classes

| Class ID | Damage Type | Estimated Cost |
|----------|-------------|----------------|
| 0 | unknown | $0 |
| 1 | head_lamp | $800 |
| 2 | door_scratch | $300 |
| 3 | glass_shatter | $1200 |
| 4 | tail_lamp | $600 |
| 5 | bumper_dent | $500 |
| 6 | door_dent | $700 |
| 7 | bumper_scratch | $250 |

## 🧪 Testing

### Test with Sample Image
```bash
# Create test image
python setup_models.py

# Test inference directly
python model_inference.py

# Test via API
curl -X POST -F "file=@test_vehicle.jpg" http://localhost:5000/predict
```

### Test Batch Processing
```bash
curl -X POST \
  -F "files=@car1.jpg" \
  -F "files=@car2.jpg" \
  -F "files=@car3.jpg" \
  http://localhost:5000/batch-predict
```

## 🎨 Web Interface

Access the web interface at `http://localhost:5000`

Features:
- Drag-and-drop image upload
- Real-time damage detection
- Visual bounding boxes
- Cost estimation
- Confidence scores
- Download annotated images

## 🔍 Troubleshooting

### Issue: YOLO model not found
**Solution**: Run `python setup_models.py` to download models

### Issue: CUDA out of memory
**Solution**: The system automatically falls back to CPU. Reduce image size or batch size.

### Issue: Import error for ultralytics
**Solution**: `pip install ultralytics`

### Issue: No detections appearing
**Solution**: 
- Lower confidence threshold (default 0.25)
- Check if YOLO model is loaded (`/health` endpoint)
- Ensure image quality is good

### Issue: ResNet model not loading
**Solution**:
- Train model using `train.py`
- Download from Kaggle notebooks
- System will work with YOLO-only detection as fallback

## 📈 Performance

### Speed Benchmarks (CPU)
- ResNet50 Classification: ~100-200ms per image
- YOLO11n Detection: ~200-400ms per image
- Total Pipeline: ~300-600ms per image

### Accuracy
- ResNet50: ~85-95% (depends on training)
- YOLO11: ~70-90% (general object detection)

## 🔗 Integration with Kaggle Notebooks

This implementation is based on:

1. **[Vehicle Damage Detection with ResNet50 & YOLO11](https://www.kaggle.com/code/sumanthvuppu/vehicle-damage-detection-with-resnet50-yolo11)**
   - ResNet50 training pipeline
   - Custom damage classification
   - Cost estimation mapping

2. **[Vehicle Damage Detection](https://www.kaggle.com/code/rsainivas/vehicle-damage-detection/notebook)**
   - YOLO training approach
   - Bounding box detection
   - Dataset preparation

## 📝 Code Structure

```
vehicle_damage_detection/
├── app.py                  # Flask application (main server)
├── model_inference.py      # Detection module (ResNet50 + YOLO)
├── train.py               # Training script for ResNet50
├── setup_models.py        # Model download and setup
├── requirements.txt       # Python dependencies
├── vehicle_damage_model.pth   # ResNet50 weights
├── yolo_vehicle_damage.pt     # YOLO weights (or yolo11n.pt)
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload storage
└── datasets/             # Training data
    ├── data.csv
    └── image/
```

## 🚦 Next Steps

1. **Improve YOLO Training**: Train YOLO on vehicle damage dataset
2. **Ensemble Methods**: Combine predictions from both models
3. **Add More Classes**: Expand damage type classifications
4. **Mobile App**: Create mobile interface
5. **Real-time Video**: Add video stream processing
6. **Database Integration**: Store detection history
7. **Report Generation**: PDF reports with damage assessment

## 📄 License

Apache 2.0 (matching Kaggle notebooks)

## 👥 Credits

- Kaggle notebooks by Sumanth Vuppu and R Sai Nivas
- Ultralytics YOLOv11
- PyTorch and TorchVision teams
