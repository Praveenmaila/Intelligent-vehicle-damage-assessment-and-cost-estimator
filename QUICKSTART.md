# 🚀 Quick Start Guide - Vehicle Damage Detection

## Integration Complete! ✓

Your Flask project has been successfully integrated with ResNet50 + YOLO vehicle damage detection.

## 📋 What's Been Added

### New Files Created:
1. **`model_inference.py`** - Core detection module combining ResNet50 and YOLO
2. **`setup_models.py`** - Automated model download and setup helper
3. **`test_integration.py`** - Test suite to verify everything works
4. **`README_INTEGRATION.md`** - Complete documentation
5. **`QUICKSTART.md`** - This file!

### Modified Files:
1. **`app.py`** - Updated with new inference pipeline
2. **`requirements.txt`** - Added ultralytics, opencv, and dependencies

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Setup Models
```bash
python setup_models.py
```
This will:
- Download YOLOv11 models automatically
- Check for ResNet50 model
- Create necessary directories
- Generate test images

### Step 3: Run the Server
```bash
python app.py
```

Open your browser: **http://localhost:5000**

## 🧪 Testing

### Quick Test (Recommended)
```bash
python test_integration.py
```

This runs a comprehensive test of:
- All imports and dependencies
- Model loading
- Inference pipeline
- Flask application

### Manual Test via API
```bash
# Windows PowerShell
$response = Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET
$response.Content | ConvertFrom-Json
```

## 📊 Features Integrated

### ✅ What Works Now:

1. **Hybrid Detection System**
   - ResNet50: Damage classification (8 classes)
   - YOLO11: Bounding box detection
   - Combined results with confidence scores

2. **Visual Annotations**
   - Bounding boxes drawn on images
   - Color-coded by damage type
   - Confidence scores displayed

3. **API Endpoints**
   - `/predict` - Single image with bounding boxes
   - `/batch-predict` - Multiple images processing
   - `/health` - System status check
   - `/model-info` - Detailed model information

4. **Cost Estimation**
   - Per-damage type pricing
   - Automatic cost calculation
   - Batch total cost computation

5. **Flexible Deployment**
   - Works with or without custom ResNet model
   - Automatic YOLO fallback
   - CPU and GPU support

## 🎯 Usage Examples

### Web Interface
1. Go to http://localhost:5000
2. Drag & drop an image of a damaged vehicle
3. Click "Analyze Damage"
4. View results with bounding boxes, damage type, and cost

### API Usage (Python)
```python
import requests

# Single image prediction
url = 'http://localhost:5000/predict'
files = {'file': open('damaged_car.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Damage: {result['damage_type']}")
print(f"Cost: ${result['estimated_cost']}")
print(f"Detections: {result['detection_count']}")
```

### cURL Example
```bash
curl -X POST -F "file=@test_vehicle.jpg" http://localhost:5000/predict
```

## 📦 Model Files

### ResNet50 Model (Optional)
- **Location**: `vehicle_damage_model.pth`
- **Purpose**: Damage classification & cost estimation
- **Source**: Train with `train.py` or download from Kaggle

### YOLO Model (Auto-downloaded)
- **Location**: `yolo11n.pt` or `yolo_vehicle_damage.pt`
- **Purpose**: Bounding box detection
- **Source**: Downloads automatically on first run

## 🔧 Configuration

### Change Detection Confidence
Edit `app.py`, line where `predict_image` is called:
```python
result = detector.predict(
    image, 
    confidence_threshold=0.25  # Lower = more detections, Higher = fewer
)
```

### Change Model Paths
Edit `app.py`, in `load_models()` function:
```python
resnet_path = 'your_custom_model.pth'
yolo_path = 'your_yolo_model.pt'
```

## 🐛 Troubleshooting

### Issue: "No module named 'ultralytics'"
**Fix**: `pip install ultralytics`

### Issue: "YOLO model not found"
**Fix**: Run `python setup_models.py` to download

### Issue: No bounding boxes appearing
**Fix**: 
- Lower confidence threshold
- Ensure YOLO model is loaded (check `/health` endpoint)
- Try different YOLO model (yolo11s.pt for better accuracy)

### Issue: "Detector not initialized"
**Fix**: 
- Restart server: `python app.py`
- Check if at least one model exists (ResNet or YOLO)
- Run `python test_integration.py` to diagnose

## 📈 Performance Tips

1. **For Faster Inference (CPU)**:
   - Use `yolo11n.pt` (nano model)
   - Reduce image size before upload
   - Lower confidence threshold reduces processing

2. **For Better Accuracy**:
   - Use `yolo11s.pt` or `yolo11m.pt`
   - Higher confidence threshold (0.4-0.5)
   - Train custom YOLO on vehicle damage dataset

3. **For GPU Acceleration**:
   - Install CUDA-enabled PyTorch
   - System automatically uses GPU if available

## 🎓 Next Steps

### 1. Train Custom Models
- Train ResNet50: `python train.py`
- Fine-tune YOLO on vehicle damage dataset

### 2. Customize Damage Classes
Edit `model_inference.py` to add more damage types:
```python
self.damage_colors = {
    'your_new_class': (R, G, B),
    # ... add more
}
```

### 3. Integrate with Database
Store detection history, user uploads, cost reports

### 4. Add More Features
- Real-time video processing
- PDF report generation
- Email notifications
- Mobile app integration

## 📚 Documentation

- **Full Documentation**: `README_INTEGRATION.md`
- **API Reference**: See `/health` and `/model-info` endpoints
- **Training Guide**: See `train.py` comments

## 🔗 Kaggle References

Based on these notebooks:
1. [Vehicle Damage Detection with ResNet50 & YOLO11](https://www.kaggle.com/code/sumanthvuppu/vehicle-damage-detection-with-resnet50-yolo11)
2. [Vehicle Damage Detection](https://www.kaggle.com/code/rsainivas/vehicle-damage-detection/notebook)

## ✅ Checklist

- [x] Install dependencies (`requirements.txt`)
- [x] Run setup script (`setup_models.py`)
- [x] Test integration (`test_integration.py`)
- [ ] Start server (`app.py`)
- [ ] Test web interface (http://localhost:5000)
- [ ] Try sample predictions

## 💡 Tips

- Keep models in the same directory as `app.py`
- Use high-quality images for best results
- YOLO works even without ResNet model
- Check `/health` endpoint to verify model status

## 🎉 You're Ready!

Your vehicle damage detection system is now integrated and ready to use!

**Start the server**: `python app.py`
**Test it**: Upload an image at http://localhost:5000

---

Questions or issues? Check `README_INTEGRATION.md` for detailed documentation.
