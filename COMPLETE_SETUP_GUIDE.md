# ========================================
# COMPLETE SETUP GUIDE
# Vehicle Damage Detection System
# ========================================

## 🚀 Quick Start (5 Steps to Working System)

### Step 1: Install Dependencies
```powershell
# Make sure you're in your virtual environment
pip install -r requirements.txt
```

### Step 2: Download Dataset
```powershell
python download_dataset.py
```

**Recommended Real-World Datasets:**

1. **Kaggle Datasets** (Best Quality)
   - `anujms/car-damage-detection` (920 images)
   - `lplenka/car-damage-detection` (1000+ images)
   - `farzadnekouei/car-damage-detection`
   
   **How to download:**
   ```powershell
   # Install kaggle CLI
   pip install kaggle
   
   # Setup Kaggle API (one-time)
   # 1. Go to https://www.kaggle.com/settings/account
   # 2. Click "Create New API Token"
   # 3. Save kaggle.json to: C:\Users\<username>\.kaggle\
   
   # Download dataset
   kaggle datasets download -d anujms/car-damage-detection
   unzip car-damage-detection.zip -d datasets/
   ```

2. **Roboflow Universe** (Pre-annotated)
   - Search: "vehicle damage detection"
   - URL: https://universe.roboflow.com/
   - Download in YOLOv8 format
   
   **How to download:**
   ```powershell
   pip install roboflow
   
   # In Python:
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("workspace-name").project("project-name")
   dataset = project.version(1).download("yolov8", location="datasets/")
   ```

3. **Manual Collection** (Internet Images)
   - Google Images: Search "car damage", "vehicle dent", etc.
   - Organize in `datasets/images/`
   - Create `datasets/data.csv`:
     ```csv
     image,classes
     images/damage1.jpg,front_bumper_dent
     images/damage2.jpg,door_scratch
     ```

### Step 3: Train the Model
```powershell
# Option A: Enhanced training (recommended)
python train_improved.py

# Option B: Basic training
python train.py
```

**Training outputs:**
- `vehicle_damage_model.pth` - Classification model
- Training takes 10-30 minutes depending on GPU

### Step 4: Train YOLO for Detection (Optional but Recommended)
```powershell
# This requires annotations in YOLO format
python train_yolo.py
```

### Step 5: Run the Application
```powershell
python app.py
```

Open browser: http://localhost:5000

---

## 📋 Dataset Structure

Your `datasets/` folder should look like this:

```
datasets/
├── data.csv              # Image paths and labels
├── images/               # All images
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img100.jpg
│   │   └── ...
│   └── test/
│       ├── img200.jpg
│       └── ...
└── annotations/          # YOLO format (optional)
    ├── img001.txt
    └── ...
```

**data.csv format:**
```csv
image,classes
images/train/img001.jpg,front_bumper_dent
images/train/img002.jpg,door_scratch
images/train/img003.jpg,hood_dent
```

---

## 🏷️ Supported Damage Classes

The system recognizes these damage types:

**Basic Classes (8):**
1. `unknown` - No damage or unrecognized
2. `minor_dent` - Small dents
3. `major_dent` - Large dents
4. `minor_scratch` - Surface scratches
5. `major_scratch` - Deep scratches
6. `glass_shatter` - Broken glass
7. `lamp_broken` - Damaged lights
8. `bumper_dent` - Bumper damage

**Extended Classes (31):**
- Lights: `head_lamp`, `rear_lamp`, `tail_lamp`
- Bumpers: `front_bumper_dent`, `rear_bumper_scratch`
- Body: `door_dent`, `hood_scratch`, `trunk_dent`, `fender_scratch`
- Glass: `windshield_crack`, `side_window_shatter`
- Others: `wheel_rim_bent`, `paint_peel`, `rust_damage`

---

## 🔧 Troubleshooting

### Problem: "No module named 'flask'"
**Solution:**
```powershell
pip install -r requirements.txt
```

### Problem: "Dataset not found"
**Solution:**
1. Run `python download_dataset.py`
2. Or manually create `datasets/data.csv`

### Problem: Model always predicts "unknown"
**Cause:** Model not trained yet
**Solution:**
1. Download dataset (Step 2)
2. Train model (Step 3)
3. Restart app

### Problem: Low accuracy on real images
**Solution:**
1. Use more diverse training data
2. Include internet images in training
3. Train for more epochs (20-30)
4. Use data augmentation (train_improved.py does this)

### Problem: CUDA out of memory
**Solution:**
```python
# In train_improved.py, reduce batch size:
BATCH_SIZE = 16  # or 8
```

---

## 📊 Cost Estimates

Repair costs in INR (based on 2024-2026 rates):

| Damage Type | Min Cost | Avg Cost | Max Cost |
|-------------|----------|----------|----------|
| Minor Scratch | ₹4,000 | ₹6,000 | ₹10,000 |
| Major Scratch | ₹12,000 | ₹18,000 | ₹25,000 |
| Minor Dent | ₹6,000 | ₹8,000 | ₹12,000 |
| Major Dent | ₹15,000 | ₹25,000 | ₹40,000 |
| Glass Shatter | ₹20,000 | ₹30,000 | ₹50,000 |
| Lamp Broken | ₹8,000 | ₹12,000 | ₹25,000 |
| Bumper Dent | ₹12,000 | ₹20,000 | ₹35,000 |

---

## 🎯 Model Performance Tips

**To improve accuracy:**

1. **More Training Data** (Most Important)
   - Aim for 200+ images per class
   - Include various angles and lighting
   - Mix of real damage photos

2. **Data Augmentation**
   - Use `train_improved.py` (has built-in augmentation)
   - Rotations, flips, brightness changes

3. **Better Model Architecture**
   ```python
   # In train_improved.py, try:
   MODEL_TYPE = 'resnet50'  # Better than resnet18
   # or
   MODEL_TYPE = 'efficientnet_b0'  # Best accuracy/speed
   ```

4. **Train Longer**
   ```python
   NUM_EPOCHS = 30  # Instead of 20
   ```

5. **Transfer Learning**
   - Already enabled in train_improved.py
   - Uses ImageNet pretrained weights

---

## 🌐 Using Internet Images

To test with internet images:

1. **Download image:**
   ```powershell
   # Save any car damage image from internet
   # Name it: test_damage.jpg
   ```

2. **Upload via web interface:**
   - Run `python app.py`
   - Open http://localhost:5000
   - Click "Choose File"
   - Select image

3. **Or test directly:**
   ```powershell
   python
   >>> from model_inference import predict_damage
   >>> result = predict_damage('test_damage.jpg')
   >>> print(result)
   ```

---

## 📈 Training Progress

Expected training output:
```
Epoch 1/20
Train Loss: 1.8234 | Train Acc: 0.3245
Val Loss: 1.5423 | Val Acc: 0.4123
✓ New best model! Accuracy: 0.4123

Epoch 10/20
Train Loss: 0.6234 | Train Acc: 0.7856
Val Loss: 0.7123 | Val Acc: 0.7456
✓ New best model! Accuracy: 0.7456

Epoch 20/20
Train Loss: 0.3234 | Train Acc: 0.8956
Val Loss: 0.5123 | Val Acc: 0.8234
✓ New best model! Accuracy: 0.8234

✅ Training Complete!
   Best Validation Accuracy: 82.34%
```

**Good accuracy:** 75-85%
**Excellent accuracy:** 85-95%

---

## 🔄 Continuous Improvement

After deploying:

1. **Collect misclassified images**
2. **Add to training dataset**
3. **Retrain periodically**
4. **Update cost estimates** (use update_costs.py)

---

## 📞 Need Help?

1. Check this guide first
2. Review error messages carefully
3. Ensure dataset is properly structured
4. Verify all dependencies installed
5. Check GPU availability: `torch.cuda.is_available()`

---

**Last Updated:** January 2026
