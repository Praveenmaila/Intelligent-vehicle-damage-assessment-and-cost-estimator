# 🚀 Installation & Startup Commands

## Step-by-Step Setup

### 1️⃣ Install Dependencies (Required)
```powershell
cd "c:\Users\maila\Desktop\major project c13\vehicle_damage_detection\vehicle_damage_detection\vehicle_damage_detection"
pip install -r requirements.txt
```

**What this installs:**
- torch, torchvision (Deep learning)
- ultralytics (YOLO models)
- opencv-python (Image processing)
- Flask (Web server)
- numpy, scipy, matplotlib, pandas (Data processing)

**Expected time:** 2-5 minutes

---

### 2️⃣ Setup Models (Recommended)
```powershell
python setup_models.py
```

**What this does:**
- Downloads YOLOv11 pretrained models
- Checks for ResNet50 model
- Creates necessary directories
- Generates test images

**Expected time:** 1-2 minutes

---

### 3️⃣ Test Integration (Optional but Recommended)
```powershell
python test_integration.py
```

**What this tests:**
- All dependencies are installed
- Model inference works
- Flask app is configured correctly

**Expected time:** 10-30 seconds

---

### 4️⃣ Start the Server
```powershell
python app.py
```

**Server will start on:** http://localhost:5000

**Expected output:**
```
============================================================
Vehicle Damage Detection - Flask Web Application
ResNet50 + YOLO Hybrid Detection System
============================================================

🔧 Initializing Vehicle Damage Detection System
...
✓ ResNet50 loaded
✓ YOLO loaded

============================================================
Starting Flask server...
============================================================

🌐 Open your browser and go to: http://localhost:5000
```

---

## 🧪 Testing Commands

### Test Web UI
```powershell
# Server must be running
start http://localhost:5000
```

### Test Health Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get | ConvertTo-Json
```

### Test Model Info
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/model-info" -Method Get | ConvertTo-Json
```

### Test Prediction (if test_vehicle.jpg exists)
```powershell
$file = "test_vehicle.jpg"
$uri = "http://localhost:5000/predict"
$fileBytes = [System.IO.File]::ReadAllBytes($file)
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"
$bodyLines = (
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"test_vehicle.jpg`"",
    "Content-Type: image/jpeg$LF",
    [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
    "--$boundary--$LF"
) -join $LF
Invoke-RestMethod -Uri $uri -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $bodyLines
```

---

## 🔧 Troubleshooting Commands

### Check Python Version
```powershell
python --version
```
**Required:** Python 3.10+

### Check Installed Packages
```powershell
pip list | Select-String -Pattern "torch|flask|ultralytics|opencv"
```

### Reinstall Dependencies
```powershell
pip install --upgrade --force-reinstall -r requirements.txt
```

### Check if Port 5000 is Available
```powershell
netstat -ano | findstr :5000
```

### Kill Process on Port 5000 (if needed)
```powershell
# Find PID from above command, then:
taskkill /PID <PID> /F
```

---

## 📦 Quick Commands Reference

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Setup models | `python setup_models.py` |
| Test integration | `python test_integration.py` |
| Start server | `python app.py` |
| Check health | `Invoke-RestMethod -Uri "http://localhost:5000/health"` |
| Stop server | `Ctrl+C` in terminal |

---

## 🎯 One-Line Complete Setup

```powershell
pip install -r requirements.txt; python setup_models.py; python test_integration.py; python app.py
```

**This will:**
1. Install all dependencies
2. Download YOLO models
3. Run tests
4. Start the server

---

## 📝 Notes

- **First run** may take longer due to model downloads
- **GPU support** is automatic if CUDA is available
- **Models work independently**: ResNet-only or YOLO-only is fine
- **Port 5000** must be available
- **Internet required** for initial YOLO download

---

## ✅ Success Indicators

After `python app.py`, you should see:
- ✓ ResNet50 loaded (or warning if missing)
- ✓ YOLO loaded (or downloading)
- ✓ Flask server starting
- 🌐 URL: http://localhost:5000

Then visit the URL and upload an image!

---

## 🆘 Common Issues

### "No module named 'ultralytics'"
```powershell
pip install ultralytics
```

### "torch not found"
```powershell
pip install torch torchvision
```

### "Port 5000 already in use"
```powershell
# Either kill the process or change port in app.py
# Find the line: app.run(debug=True, host='0.0.0.0', port=5000)
# Change to: app.run(debug=True, host='0.0.0.0', port=5001)
```

### "ResNet model not found" (Warning only)
```
⚠️ This is OK! The system will work with YOLO-only detection.
To get ResNet model:
1. Train it: python train.py
2. Or download from Kaggle (see README_INTEGRATION.md)
```

---

## 🎓 What Each File Does

| File | Purpose |
|------|---------|
| `app.py` | Flask web server |
| `model_inference.py` | ResNet50 + YOLO detection logic |
| `setup_models.py` | Download and setup models |
| `test_integration.py` | Verify everything works |
| `train.py` | Train ResNet50 model |
| `requirements.txt` | Python dependencies |

---

**Ready to start?** Run the commands above in order! 🚀
