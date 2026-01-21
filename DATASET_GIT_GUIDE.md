# 📦 Managing Datasets with Git

## The Problem
Dataset images are typically **too large** for GitHub:
- GitHub file size limit: 100 MB per file
- Repository size recommendations: < 1 GB
- A typical vehicle damage dataset: 500 MB - 5 GB

## ✅ Solution Applied

Your `.gitignore` is now configured to:
- ✅ **COMMIT:** CSV files (`data.csv`) - Small, text-based
- ✅ **COMMIT:** Dataset structure (folders, README)
- ✅ **COMMIT:** Training scripts
- ❌ **IGNORE:** Image files (*.jpg, *.png, etc.) - Too large
- ❌ **IGNORE:** Model weights (*.pth, *.pt) - Too large

## 📁 What Gets Committed

```
✅ datasets/data.csv           # Image paths and labels
✅ datasets/README.md          # Instructions
✅ train_improved.py           # Training script
✅ download_dataset.py         # Dataset downloader
✅ setup_wizard.py             # Setup automation
❌ datasets/images/            # Actual images (too large)
❌ vehicle_damage_model.pth    # Trained model (too large)
```

## 🔄 Workflow for Team Collaboration

### Person A (You) - Initial Setup
```powershell
# 1. Train model locally with your dataset
python train_improved.py

# 2. Commit code and CSV (not images)
git add .
git commit -m "Add training pipeline and data.csv"
git push origin main
```

### Person B (Teammate) - Clone and Setup
```powershell
# 1. Clone repository
git clone <your-repo-url>
cd vehicle_damage_detection

# 2. Download dataset using provided script
python download_dataset.py
# Or manually download from Kaggle/Roboflow

# 3. Train their own model
python train_improved.py

# 4. Start using the app
python app.py
```

## 🌐 Sharing Large Files

### Option 1: Git LFS (Large File Storage)
For files 100 MB - 2 GB:

```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes

# Now you can commit large model files
git add vehicle_damage_model.pth
git commit -m "Add trained model via LFS"
git push origin main
```

**Cost:** Free tier: 1 GB storage + 1 GB bandwidth/month

### Option 2: Cloud Storage Links
For files > 2 GB:

**Upload to:**
- Google Drive
- Dropbox
- OneDrive
- AWS S3
- Azure Blob Storage

**Add download link to README:**
```markdown
## Download Pre-trained Model
[Download vehicle_damage_model.pth (150 MB)](https://drive.google.com/file/d/xxx)

Place in project root folder.
```

### Option 3: Kaggle Datasets
**Best for datasets:**

```powershell
# Upload your dataset to Kaggle
kaggle datasets init -p datasets/
# Edit dataset-metadata.json
kaggle datasets create -p datasets/

# Share dataset link in README
# Others can download with:
kaggle datasets download -d yourusername/your-dataset
```

### Option 4: Hugging Face Hub
**Best for ML models:**

```python
# Install
pip install huggingface_hub

# Upload model
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="vehicle_damage_model.pth",
    path_in_repo="vehicle_damage_model.pth",
    repo_id="yourusername/vehicle-damage-detector",
    repo_type="model"
)

# Download (by others)
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="yourusername/vehicle-damage-detector",
    filename="vehicle_damage_model.pth"
)
```

## 📝 Update Your Repository README

Add this section to your main README:

```markdown
## 📥 Dataset Setup

The dataset images are not included in this repository due to size constraints.

### Option 1: Download Pre-prepared Dataset
Download from: [Link to your Google Drive/Kaggle]

Extract to `datasets/images/`

### Option 2: Use Public Dataset
```bash
pip install kaggle
kaggle datasets download -d anujms/car-damage-detection
unzip car-damage-detection.zip -d datasets/images/
```

### Option 3: Use Your Own Images
1. Place images in `datasets/images/`
2. Update `datasets/data.csv` with correct paths
3. Run `python train_improved.py`

## 🤖 Pre-trained Model

Download pre-trained model (optional):
[vehicle_damage_model.pth (150 MB)](your-link-here)

Or train your own:
```bash
python train_improved.py
```
```

## 🔒 Current .gitignore Explained

```gitignore
# Python
venv/                          # Virtual environment
__pycache__/                   # Python cache
*.pyc                          # Compiled Python

# Dataset images (too large)
datasets/images/               # All images in images folder
datasets/train/                # Training images
datasets/val/                  # Validation images
datasets/test/                 # Test images
datasets/**/*.jpg              # All JPG files in any subdirectory
datasets/**/*.jpeg
datasets/**/*.png
datasets/**/*.bmp
datasets/**/*.gif

# Keep CSV files (small)
!datasets/*.csv                # Allow CSV files in datasets/
!datasets/data.csv             # Explicitly allow data.csv

# User uploads
uploads/                       # User uploaded images

# Model files (too large)
*.pth                          # PyTorch models
*.pt                           # PyTorch/YOLO models
!yolo11n.pt                    # Exception: keep small YOLO model

# Environment
.env                           # Environment variables
```

## ✅ Verify Your Setup

```powershell
# Check what will be committed
git status

# Should see:
# ✅ datasets/data.csv
# ✅ datasets/README.md
# ❌ Not seeing datasets/images/ (correct!)
# ❌ Not seeing *.pth files (correct!)

# Check gitignore is working
git check-ignore datasets/images/test.jpg
# Should output: datasets/images/test.jpg (means it's ignored)

git check-ignore datasets/data.csv
# Should output nothing (means it will be committed)
```

## 🚀 Push to Remote

```powershell
# Push your changes
git push origin main

# Verify on GitHub
# ✅ You should see: datasets/data.csv
# ✅ You should see: training scripts
# ❌ You should NOT see: datasets/images/
# ❌ You should NOT see: *.pth files
```

## 💡 Best Practices

1. **Always commit CSV/metadata** - Small and essential
2. **Never commit large images** - Use cloud storage
3. **Document data sources** - Tell others where to get data
4. **Use Git LFS for models < 2GB** - If you want to share trained models
5. **Create download scripts** - Automate dataset setup for teammates
6. **Test clean clone** - Verify others can reproduce your setup

## 🆘 If You Accidentally Committed Large Files

```powershell
# Remove from Git history (before pushing)
git rm --cached datasets/images/*.jpg
git commit -m "Remove large image files"

# If already pushed, use BFG Repo Cleaner
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --strip-blobs-bigger-than 10M
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

## 📊 Summary

| Item | Size | Action | Tool |
|------|------|--------|------|
| CSV files | < 1 MB | ✅ Commit | Git |
| Training scripts | < 1 MB | ✅ Commit | Git |
| Images (dataset) | 500 MB - 5 GB | 🌐 Share link | Kaggle/Drive |
| Model weights | 50-200 MB | 🌐 Share link | Git LFS/Hugging Face |
| Code & docs | < 10 MB | ✅ Commit | Git |

---

**Your current setup is correct!** 
- CSV files will be committed ✅
- Images will be ignored ❌ (good!)
- Team can download images separately 🌐
