# Dataset Folder

This folder should contain your training dataset.

## Structure

```
datasets/
├── data.csv          # Image paths and labels (COMMITTED)
├── README.md         # This file (COMMITTED)
├── images/           # Image files (NOT COMMITTED - too large)
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── annotations/      # YOLO annotations (optional)
```

## How to Get Dataset

Since images are not committed (too large for GitHub), download them separately:

### Option 1: Kaggle
```bash
pip install kaggle
kaggle datasets download -d anujms/car-damage-detection
unzip car-damage-detection.zip -d images/
```

### Option 2: Roboflow
```bash
pip install roboflow
# Use your API key to download
```

### Option 3: Manual
1. Download vehicle damage images from any source
2. Place in `datasets/images/`
3. Ensure paths in `data.csv` match your folder structure

## data.csv Format

```csv
image,classes
images/train/img001.jpg,front_bumper_dent
images/train/img002.jpg,door_scratch
images/val/img100.jpg,hood_dent
```

## After Adding Dataset

Run training:
```bash
python train_improved.py
```

Or use the wizard:
```bash
python setup_wizard.py
```
