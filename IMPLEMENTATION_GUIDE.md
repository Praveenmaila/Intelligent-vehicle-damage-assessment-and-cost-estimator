# Intelligent Vehicle Damage Assessment System
## Complete Research Paper Implementation

**Based on:** *"Assessment of Intelligent Vehicle Damage and Cost Estimator for Insurance Companies"*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Modules](#modules)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [API Reference](#api-reference)
9. [Research Paper Alignment](#research-paper-alignment)

---

## 🎯 Overview

This project implements a complete end-to-end deep learning system for automated vehicle damage assessment, as described in the research paper. The system can:

- **Detect vehicle presence** in user-submitted images (98.9% accuracy)
- **Segment vehicle parts** into 13 categories (0.804 mIoU)
- **Localize and classify damage** into 3 types (0.463 mIoU)
- **Estimate damage severity** based on size and camera distance
- **Aggregate multi-view predictions** with confidence scoring
- **Flag low-confidence assessments** for human review

### Key Features

✅ **Modular Architecture** - Separate modules for detection, segmentation, and post-processing  
✅ **State-of-the-Art Models** - MobileNet, DeepLabV3+, EfficientNet-b5  
✅ **Extensive Augmentation** - 10+ augmentation techniques for robustness  
✅ **Multi-View Support** - Cross-view consistency checking  
✅ **Human-in-the-Loop** - Automatic flagging for manual review  
✅ **Comprehensive Metrics** - IoU, Dice, confusion matrices  
✅ **Production-Ready API** - Flask web service with batch processing  

---

## 🏗️ System Architecture

```
User Image(s)
    ↓
┌─────────────────────────────────────────┐
│  Module 1: Vehicle Detection            │
│  Architecture: MobileNetV2              │
│  Task: Binary classification            │
│  Output: Vehicle present (yes/no)       │
└─────────────────────────────────────────┘
    ↓ (if vehicle detected)
┌─────────────────────────────────────────┐
│  Module 2: Part Localization            │
│  Architecture: DeepLabV3+ + EffNet-b5   │
│  Task: Semantic segmentation (13 parts) │
│  Output: Part mask + probabilities      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Module 3: Damage Localization          │
│  Architecture: DeepLabV3+ + EffNet-b5   │
│  Task: Semantic segmentation (3 types)  │
│  Output: Damage mask + probabilities    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Module 4: Post-Processing              │
│  - Combine part + damage masks          │
│  - Estimate severity (size + distance)  │
│  - Calculate confidence scores          │
│  - Aggregate multi-view predictions     │
│  - Flag for human review                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Damage Assessment Report               │
│  - Damaged parts list                   │
│  - Damage types and severity            │
│  - Confidence scores                    │
│  - Review flags                         │
│  - Visualizations                       │
└─────────────────────────────────────────┘
```

---

## 📦 Modules

### 1. Vehicle Detector (`models/vehicle_detector.py`)

**Purpose:** Filter out non-vehicle images before processing.

**Architecture:** MobileNetV2
- **Input:** 224×224 RGB image
- **Output:** Binary classification (vehicle / no vehicle)
- **Performance:** 98.9% on OE Fleet, 91% on OEM user data

**Key Methods:**
- `predict(image)` → vehicle detection result
- `train_vehicle_detector()` → training function

### 2. Part Localizer (`models/part_localizer.py`)

**Purpose:** Segment vehicle into 13 anatomical parts.

**Architecture:** DeepLabV3+ with EfficientNet-b5 encoder
- **Input:** 512×512 RGB image
- **Output:** 14-class segmentation (13 parts + background)
- **Performance:** 0.804 mIoU on OE Fleet, 0.611 on OEM

**Vehicle Parts (13 classes):**
1. Hood
2. Front Bumper
3. Rear Bumper
4. Door Shell
5. Lamps (merged: front/fog/rear)
6. Mirror
7. Trunk
8. Fender
9. Grille
10. Wheel
11. Window
12. Windshield
13. Roof

**Key Methods:**
- `predict(image)` → part segmentation mask + probabilities
- `get_colored_mask(mask)` → visualization
- `train_part_localizer()` → training function

### 3. Damage Localizer (`models/damage_localizer.py`)

**Purpose:** Segment and classify damage regions.

**Architecture:** DeepLabV3+ with EfficientNet-b5 (joint model)
- **Input:** 512×512 RGB image
- **Output:** 4-class segmentation (3 damage types + no damage)
- **Performance:** 0.463 mIoU on OE Fleet, 0.392 on OEM

**Damage Types (3 categories):**
1. **Body Damage** - Dents, missing parts
2. **Surface Damage** - Scratches, paint chips, corrosion
3. **Deformity** - Cracks, shatters

**Key Methods:**
- `predict(image)` → damage segmentation mask + probabilities
- `get_colored_mask(mask)` → visualization
- `train_damage_localizer()` → training function

### 4. Post-Processor (`models/post_processor.py`)

**Purpose:** Combine predictions, estimate severity, aggregate views.

**Key Functions:**

**a) Mask Combination**
- Intersect part and damage masks
- Identify damaged vehicle parts
- Label connected components

**b) Severity Estimation**
- Use number of visible parts as camera distance proxy
- Adjust damage size based on zoom level
- Categorize: minor / moderate / major / severe

**c) Confidence Scoring**
- Average probabilities across damaged pixels
- Adjust based on multi-view agreement
- Flag if below threshold (default 0.7)

**d) Multi-View Aggregation**
- Build consensus across images
- Increase confidence for consistent predictions
- Decrease for disagreements

**Key Methods:**
- `combine_masks()` → damaged part instances
- `estimate_damage_size()` → severity estimation
- `aggregate_multi_view()` → multi-view fusion
- `flag_for_review()` → human-in-the-loop
- `generate_report()` → structured assessment
- `visualize_results()` → annotated images

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU)
- 8GB+ RAM (16GB recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/Praveenmaila/Intelligent-vehicle-damage-assessment-and-cost-estimator.git
cd vehicle_damage_detection/vehicle_damage_detection/vehicle_damage_detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights

```bash
# Option 1: Download pretrained weights (if available)
python setup_models.py

# Option 2: Train from scratch (see Training section)
```

---

## 💻 Usage

### Quick Assessment (Single Image)

```python
from integrated_system import quick_assess

# Assess damage in single image
result = quick_assess(
    'path/to/damaged_car.jpg',
    save_output=True,
    output_dir='results'
)

# Print summary
if result['success']:
    summary = result['assessment']['summary']
    print(f"Damages detected: {summary['total_damages_detected']}")
    print(f"Confidence: {summary['average_confidence']}%")
    print(f"Needs review: {summary['needs_human_review']}")
    
    # Print each damage
    for damage in result['assessment']['damages']:
        print(f"{damage['part']}: {damage['damage_type']} - {damage['severity']}")
```

### Multi-View Assessment

```python
from integrated_system import IntegratedDamageAssessor
from PIL import Image

# Initialize system
assessor = IntegratedDamageAssessor(
    confidence_threshold=0.7
)

# Load multiple views
images = [
    Image.open('front_view.jpg'),
    Image.open('side_view.jpg'),
    Image.open('rear_view.jpg')
]

# Assess with multi-view aggregation
result = assessor.assess_multiple_views(images)

# Get aggregated report
if result['success']:
    report = result['assessment']
    print(f"Views processed: {report['summary']['num_images']}")
    print(f"Total damages: {report['summary']['total_damages_detected']}")
```

### Web API Usage

```bash
# Start Flask server
python app.py
```

```python
import requests

# Single image assessment
files = {'image': open('damaged_car.jpg', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

# Batch assessment
files = [
    ('images', open('view1.jpg', 'rb')),
    ('images', open('view2.jpg', 'rb'))
]
response = requests.post('http://localhost:5000/batch-predict', files=files)
results = response.json()
```

---

## 🎓 Training

### Dataset Structure

```
datasets/
├── vehicle_detection/
│   ├── train/
│   │   ├── vehicle/
│   │   └── no_vehicle/
│   └── val/
├── segmentation/
│   ├── images/
│   ├── part_masks/
│   └── damage_masks/
└── data.csv
```

### Train Vehicle Detector

```python
from models.vehicle_detector import train_vehicle_detector
from torch.utils.data import DataLoader

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train
model = train_vehicle_detector(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=0.001,
    save_path='vehicle_detector.pth'
)
```

### Train Part Localizer

```python
from models.part_localizer import train_part_localizer

model = train_part_localizer(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=0.0001,
    save_path='part_localizer.pth'
)
```

### Train Damage Localizer

```python
from models.damage_localizer import train_damage_localizer

model = train_damage_localizer(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=0.0001,
    save_path='damage_localizer.pth',
    joint=True  # Joint training with parts
)
```

---

## 📊 Evaluation

### Compute Metrics

```python
from utils.metrics import (
    evaluate_segmentation_model,
    compute_vehicle_detection_metrics
)

# Evaluate segmentation
results = evaluate_segmentation_model(
    model=part_localizer,
    data_loader=test_loader,
    device=device,
    num_classes=14,
    class_names=list(VEHICLE_PART_CLASSES.values())
)

print(f"mIoU: {results['mean_iou']:.4f}")
print(f"Dice: {results['mean_dice']:.4f}")
print(f"Pixel Acc: {results['pixel_accuracy']:.4f}")

# Evaluate detection
det_results = compute_vehicle_detection_metrics(
    model=vehicle_detector,
    data_loader=test_loader,
    device=device
)

print(f"Accuracy: {det_results['accuracy']:.4f}")
print(f"Precision: {det_results['precision']:.4f}")
print(f"Recall: {det_results['recall']:.4f}")
```

---

## 📚 Research Paper Alignment

### Implemented Components

| Paper Component | Implementation | Status |
|----------------|----------------|--------|
| Vehicle Detection (MobileNet) | `models/vehicle_detector.py` | ✅ Complete |
| Part Localization (DeepLabV3+) | `models/part_localizer.py` | ✅ Complete |
| Damage Localization (DeepLabV3+) | `models/damage_localizer.py` | ✅ Complete |
| Dice Loss | `models/part_localizer.py`, `damage_localizer.py` | ✅ Complete |
| Data Augmentation Pipeline | `utils/augmentation.py` | ✅ Complete |
| Post-Processing | `models/post_processor.py` | ✅ Complete |
| Multi-View Aggregation | `integrated_system.py` | ✅ Complete |
| Confidence Scoring | `post_processor.py` | ✅ Complete |
| Human-in-the-Loop | `post_processor.py` | ✅ Complete |
| Evaluation Metrics (IoU, Dice) | `utils/metrics.py` | ✅ Complete |
| Confusion Matrix Analysis | `utils/metrics.py` | ✅ Complete |

### Architecture Comparison

| Component | Paper Specification | Our Implementation |
|-----------|-------------------|-------------------|
| Vehicle Detection | MobileNet | MobileNetV2 ✅ |
| Part Segmentation | DeepLabV3+ + EfficientNet-b5 | DeepLabV3+ + EfficientNet-b5 ✅ |
| Damage Segmentation | DeepLabV3+ + EfficientNet-b5 | DeepLabV3+ + EfficientNet-b5 ✅ |
| Loss Function | Dice Loss | Dice Loss ✅ |
| Optimizer | Adam + Cosine Annealing | Adam + Cosine Annealing ✅ |
| Batch Size | 4 | 4 ✅ |
| Input Size (Segmentation) | 512×512 | 512×512 ✅ |
| Input Size (Detection) | 224×224 | 224×224 ✅ |

### Performance Targets

| Metric | Paper Target | Our Implementation |
|--------|-------------|-------------------|
| Vehicle Detection Accuracy (OE) | 98.9% | Matches architecture |
| Vehicle Detection Accuracy (OEM) | 91% | Matches architecture |
| Part mIoU (OE) | 0.804 | Matches architecture |
| Part mIoU (OEM) | 0.611 | Matches architecture |
| Damage mIoU (OE) | 0.463 | Matches architecture |
| Damage mIoU (OEM) | 0.392 | Matches architecture |

---

## 📄 License

MIT License - See LICENSE file

## 👥 Contributors

- Research Paper Authors
- Implementation: [Your Name]

## 📞 Contact

For questions or issues, please open a GitHub issue or contact: [your-email@example.com]

---

**Note:** This implementation faithfully follows the research paper methodology. Model weights need to be trained on appropriate datasets to achieve reported performance metrics.
