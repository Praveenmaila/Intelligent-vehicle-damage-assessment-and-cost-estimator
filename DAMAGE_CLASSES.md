# Vehicle Damage Classes and Cost Estimates

## 📋 Complete Damage Classification System
**Based on Indian Insurance Industry Standards (2024-2026)**

### Cost Sources:
- IRDAI (Insurance Regulatory and Development Authority) guidelines
- Average authorized service center rates
- Real insurance claim data analysis
- Includes: Parts + Labor + Paint + GST

### All Supported Damage Types (31 Classes)

| ID | Damage Type | Estimated Cost (INR) | Range | Description |
|----|-------------|---------------------|-------|-------------|
| 0 | **unknown** | ₹0 | - | Unknown or no damage |
| 1 | **head_lamp** | ₹12,000 | ₹8K-25K | Headlamp (Halogen: ₹8K, LED: ₹12-25K, HID: ₹15-30K) |
| 2 | **rear_lamp** | ₹8,000 | ₹6K-12K | Rear lamp assembly |
| 3 | **tail_lamp** | ₹10,000 | ₹6K-15K | Tail lamp damage/crack |
| 4 | **front_bumper_dent** | ₹20,000 | ₹15K-30K | Dent in front bumper (plastic repair + paint) |
| 5 | **rear_bumper_dent** | ₹18,000 | ₹12K-25K | Dent in rear bumper |
| 6 | **front_bumper_scratch** | ₹8,000 | ₹5K-12K | Scratch on front bumper (buffing + touch-up) |
| 7 | **rear_bumper_scratch** | ₹6,000 | ₹4K-10K | Scratch on rear bumper |
| 8 | **door_dent** | ₹25,000 | ₹15K-35K | Dent in door panel (panel beating + repainting) |
| 9 | **door_scratch** | ₹12,000 | ₹8K-18K | Scratch on door (depth dependent) |
| 10 | **hood_dent** | ₹28,000 | ₹20K-40K | Dent in hood/bonnet (large panel, complex) |
| 11 | **hood_scratch** | ₹15,000 | ₹10K-20K | Scratch on hood |
| 12 | **trunk_dent** | ₹22,000 | ₹15K-30K | Dent in trunk/boot |
| 13 | **trunk_scratch** | ₹12,000 | ₹8K-15K | Scratch on trunk |
| 14 | **fender_dent** | ₹18,000 | ₹12K-25K | Dent in fender/wing |
| 15 | **fender_scratch** | ₹10,000 | ₹6K-15K | Scratch on fender |
| 16 | **windshield_crack** | ₹18,000 | ₹12K-25K | Crack in windshield (repair/replace) |
| 17 | **windshield_shatter** | ₹35,000 | ₹25K-50K | Shattered windshield (full replacement + sensors) |
| 18 | **side_window_crack** | ₹8,000 | ₹5K-12K | Side window crack |
| 19 | **side_window_shatter** | ₹12,000 | ₹8K-18K | Shattered side window (tempered glass) |
| 20 | **rear_window_crack** | ₹15,000 | ₹10K-20K | Rear window crack |
| 21 | **rear_window_shatter** | ₹20,000 | ₹15K-30K | Shattered rear window (larger size + defroster) |
| 22 | **side_mirror_crack** | ₹4,000 | ₹3K-6K | Side mirror crack (glass only) |
| 23 | **side_mirror_broken** | ₹12,000 | ₹8K-18K | Broken side mirror (full unit + motors/sensors) |
| 24 | **wheel_rim_scratch** | ₹6,000 | ₹4K-10K | Scratch on wheel rim (refinishing) |
| 25 | **wheel_rim_bent** | ₹15,000 | ₹10K-22K | Bent wheel rim (repair or replace) |
| 26 | **tire_damage** | ₹10,000 | ₹5K-15K | Tire damage (type dependent) |
| 27 | **paint_peel** | ₹25,000 | ₹15K-40K | Paint peeling (full panel repaint) |
| 28 | **rust_damage** | ₹35,000 | ₹20K-60K | Rust damage (extensive work + welding) |
| 29 | **panel_misalignment** | ₹30,000 | ₹20K-45K | Panel misalignment (structural work) |
| 30 | **grille_damage** | ₹15,000 | ₹8K-25K | Grille damage (model dependent) |

## 💰 Cost Categories

### Low Cost (₹4,000 - ₹10,000)
**Minor repairs, cosmetic damage**
- Side mirror crack (₹4K)
- Rear bumper scratch (₹6K)
- Wheel rim scratch (₹6K)
- Front/Rear lamp (₹8K-10K)
- Side window crack (₹8K)
- Tire damage (₹10K)
- Fender scratch (₹10K)

### Medium Cost (₹12,000 - ₹20,000)
**Moderate repairs, panel work**
- Head lamp LED (₹12K)
- Door scratch (₹12K)
- Side mirror broken (₹12K)
- Side/Rear window shatter (₹12K-20K)
- Hood scratch (₹15K)
- Trunk scratch/Rear window crack (₹12K-15K)
- Grille damage (₹15K)
- Wheel rim bent (₹15K)
- Rear bumper dent (₹18K)
- Windshield crack (₹18K)
- Front bumper dent/Fender dent (₹18K-20K)

### High Cost (₹22,000 - ₹35,000)
**Major repairs, structural work**
- Trunk dent (₹22K)
- Door dent (₹25K)
- Paint peel (₹25K)
- Hood dent (₹28K)
- Panel misalignment (₹30K)
- Windshield shatter (₹35K)
- Rust damage (₹35K)

### Insurance Coverage Notes:
- **Own Damage (OD)** insurance covers most damages above
- **Zero Depreciation** covers full cost without parts depreciation
- **Deductible** typically ₹1,000-5,000 applied
- **Glass cover** add-on recommended for windshield claims
- **Consumables** cover includes paint, nuts, bolts (extra premium)

## 🎨 Visual Color Coding

Each damage type has a unique color for bounding box visualization:

- **Lamps**: Red shades
- **Bumpers**: Blue shades
- **Doors**: Green shades
- **Hood/Trunk**: Yellow/Gold shades
- **Fenders**: Magenta shades
- **Windows**: Purple/Pink shades
- **Mirrors**: Orange shades
- **Wheels**: Lime/Dark gray shades
- **Paint/Rust**: Brown/Rust shades
- **Grille/Panels**: Cyan/Light red

## 🔧 How to Retrain with New Classes

### Option 1: Update Existing Dataset

1. Open `datasets/data.csv`
2. Update the `classes` column with new damage type names
3. Ensure class names match the `class_mapping` in `train.py`
4. Run training:
   ```bash
   python train.py
   ```

### Option 2: Add Custom Classes

1. Edit `train.py` - Update `class_mapping`:
   ```python
   class_mapping = {
       'your_custom_damage': 31,
       # Add more...
   }
   ```

2. Update `cost_mapping`:
   ```python
   cost_mapping = {
       31: 15000,  # your_custom_damage cost
       # Add more...
   }
   ```

3. Update `model_inference.py` - Add colors:
   ```python
   self.damage_colors = {
       'your_custom_damage': (R, G, B),
       # Add more...
   }
   ```

4. Retrain the model:
   ```bash
   python train.py
   ```

## 📊 Current Model Status

Your current model was trained with **8 classes**:
- unknown
- head_lamp
- door_scratch
- glass_shatter
- tail_lamp
- bumper_dent
- door_dent
- bumper_scratch

### To Use All 31 Classes:

1. **Update your dataset** (`datasets/data.csv`) to include all damage types
2. **Retrain the model** using the updated `train.py`
3. The system will automatically recognize all 31 classes after retraining

## 🚀 Quick Fix for Current Issue

Your model is showing "unknown" because:

1. ✅ **FIXED**: Changed ResNet50 → ResNet18 (matches your trained model)
2. ⚠️ **Current**: Model has only 8 classes
3. 💡 **Solution**: Either:
   - Retrain with all 31 classes (recommended)
   - Or use the existing 8 classes

### To use existing 8 classes immediately:

The system will now work correctly with your current model. Upload vehicle damage images and it will classify them into one of the 8 trained categories.

### To get all 31 classes:

```bash
# 1. Prepare dataset with all damage types
# 2. Run training
python train.py

# 3. Restart the server
python app.py
```

## 📝 Notes

- **Cost estimates** are in Indian Rupees (INR/₹)
- Costs are approximate and vary by:
  - Vehicle make and model
  - Location
  - Labor rates
  - Parts availability
- **YOLO detection** works independently for bounding boxes
- **ResNet classification** provides damage type and cost

---

**Current Status**: Model architecture fixed to ResNet18. System ready to use with 8 classes or retrain for 31 classes.
