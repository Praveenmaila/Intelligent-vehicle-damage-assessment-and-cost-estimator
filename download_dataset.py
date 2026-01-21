# -------------------------------
# Dataset Downloader and Preparation Script
# Downloads real vehicle damage datasets from multiple sources
# -------------------------------

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json
import pandas as pd

def create_directories():
    """Create necessary directory structure"""
    dirs = [
        'datasets',
        'datasets/train',
        'datasets/val',
        'datasets/test',
        'datasets/images',
        'datasets/annotations'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created dataset directories")

def download_coco_vehicle_dataset():
    """
    Download vehicle damage images from COCO dataset subset
    This uses a curated subset focusing on vehicles and damage
    """
    print("\n📥 Downloading COCO Vehicle Damage Dataset...")
    print("This may take 5-10 minutes depending on your connection.\n")
    
    # Use Roboflow public datasets (vehicle damage detection)
    dataset_url = "https://universe.roboflow.com/ds/DATASET_ID"
    
    print("Note: For best results, manually download vehicle damage datasets from:")
    print("1. Roboflow Universe: https://universe.roboflow.com/")
    print("   Search for: 'vehicle damage detection'")
    print("2. Kaggle: https://www.kaggle.com/datasets")
    print("   Search for: 'car damage detection' or 'vehicle damage assessment'")
    print("\nRecommended datasets:")
    print("- Car Damage Detection (Kaggle)")
    print("- Vehicle Damage Assessment (Roboflow)")
    print("- Car Parts Segmentation (Kaggle)")
    
    return False

def create_sample_dataset():
    """
    Create a sample dataset structure with placeholder data
    This can be used for testing before getting real data
    """
    print("\n🔨 Creating sample dataset structure...")
    
    # Sample data CSV
    sample_data = {
        'image': [],
        'classes': []
    }
    
    # Create sample entries
    damage_types = [
        'front_bumper_dent', 'rear_bumper_scratch', 'door_dent', 
        'hood_scratch', 'head_lamp', 'windshield_crack', 
        'side_mirror_broken', 'wheel_rim_scratch'
    ]
    
    for i, damage in enumerate(damage_types):
        for j in range(5):  # 5 samples per class
            sample_data['image'].append(f'train/{damage}_{j}.jpg')
            sample_data['classes'].append(damage)
    
    # Save CSV
    df = pd.DataFrame(sample_data)
    df.to_csv('datasets/data.csv', index=False)
    print(f"✓ Created sample dataset CSV with {len(df)} entries")
    print(f"✓ Saved to: datasets/data.csv")
    
    return True

def download_kaggle_dataset(dataset_name, api_key=None):
    """
    Download dataset from Kaggle using API
    Requires: pip install kaggle
    """
    print(f"\n📥 Downloading from Kaggle: {dataset_name}")
    
    try:
        import kaggle
        
        # Download and extract
        kaggle.api.dataset_download_files(
            dataset_name,
            path='datasets/',
            unzip=True
        )
        print(f"✓ Downloaded {dataset_name}")
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed. Install with: pip install kaggle")
        print("📖 Setup instructions:")
        print("1. Create Kaggle account: https://www.kaggle.com/")
        print("2. Go to Account -> API -> Create New API Token")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def setup_roboflow_dataset(api_key, workspace, project, version=1):
    """
    Download dataset from Roboflow
    Requires: pip install roboflow
    """
    print(f"\n📥 Downloading from Roboflow: {workspace}/{project}")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov8", location="datasets/")
        
        print(f"✓ Downloaded Roboflow dataset")
        return True
        
    except ImportError:
        print("❌ Roboflow not installed. Install with: pip install roboflow")
        return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def prepare_coco_format_data(images_dir, annotations_file):
    """Convert COCO format annotations to our training format"""
    print("\n🔄 Converting COCO format to training format...")
    
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Extract image and annotation info
        data_rows = []
        for img in coco_data['images']:
            img_id = img['id']
            img_filename = img['file_name']
            
            # Find annotations for this image
            anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            if anns:
                # Get category (damage type)
                cat_id = anns[0]['category_id']
                category = next((cat for cat in coco_data['categories'] if cat['id'] == cat_id), None)
                
                if category:
                    data_rows.append({
                        'image': f"images/{img_filename}",
                        'classes': category['name']
                    })
        
        # Save as CSV
        df = pd.DataFrame(data_rows)
        df.to_csv('datasets/data.csv', index=False)
        print(f"✓ Converted {len(df)} annotations")
        return True
        
    except Exception as e:
        print(f"❌ Error converting COCO data: {e}")
        return False

def main():
    """Main setup function"""
    print("="*70)
    print("🚗 Vehicle Damage Detection Dataset Setup")
    print("="*70)
    
    # Create directory structure
    create_directories()
    
    print("\n📋 Choose dataset source:")
    print("1. Create sample dataset (for testing)")
    print("2. Download from Kaggle (requires API key)")
    print("3. Download from Roboflow (requires API key)")
    print("4. Manual setup instructions")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        create_sample_dataset()
        print("\n⚠️  WARNING: Sample dataset is for testing only!")
        print("For production use, download real vehicle damage images.")
        
    elif choice == '2':
        print("\n📦 Recommended Kaggle datasets:")
        print("1. 'anujms/car-damage-detection' - 920 images")
        print("2. 'lplenka/car-damage-detection' - 1000+ images")
        print("3. 'farzadnekouei/car-damage-detection' - Multiple angles")
        
        dataset = input("\nEnter dataset name (e.g., anujms/car-damage-detection): ").strip()
        if dataset:
            download_kaggle_dataset(dataset)
        
    elif choice == '3':
        print("\n🌐 Setting up Roboflow dataset...")
        print("Get your API key from: https://app.roboflow.com/settings/api")
        
        api_key = input("Enter Roboflow API key: ").strip()
        workspace = input("Enter workspace name: ").strip()
        project = input("Enter project name: ").strip()
        
        if api_key and workspace and project:
            setup_roboflow_dataset(api_key, workspace, project)
    
    else:
        print("\n📖 Manual Dataset Setup Instructions:")
        print("="*70)
        print("\n1. Download vehicle damage dataset from:")
        print("   - Kaggle: https://www.kaggle.com/datasets")
        print("     Search: 'car damage detection'")
        print("   - Roboflow: https://universe.roboflow.com/")
        print("     Search: 'vehicle damage detection'")
        print("\n2. Extract images to: datasets/images/")
        print("\n3. Create datasets/data.csv with columns:")
        print("   - image: path to image (e.g., train/img001.jpg)")
        print("   - classes: damage type (e.g., front_bumper_dent)")
        print("\n4. Supported damage classes:")
        print("   front_bumper_dent, rear_bumper_scratch, door_dent,")
        print("   hood_scratch, head_lamp, windshield_crack, etc.")
        print("\n5. Run: python train.py")
        print("="*70)
        
        # Create sample CSV template
        create_sample_dataset()
        print("\n✓ Created template CSV at: datasets/data.csv")
        print("  Edit this file with your actual image paths and labels")
    
    print("\n" + "="*70)
    print("✅ Setup complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Verify datasets/data.csv exists and has correct paths")
    print("2. Ensure images are in datasets/ directory")
    print("3. Run: python train.py")
    print("4. After training, run: python app.py")
    print("="*70)

if __name__ == "__main__":
    main()
