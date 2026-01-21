# -------------------------------
# Intelligent Vehicle Damage Classifier (7 Classes, with Debug Prints)
# -------------------------------

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# -------------------------------
# 2. Dataset Class (safe version)
# -------------------------------
class VehicleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        # Filter out missing files right at initialization
        self.dataframe = dataframe.copy()
        self.dataframe['exists'] = self.dataframe['image'].apply(lambda x: os.path.exists(x))
        missing_count = (~self.dataframe['exists']).sum()
        if missing_count > 0:
            print(f"WARNING: {missing_count} images missing, they will be skipped.")
        self.dataframe = self.dataframe[self.dataframe['exists']].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 3. Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Main training script
# -------------------------------
if __name__ == '__main__':  # Required for Windows multiprocessing
    # -------------------------------
    # 1. Load Dataset
    # -------------------------------
    csv_path = r'datasets/data.csv'
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total records: {len(df)}")
    print("Sample data:")
    print(df.head())
    
    # Fix image paths - prepend 'datasets/' to make them correct
    df['image'] = df['image'].apply(lambda x: os.path.join('datasets', x))
    print(f"Image paths corrected. Sample path: {df['image'].iloc[0]}")

    # Map classes to numeric labels - COMPREHENSIVE DAMAGE CLASSES
    class_mapping = {
        'unknown': 0,
        'head_lamp': 1,
        'rear_lamp': 2,
        'tail_lamp': 3,
        'front_bumper_dent': 4,
        'rear_bumper_dent': 5,
        'front_bumper_scratch': 6,
        'rear_bumper_scratch': 7,
        'door_dent': 8,
        'door_scratch': 9,
        'hood_dent': 10,
        'hood_scratch': 11,
        'trunk_dent': 12,
        'trunk_scratch': 13,
        'fender_dent': 14,
        'fender_scratch': 15,
        'windshield_crack': 16,
        'windshield_shatter': 17,
        'side_window_crack': 18,
        'side_window_shatter': 19,
        'rear_window_crack': 20,
        'rear_window_shatter': 21,
        'side_mirror_crack': 22,
        'side_mirror_broken': 23,
        'wheel_rim_scratch': 24,
        'wheel_rim_bent': 25,
        'tire_damage': 26,
        'paint_peel': 27,
        'rust_damage': 28,
        'panel_misalignment': 29,
        'grille_damage': 30
    }
    
    # Realistic cost estimates in INR - Based on insurance industry standards (2024-2026)
    # Sources: IRDAI guidelines, average garage rates, insurance claim data
    cost_mapping = {
        0: 0,        # unknown - no damage
        1: 12000,    # head_lamp - Halogen: ₹8K, LED: ₹12-25K, HID: ₹15-30K
        2: 8000,     # rear_lamp - ₹6K-12K
        3: 10000,    # tail_lamp - ₹6K-15K
        4: 20000,    # front_bumper_dent - ₹15K-30K (plastic repair + paint)
        5: 18000,    # rear_bumper_dent - ₹12K-25K
        6: 8000,     # front_bumper_scratch - ₹5K-12K (buffing + touch-up)
        7: 6000,     # rear_bumper_scratch - ₹4K-10K
        8: 25000,    # door_dent - ₹15K-35K (panel beating + repainting)
        9: 12000,    # door_scratch - ₹8K-18K (depends on depth)
        10: 28000,   # hood_dent - ₹20K-40K (large panel, complex repair)
        11: 15000,   # hood_scratch - ₹10K-20K
        12: 22000,   # trunk_dent - ₹15K-30K
        13: 12000,   # trunk_scratch - ₹8K-15K
        14: 18000,   # fender_dent - ₹12K-25K
        15: 10000,   # fender_scratch - ₹6K-15K
        16: 18000,   # windshield_crack - ₹12K-25K (repair if small, replace if large)
        17: 35000,   # windshield_shatter - ₹25K-50K (full replacement + sensors)
        18: 8000,    # side_window_crack - ₹5K-12K
        19: 12000,   # side_window_shatter - ₹8K-18K (tempered glass)
        20: 15000,   # rear_window_crack - ₹10K-20K
        21: 20000,   # rear_window_shatter - ₹15K-30K (larger size, defroster)
        22: 4000,    # side_mirror_crack - ₹3K-6K (glass only)
        23: 12000,   # side_mirror_broken - ₹8K-18K (full unit with motors/sensors)
        24: 6000,    # wheel_rim_scratch - ₹4K-10K (refinishing)
        25: 15000,   # wheel_rim_bent - ₹10K-22K (repair or replace)
        26: 10000,   # tire_damage - ₹5K-15K (depends on tire type)
        27: 25000,   # paint_peel - ₹15K-40K (full panel repaint)
        28: 35000,   # rust_damage - ₹20K-60K (extensive work, welding)
        29: 30000,   # panel_misalignment - ₹20K-45K (structural work)
        30: 15000    # grille_damage - ₹8K-25K (varies by car model)
    }
    df['label'] = df['classes'].map(class_mapping).fillna(0).astype(int)
    
    # Show class distribution
    print("\nClass Distribution in Dataset:")
    print(df['classes'].value_counts())
    print("\nClass mapping applied. Sample labels:")
    print(df[['classes', 'label']].head())

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Create datasets and loaders
    train_dataset = VehicleDataset(train_df, transform=transform)
    val_dataset = VehicleDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print("DataLoaders created.")

    # -------------------------------
    # 4. Model
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}, Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 8)  # 8 classes now (was 7)
    model = model.to(device)
    print("Model ready with 8 output classes.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # -------------------------------
    # 5. Training Loop
    # -------------------------------
    epochs = 15  # Increased from 5 to 15 for better accuracy
    print(f"\nStarting training for {epochs} epochs...")
    print("This will take approximately 5-10 minutes with GPU.\n")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {running_loss/len(train_loader.dataset):.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------
    # 6. Validation
    # -------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(f"Validation Batch {batch_idx+1}/{len(val_loader)}: Batch Accuracy = {(preds==labels).sum().item()/labels.size(0):.2f}")

    print(f"Validation Accuracy: {correct/total:.2f}")

    # -------------------------------
    # 7. Save the Trained Model with cost mapping
    # -------------------------------
    model_save_path = 'vehicle_damage_model.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': correct/total,
        'class_mapping': class_mapping,
        'cost_mapping': cost_mapping
    }, model_save_path)
    print(f"\n{'='*50}")
    print(f"Model saved successfully to '{model_save_path}'")
    print(f"Final validation accuracy: {correct/total:.2%}")
    print(f"{'='*50}\n")

    def predict_cost(image_path):
        print(f"Predicting for image: {image_path}")
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(image)
            else:
                outputs = model(image)
            _, pred = torch.max(outputs, 1)
            predicted_class = pred.item()
            cost = cost_mapping[predicted_class]
        print(f"Predicted Class: {predicted_class}, Estimated Repair Cost: ₹{cost}")
        return predicted_class, cost

    # Example usage
    predict_cost('datasets/image/1.jpeg')
