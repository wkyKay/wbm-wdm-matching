import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import numpy as np
import os
from PIL import Image
import pandas as pd

# Step 1: Custom Dataset Class for WM-811K
class WaferMapDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_csv)  # CSV with file names and defect labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels.iloc[idx, 1])  # Defect class label

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Step 2: Data Preparation
# Directory where images are stored and label file
DATA_DIR = "/path/to/wafer_images"  # Replace with your path
LABELS_CSV = "/path/to/labels.csv"  # Replace with your path

# Image transformations for Vision Transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
train_dataset = WaferMapDataset(DATA_DIR, LABELS_CSV, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Step 3: Load Pretrained Vision Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=6).to(device)  # 6 classes for WM-811K

# Step 4: Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2
)

# Step 5: HuggingFace Trainer Setup
def collate_fn(batch):
    images, labels = zip(*batch)
    inputs = feature_extractor(images=list(images), return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)

# Step 6: Train the Model
trainer.train()

# Step 7: Evaluation Example
model.eval()
test_image_path = "/path/to/single_test_image.png"  # Replace with a test image
test_image = Image.open(test_image_path).convert("RGB")
test_input = feature_extractor(images=test_image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**test_input)
    predicted_class = torch.argmax(output.logits, dim=1).item()
    print(f"Predicted Defect Class: {predicted_class}")
