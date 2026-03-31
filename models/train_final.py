# train_final.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_RATIO = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# Dataset
class AIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Real images (label = 0)
        for folder in ["real"]:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in (self.data_dir / folder).glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(0)
        
        # AI images (label = 1) - all AI folders
        for folder in ["ai", "gemini"]:
            if (self.data_dir / folder).exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    for img_path in (self.data_dir / folder).glob(ext):
                        self.images.append(str(img_path))
                        self.labels.append(1)
        
        print(f"Total images: {len(self.images)}")
        print(f"Real: {self.labels.count(0)} | AI: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
        except:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Model
class AIImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading dataset...")
dataset = AIDataset("dataset", transform=train_transform)

# Split: 70% train, 30% test
train_size = int(len(dataset) * TRAIN_RATIO)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# Apply test transform to test set only
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Training: {len(train_dataset)} images")
print(f"Testing: {len(test_dataset)} images")

# Create model
model = AIImageDetector().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train
print("\nTraining...")
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch}/{EPOCHS} - Train Accuracy: {train_acc:.2f}%")
    
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), "trained_model/best_model.pth")
    
    scheduler.step()

# Load best model and test
print("\nTesting on 30% unseen data...")
model.load_state_dict(torch.load("trained_model/best_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

# Results
print("\n" + "="*60)
print("FINAL RESULTS (30% Test Data)")
print("="*60)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"  True Real Correct: {cm[0][0]}")
print(f"  False AI (Wrong): {cm[0][1]}")
print(f"  False Real (Wrong): {cm[1][0]}")
print(f"  True AI Correct: {cm[1][1]}")

# Save final model
torch.save(model.state_dict(), "trained_model/ai_detector.pth")
print("\nModel saved to trained_model/ai_detector.pth")
