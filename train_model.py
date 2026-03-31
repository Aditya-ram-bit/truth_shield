import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

class Config:
    DATA_DIR = "dataset"
    OUTPUT_DIR = "trained_model"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in real_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(0)
        
        ai_dir = self.data_dir / "ai"
        if ai_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in ai_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.images)} images | Real: {self.labels.count(0)} | AI: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label

class AIImageDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(AIImageDetector, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
        "f1_score": f1_score(all_labels, all_preds, average='weighted'),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    return metrics

def main():
    print("="*60)
    print("Truth Shield - Model Training")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    train_transform, val_transform = get_transforms()
    
    print("Loading dataset...")
    full_dataset = AIDataset(Config.DATA_DIR, transform=train_transform)
    
    total_size = len(full_dataset)
    train_size = int(total_size * Config.TRAIN_RATIO)
    val_size = int(total_size * Config.VAL_RATIO)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    print("Creating model...")
    model = AIImageDetector(num_classes=2)
    model = model.to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "best_model.pth"))
        scheduler.step()
    
    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, "best_model.pth")))
    metrics = evaluate_model(model, test_loader)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score: {metrics['f1_score']*100:.2f}%")
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    with open(os.path.join(Config.OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "ai_detector.pth"))
    print("\nModel saved to trained_model/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=Config.DATA_DIR)
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    args = parser.parse_args()
    Config.DATA_DIR = args.dataset_path
    Config.NUM_EPOCHS = args.epochs
    main()
