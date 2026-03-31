import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Create model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

# Load saved checkpoint properly
checkpoint = torch.load("trained_model/ai_detector.pth", map_location='cpu')

# Check if it's a checkpoint dict or just state_dict
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Try loading directly - handle "model." prefix
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import sys
if len(sys.argv) < 2:
    print("Usage: python test_model.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
image = Image.open(img_path).convert('RGB')
img_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)

print(f"Real: {probs[0][0].item():.2%} | AI: {probs[0][1].item():.2%}")
print("Prediction:", "AI Generated" if probs[0][1] > probs[0][0] else "Real Image")
