import shutil
import os
from pathlib import Path

AI_SOURCE = Path("AiArtdata/AiArtData")
REAL_SOURCE = Path("RealArt/RealArt")

AI_DEST = Path("dataset/ai")
REAL_DEST = Path("dataset/real")

AI_DEST.mkdir(parents=True, exist_ok=True)
REAL_DEST.mkdir(parents=True, exist_ok=True)

print("Copying AI images...")
ai_count = 0
for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
    for img in AI_SOURCE.rglob(ext):
        try:
            dest_name = f"ai_{img.stem}{img.suffix}"
            shutil.copy(img, AI_DEST / dest_name)
            ai_count += 1
        except Exception as e:
            print(f"Error: {e}")

print("Copying Real images...")
real_count = 0
for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
    for img in REAL_SOURCE.rglob(ext):
        try:
            dest_name = f"real_{img.stem}{img.suffix}"
            shutil.copy(img, REAL_DEST / dest_name)
            real_count += 1
        except Exception as e:
            print(f"Error: {e}")

print(f"\nDone!")
print(f"AI images copied: {ai_count}")
print(f"Real images copied: {real_count}")
print(f"Total: {ai_count + real_count}")
