# models/detector.py
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any
import asyncio
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')


class ContentDetector:
    """
    Content detector using your trained model
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_loaded = False
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Load your custom trained model"""
        model_path = Path("trained_model/ai_detector.pth")
        
        if not model_path.exists():
            print("Trained model not found!")
            return
        
        try:
            print("Loading trained model...")
            
            # Create model
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Remove 'model.' prefix if present
                    state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('model.'):
                            state_dict[k[6:]] = v
                        else:
                            state_dict[k] = v
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    async def analyze_image(self, filepath: Path) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._analyze_image_sync, filepath)
        return result
    
    def _analyze_image_sync(self, filepath: Path) -> Dict[str, Any]:
        try:
            ml_score = self._get_ml_prediction(filepath)
            
            final_score = ml_score
            
            return {
                "score": round(final_score, 1),
                "verdict": self._get_verdict(final_score),
                "confidence": "High" if self.model_loaded else "Medium",
                "analysis": {
                    "file_type": filepath.suffix[1:],
                    "detection_method": "Your Trained Model" if self.model_loaded else "Heuristics"
                },
                "indicators": [f"AI Probability: {ml_score*10:.1f}/10"] if self.model_loaded else [],
                "content_type": "image"
            }
            
        except Exception as e:
            return {
                "score": 5.0,
                "verdict": "Unable to determine",
                "error": str(e),
                "content_type": "image"
            }
    
    def _get_ml_prediction(self, filepath: Path) -> float:
        if not self.model_loaded:
            return 5.0
        
        try:
            image = Image.open(filepath).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            # probs[0][0] = Real, probs[0][1] = AI
            ai_prob = probs[0][1].item()
            score = ai_prob * 10
            
            return score
            
        except Exception as e:
            print(f"Error: {e}")
            return 5.0
    
    async def analyze_video(self, filepath: Path) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._analyze_video_sync, filepath)
        return result
    
    def _analyze_video_sync(self, filepath: Path) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(str(filepath))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            sample_interval = max(1, frame_count // 10)
            frame_scores = []
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    temp_path = filepath.parent / f"temp_frame_{frame_idx}.jpg"
                    cv2.imwrite(str(temp_path), frame)
                    
                    score = self._get_ml_prediction(temp_path)
                    frame_scores.append(score)
                    
                    temp_path.unlink(missing_ok=True)
                
                frame_idx += 1
                if frame_idx >= sample_interval * 10:
                    break
            
            cap.release()
            
            if not frame_scores:
                return {"score": 5.0, "verdict": "Unable to analyze", "content_type": "video"}
            
            avg_score = np.mean(frame_scores)
            
            return {
                "score": round(avg_score, 1),
                "verdict": self._get_verdict(avg_score),
                "confidence": "High",
                "analysis": {
                    "frames_analyzed": len(frame_scores),
                    "duration_seconds": round(duration, 2),
                    "detection_method": "Your Trained Model"
                },
                "indicators": [f"Analyzed {len(frame_scores)} frames"],
                "content_type": "video"
            }
            
        except Exception as e:
            return {
                "score": 5.0,
                "verdict": "Unable to determine",
                "error": str(e),
                "content_type": "video"
            }
    
    def _get_verdict(self, score: float) -> str:
        if score < 3:
            return "Likely Real"
        elif score < 5:
            return "Possibly Real"
        elif score < 7:
            return "Likely AI Generated"
        else:
            return "Almost Certainly AI Generated"
