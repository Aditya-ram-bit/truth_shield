# models/text_detector.py
import re
import numpy as np
from typing import Dict, Any, List
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class TextDetector:
    """
    AI-generated text detection using ML model + heuristics
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
        # Try to load ML model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model for AI text detection"""
        try:
            print("Loading AI text detection model...")
            # Using a lightweight model for text classification
            model_name = "roberta-base-openai-detector"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Could not load ML model: {e}")
            print("Using heuristic-based detection instead")
            self.model_loaded = False
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Full text analysis with ML + heuristics"""
        
        words = text.split()
        if len(words) < 10:
            return {
                "score": 5.0,
                "verdict": "Text too short to analyze",
                "confidence": "Low",
                "analysis": {"word_count": len(words)},
                "indicators": ["Text must be at least 10 words"],
                "content_type": "text"
            }
        
        # Get ML-based score
        ml_score = self._get_ml_score(text)
        
        # Get heuristic scores
        heuristic_score = self._get_heuristic_score(text)
        
        # Combine scores
        if self.model_loaded:
            # Weight ML model higher (70% ML, 30% heuristics)
            final_score = (ml_score * 0.7) + (heuristic_score * 0.3)
            source = "ML Model + Heuristics"
        else:
            final_score = heuristic_score
            source = "Heuristics Only"
        
        # Clamp to 0-10
        final_score = max(0, min(10, final_score))
        
        return {
            "score": round(final_score, 1),
            "verdict": self._get_verdict(final_score),
            "confidence": "High" if self.model_loaded else "Medium",
            "analysis": {
                "word_count": len(words),
                "sentence_count": len(re.split(r'[.!?]+', text)),
                "detection_method": source,
                "ml_score": round(ml_score, 1) if self.model_loaded else None,
                "heuristic_score": round(heuristic_score, 1)
            },
            "indicators": self._get_indicators(text, ml_score, heuristic_score),
            "content_type": "text"
        }
    
    def _get_ml_score(self, text: str) -> float:
        """Get AI probability from ML model"""
        
        if not self.model_loaded or not self.tokenizer:
            return 5.0
        
        try:
            # Truncate text if too long (model has max length)
            text = text[:512]
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Model outputs: [0] = real, [1] = AI
            # Convert to 0-10 scale (higher = more likely AI)
            ai_prob = probs[0][1].item()
            score = ai_prob * 10
            
            return score
            
        except Exception as e:
            print(f"ML detection error: {e}")
            return 5.0
    
    def _get_heuristic_score(self, text: str) -> float:
        """Fallback heuristic-based detection"""
        
        text_lower = text.lower()
        words = text_lower.split()
        
        score = 0
        indicators = 0
        
        # AI-typical patterns
        ai_patterns = [
            'it is worth noting', 'importance of', 'additionally', 'furthermore',
            'moreover', 'nevertheless', 'however', 'ultimately', 'in conclusion',
            'it can be observed', 'it is clear that', 'variety of', 'wide range'
        ]
        
        for pattern in ai_patterns:
            if pattern in text_lower:
                score += 0.5
                indicators += 1
        
        # Sentence length uniformity
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 2:
            sent_lens = [len(s.split()) for s in sentences]
            std = np.std(sent_lens)
            if std < 3:  # Too uniform
                score += 2
        
        # Vocabulary sophistication
        long_words = len([w for w in words if len(w) > 6])
        if long_words / len(words) > 0.3:
            score += 1.5
        
        # Normalize to 0-10
        score = min(10, score)
        
        return score
    
    def _get_verdict(self, score: float) -> str:
        """Human-readable verdict"""
        
        if score < 3:
            return "Likely Human Written"
        elif score < 5:
            return "Possibly Human Written"
        elif score < 7:
            return "Likely AI Generated"
        else:
            return "Almost Certainly AI Generated"
    
    def _get_indicators(self, text: str, ml_score: float, heuristic_score: float) -> List[str]:
        """Generate detection indicators"""
        
        indicators = []
        
        if self.model_loaded:
            if ml_score > 6:
                indicators.append(f"ML Model detected AI patterns (score: {ml_score:.1f}/10)")
            elif ml_score < 4:
                indicators.append(f"ML Model indicates human-written (score: {ml_score:.1f}/10)")
            else:
                indicators.append(f"ML Model uncertain (score: {ml_score:.1f}/10)")
        
        # Add heuristic indicators
        text_lower = text.lower()
        if 'additionally' in text_lower or 'furthermore' in text_lower:
            indicators.append("Contains formal connector words (AI-typical)")
        
        if heuristic_score > 6:
            indicators.append("Multiple heuristic AI markers detected")
        
        return indicators
