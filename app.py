# app.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import json

from models.detector import ContentDetector
from models.text_detector import TextDetector

# Initialize app
app = FastAPI(title="Truth Shield")

# Setup paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Feedback storage
FEEDBACK_FILE = BASE_DIR / "feedback.json"
if not FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump({"correct": [], "incorrect": []}, f)

# Initialize detectors
content_detector = ContentDetector()
text_detector = TextDetector()

# HTML with Feedback System
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truth Shield - AI Content Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { min-height: 100vh; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        h1 { font-size: 3rem; margin-bottom: 0.5rem; }
        .subtitle { color: #888; margin-bottom: 2rem; }
        .buttons { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-bottom: 2rem; }
        .btn { padding: 1rem 2rem; border: 2px solid #00d4ff; background: transparent; color: #00d4ff; border-radius: 10px; cursor: pointer; font-size: 1.1rem; transition: 0.3s; }
        .btn:hover, .btn.active { background: #00d4ff; color: #1a1a2e; }
        .input-area { background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 15px; margin-bottom: 1rem; }
        textarea { width: 100%; min-height: 150px; background: rgba(0,0,0,0.3); border: 1px solid #333; border-radius: 10px; padding: 1rem; color: #fff; font-size: 1rem; }
        .upload-box { border: 2px dashed #444; padding: 3rem; border-radius: 10px; cursor: pointer; }
        .upload-box:hover { border-color: #00d4ff; }
        .analyze-btn { width: 100%; padding: 1.2rem; background: linear-gradient(90deg, #00d4ff, #7c3aed); border: none; border-radius: 10px; color: #fff; font-size: 1.2rem; cursor: pointer; margin-top: 1rem; }
        .analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .result { display: none; margin-top: 2rem; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px; }
        .result.show { display: block; }
        
        .score { font-size: 4rem; font-weight: bold; }
        .real { color: #22c55e; }
        .ai { color: #ef4444; }
        .mixed { color: #f59e0b; }
        .verdict { font-size: 1.5rem; margin: 1rem 0; }
        
        /* Feedback Section */
        .feedback-section { display: none; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #333; }
        .feedback-section.show { display: block; }
        .feedback-question { color: #aaa; margin-bottom: 1rem; font-size: 1.1rem; }
        .feedback-buttons { display: flex; gap: 1rem; justify-content: center; }
        .feedback-btn { padding: 0.8rem 2rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; transition: 0.3s; }
        .feedback-btn.yes { background: #22c55e; color: white; }
        .feedback-btn.yes:hover { background: #16a34a; }
        .feedback-btn.no { background: #ef4444; color: white; }
        .feedback-btn.no:hover { background: #dc2626; }
        
        .feedback-thanks { color: #22c55e; margin-top: 1rem; display: none; font-size: 1.1rem; }
        .feedback-thanks.show { display: block; }
        
        .stats { background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; color: #888; }
        
        .loading { display: none; }
        .loading.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Truth Shield</h1>
        <p class="subtitle">AI Content Detection with Feedback Learning</p>
        
        <div class="buttons">
            <button class="btn active" onclick="setType('image')">Image</button>
            <button class="btn" onclick="setType('video')">Video</button>
            <button class="btn" onclick="setType('text')">Text</button>
        </div>
        
        <div class="input-area" id="file-input-area">
            <div class="upload-box" onclick="document.getElementById('file').click()">
                <p>Click to upload image or video</p>
                <p style="color:#666;font-size:0.9rem;margin-top:0.5rem">PNG, JPG, MP4</p>
            </div>
            <input type="file" id="file" style="display:none" onchange="handleFile(this)">
            <p id="filename" style="margin-top:1rem;color:#00d4ff"></p>
        </div>
        
        <div class="input-area" id="text-input-area" style="display:none">
            <textarea id="text" placeholder="Paste text here..." oninput="checkInput()"></textarea>
        </div>
        
        <button class="analyze-btn" id="analyze-btn" onclick="analyze()">Analyze Content</button>
        
        <div class="loading" id="loading">Analyzing...</div>
        
        <div class="result" id="result">
            <div class="score" id="score">0.0</div>
            <div class="verdict" id="verdict">Likely Real</div>
            <p id="confidence">Confidence: High</p>
            
            <!-- Feedback Section -->
            <div class="feedback-section" id="feedback-section">
                <p class="feedback-question">Is the prediction correct?</p>
                <div class="feedback-buttons">
                    <button class="feedback-btn yes" id="btn-yes">Yes, Correct!</button>
                    <button class="feedback-btn no" id="btn-no">No, It's Wrong</button>
                </div>
                <p class="feedback-thanks" id="feedback-thanks">Thanks for your feedback!</p>
            </div>
            
            <div class="stats">
                Model trained on 2,224 images<br>
                Test Accuracy: 96.22%
            </div>
        </div>
    </div>
    
    <script>
        let currentType = 'image';
        let selectedFile = null;
        let lastResult = null;
        
        function setType(type) {
            currentType = type;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('file-input-area').style.display = type === 'text' ? 'none' : 'block';
            document.getElementById('text-input-area').style.display = type === 'text' ? 'block' : 'none';
            checkInput();
        }
        
        function handleFile(input) {
            if (input.files[0]) {
                selectedFile = input.files[0];
                document.getElementById('filename').textContent = selectedFile.name;
                checkInput();
            }
        }
        
        function checkInput() {
            const btn = document.getElementById('analyze-btn');
            if (currentType === 'text') {
                btn.disabled = document.getElementById('text').value.length < 10;
            } else {
                btn.disabled = !selectedFile;
            }
        }
        
        async function analyze() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            document.getElementById('feedback-section').classList.remove('show');
            document.getElementById('feedback-thanks').classList.remove('show');
            
            const formData = new FormData();
            formData.append('content_type', currentType);
            
            if (currentType === 'text') {
                formData.append('text', document.getElementById('text').value);
            } else {
                formData.append('file', selectedFile);
            }
            
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                
                lastResult = data;
                
                document.getElementById('score').textContent = data.score.toFixed(1);
                document.getElementById('score').className = 'score ' + (data.score < 4 ? 'real' : data.score > 6 ? 'ai' : 'mixed');
                document.getElementById('verdict').textContent = data.verdict;
                document.getElementById('confidence').textContent = 'Confidence: ' + (data.confidence || 'Medium');
                
                document.getElementById('result').classList.add('show');
                document.getElementById('feedback-section').classList.add('show');
                
            } catch (e) {
                alert('Error: ' + e.message);
            }
            
            document.getElementById('loading').classList.remove('show');
        }
        
        // Feedback button handlers
        document.getElementById('btn-yes').addEventListener('click', function() {
            submitFeedback(true);
        });
        
        document.getElementById('btn-no').addEventListener('click', function() {
            submitFeedback(false);
        });
        
        async function submitFeedback(isCorrect) {
            if (!lastResult) return;
            
            try {
                const formData = new FormData();
                formData.append('is_correct', isCorrect ? 'true' : 'false');
                formData.append('content_type', currentType);
                formData.append('score', lastResult.score);
                formData.append('verdict', lastResult.verdict);
                
                const response = await fetch('/feedback', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                document.getElementById('feedback-thanks').textContent = data.message || "Thanks for your feedback!";
                document.getElementById('feedback-thanks').classList.add('show');
                
            } catch (e) {
                console.error('Feedback error:', e);
                alert('Error submitting feedback');
            }
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

@app.post("/analyze")
async def analyze_content(
    content_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    try:
        if content_type in ["image", "video"]:
            if not file:
                raise HTTPException(status_code=400, detail="File required")
            
            file_ext = file.filename.split(".")[-1].lower() if file.filename else "jpg"
            filename = f"{uuid.uuid4()}.{file_ext}"
            filepath = UPLOAD_DIR / filename
            
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            if content_type == "image":
                result = await content_detector.analyze_image(filepath)
            else:
                result = await content_detector.analyze_video(filepath)
            
            filepath.unlink(missing_ok=True)
            return JSONResponse(content=result)
            
        elif content_type == "text":
            if not text or len(text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Text too short")
            
            result = text_detector.analyze(text)
            return JSONResponse(content=result)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid type")
            
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/feedback")
async def submit_feedback(
    is_correct: str = Form(...),
    content_type: str = Form(...),
    score: float = Form(...),
    verdict: str = Form(...)
):
    """Handle user feedback"""
    
    try:
        # Load existing feedback
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_data = json.load(f)
        
        # Parse is_correct
        correct = is_correct.lower() == 'true'
        
        # Add new feedback
        feedback_entry = {
            "content_type": content_type,
            "predicted_score": score,
            "predicted_verdict": verdict,
            "was_correct": correct
        }
        
        if correct:
            feedback_data["correct"].append(feedback_entry)
            message = "Thanks! We'll use this to improve the model."
        else:
            feedback_data["incorrect"].append(feedback_entry)
            message = "Thanks for the correction! We'll learn from this."
        
        # Save feedback
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return JSONResponse(content={
            "message": message,
            "total_correct": len(feedback_data["correct"]),
            "total_incorrect": len(feedback_data["incorrect"])
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "feedback_enabled": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
