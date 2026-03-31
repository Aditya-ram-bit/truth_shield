# app.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional

from models.detector import ContentDetector
from models.text_detector import TextDetector

# Initialize app
app = FastAPI(title="Truth Shield")

# Setup paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize detectors
content_detector = ContentDetector()
text_detector = TextDetector()

# Simple HTML (no emojis to avoid encoding issues)
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
        .loading { display: none; }
        .loading.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Truth Shield</h1>
        <p class="subtitle">Detect AI-Generated Content</p>
        
        <div class="buttons">
            <button class="btn active" onclick="setType('image')">Image</button>
            <button class="btn" onclick="setType('video')">Video</button>
            <button class="btn" onclick="setType('text')">Text</button>
        </div>
        
        <div class="input-area" id="file-input-area">
            <div class="upload-box" onclick="document.getElementById('file').click()">
                <p>Click to upload image or video</p>
                <p style="color:#666;font-size:0.9rem;margin-top:0.5rem">PNG, JPG, MP4, AVI, MOV</p>
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
        </div>
    </div>
    
    <script>
        let currentType = 'image';
        let selectedFile = null;
        
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
                
                document.getElementById('score').textContent = data.score.toFixed(1);
                document.getElementById('score').className = 'score ' + (data.score < 4 ? 'real' : data.score > 6 ? 'ai' : 'mixed');
                document.getElementById('verdict').textContent = data.verdict;
                document.getElementById('confidence').textContent = 'Confidence: ' + (data.confidence || 'Medium');
                document.getElementById('result').classList.add('show');
            } catch (e) {
                alert('Error: ' + e.message);
            }
            
            document.getElementById('loading').classList.remove('show');
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    return HTML_CONTENT

@app.post("/analyze")
async def analyze_content(
    content_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Analyze content and return AI detection score"""
    
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
