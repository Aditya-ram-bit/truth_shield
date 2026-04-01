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

# Enhanced Beautiful UI
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truth Shield - AI Content Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }
        
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #ec4899;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --dark: #0f172a;
            --dark2: #1e293b;
            --dark3: #334155;
            --light: #f8fafc;
            --gray: #94a3b8;
        }
        
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #fff;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated background */
        body::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.15) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        /* Header */
        header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .logo {
            font-size: 4rem;
            margin-bottom: 0.5rem;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #6366f1, #ec4899, #6366f1);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .subtitle {
            color: var(--gray);
            font-size: 1.1rem;
            font-weight: 300;
        }
        
        /* Stats Bar */
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            animation: fadeIn 1s ease 0.3s both;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .stat-item {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stat-item span {
            color: var(--primary);
            font-weight: 600;
        }
        
        /* Main Card */
        .main-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeInUp 0.8s ease 0.2s both;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Type Selector */
        .type-selector {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            background: var(--dark);
            padding: 0.5rem;
            border-radius: 16px;
        }
        
        .type-btn {
            flex: 1;
            padding: 1rem;
            border: none;
            background: transparent;
            color: var(--gray);
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .type-btn:hover {
            color: #fff;
            background: rgba(99, 102, 241, 0.2);
        }
        
        .type-btn.active {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: #fff;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }
        
        .type-btn svg {
            width: 24px;
            height: 24px;
        }
        
        /* Upload Area */
        .upload-area {
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(15, 23, 42, 0.5);
            margin-bottom: 1.5rem;
        }
        
        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
            transform: scale(1.01);
        }
        
        .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.15);
        }
        
        .upload-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .upload-text {
            color: var(--gray);
            font-size: 1.1rem;
        }
        
        .upload-text strong {
            color: var(--primary);
        }
        
        .file-preview {
            display: none;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: var(--dark);
            border-radius: 16px;
            margin-bottom: 1.5rem;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .file-preview.show {
            display: flex;
        }
        
        .file-preview img {
            width: 60px;
            height: 60px;
            object-fit: cover;
            border-radius: 12px;
        }
        
        .file-info {
            flex: 1;
        }
        
        .file-name {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .file-size {
            color: var(--gray);
            font-size: 0.85rem;
        }
        
        .remove-file {
            background: rgba(239, 68, 68, 0.2);
            border: none;
            color: var(--danger);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .remove-file:hover {
            background: var(--danger);
            color: #fff;
        }
        
        /* Text Area */
        .text-area {
            display: none;
            margin-bottom: 1.5rem;
        }
        
        .text-area.show {
            display: block;
        }
        
        textarea {
            width: 100%;
            min-height: 180px;
            background: var(--dark);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.2rem;
            color: #fff;
            font-size: 1rem;
            resize: vertical;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        
        textarea::placeholder {
            color: var(--gray);
        }
        
        /* Analyze Button */
        .analyze-btn {
            width: 100%;
            padding: 1.2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            border-radius: 16px;
            color: #fff;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
        }
        
        .analyze-btn:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
        }
        
        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Result Card */
        .result-card {
            display: none;
            background: var(--dark);
            border-radius: 24px;
            padding: 2rem;
            text-align: center;
            animation: scaleIn 0.5s ease;
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .result-card.show {
            display: block;
        }
        
        .result-header {
            margin-bottom: 1.5rem;
        }
        
        .score-display {
            position: relative;
            display: inline-block;
            margin-bottom: 1rem;
        }
        
        .score-circle {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 8px solid;
            position: relative;
        }
        
        .score-circle.real {
            border-color: var(--success);
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
        }
        
        .score-circle.ai {
            border-color: var(--danger);
            box-shadow: 0 0 40px rgba(239, 68, 68, 0.3);
        }
        
        .score-circle.uncertain {
            border-color: var(--warning);
            box-shadow: 0 0 40px rgba(245, 158, 11, 0.3);
        }
        
        .score-value {
            font-size: 3.5rem;
            font-weight: 700;
        }
        
        .score-label {
            font-size: 0.9rem;
            color: var(--gray);
        }
        
        .verdict {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .verdict.real { color: var(--success); }
        .verdict.ai { color: var(--danger); }
        .verdict.uncertain { color: var(--warning); }
        
        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(99, 102, 241, 0.2);
            color: var(--primary);
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Feedback Section */
        .feedback-section {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .feedback-title {
            color: var(--gray);
            margin-bottom: 1rem;
            font-size: 1rem;
        }
        
        .feedback-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .feedback-btn {
            padding: 0.8rem 2rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .feedback-btn.yes {
            background: var(--success);
            color: #fff;
        }
        
        .feedback-btn.yes:hover {
            background: #059669;
            transform: translateY(-2px);
        }
        
        .feedback-btn.no {
            background: var(--danger);
            color: #fff;
        }
        
        .feedback-btn.no:hover {
            background: #dc2626;
            transform: translateY(-2px);
        }
        
        .feedback-thanks {
            color: var(--success);
            margin-top: 1rem;
            font-weight: 500;
            display: none;
        }
        
        .feedback-thanks.show {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 3rem;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Footer */
        footer {
            text-align: center;
            margin-top: 3rem;
            color: var(--gray);
            font-size: 0.9rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem; }
            .type-btn { padding: 0.8rem; font-size: 0.9rem; }
            .score-circle { width: 150px; height: 150px; }
            .score-value { font-size: 2.5rem; }
            .stats-bar { gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🛡️</div>
            <h1>Truth Shield</h1>
            <p class="subtitle">Advanced AI-Generated Content Detection System</p>
        </header>
        
        <div class="stats-bar">
            <div class="stat-item">
                📊 <span>96.22%</span> Accuracy
            </div>
            <div class="stat-item">
                📈 <span>6000+</span> Images Trained
            </div>
            <div class="stat-item">
                💬 <span>Feedback</span> Enabled
            </div>
        </div>
        
        <div class="main-card">
            <!-- Type Selector -->
            <div class="type-selector">
                <button class="type-btn active" onclick="setType('image')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <path d="M21 15l-5-5L5 21"/>
                    </svg>
                    Image
                </button>
                <button class="type-btn" onclick="setType('text')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M16 13H8"/>
                        <path d="M16 17H8"/>
                        <path d="M10 9H8"/>
                    </svg>
                    Text
                </button>
            </div>
            
            <!-- Image Upload -->
            <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                <div class="upload-icon">📁</div>
                <p class="upload-text">
                    <strong>Click to upload</strong> or drag and drop<br>
                    PNG, JPG, JPEG (Max 10MB)
                </p>
                <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFile(this)">
            </div>
            
            <!-- File Preview -->
            <div class="file-preview" id="file-preview">
                <img id="preview-img" src="" alt="Preview">
                <div class="file-info">
                    <div class="file-name" id="file-name">filename.jpg</div>
                    <div class="file-size" id="file-size">0 KB</div>
                </div>
                <button class="remove-file" onclick="removeFile(event)">✕</button>
            </div>
            
            <!-- Text Input -->
            <div class="text-area" id="text-area">
                <textarea id="text-input" placeholder="Paste or type your text here for AI detection analysis..." oninput="checkInput()"></textarea>
            </div>
            
            <!-- Analyze Button -->
            <button class="analyze-btn" id="analyze-btn" onclick="analyze()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="M21 21l-4.35-4.35"/>
                </svg>
                Analyze Content
            </button>
            
            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing content with AI model...</p>
            </div>
            
            <!-- Result Card -->
            <div class="result-card" id="result-card">
                <div class="result-header">
                    <div class="score-display">
                        <div class="score-circle" id="score-circle">
                            <span class="score-value" id="score-value">0.0</span>
                            <span class="score-label">/ 10</span>
                        </div>
                    </div>
                    <div class="verdict" id="verdict">Likely Real</div>
                    <div class="confidence-badge" id="confidence-badge">High Confidence</div>
                </div>
                
                <!-- Feedback -->
                <div class="feedback-section">
                    <p class="feedback-title">Was this prediction correct?</p>
                    <div class="feedback-buttons">
                        <button class="feedback-btn yes" onclick="submitFeedback(true)">
                            ✓ Yes, Correct
                        </button>
                        <button class="feedback-btn no" onclick="submitFeedback(false)">
                            ✗ No, Wrong
                        </button>
                    </div>
                    <p class="feedback-thanks" id="feedback-thanks">Thanks for your feedback!</p>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Truth Shield v2.0 | Powered by Machine Learning</p>
        </footer>
    </div>
    
    <script>
        let currentType = 'image';
        let selectedFile = null;
        let lastResult = null;
        
        function setType(type) {
            currentType = type;
            document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
            event.target.closest('.type-btn').classList.add('active');
            
            if (type === 'text') {
                document.getElementById('upload-area').style.display = 'none';
                document.getElementById('file-preview').classList.remove('show');
                document.getElementById('text-area').classList.add('show');
            } else {
                document.getElementById('upload-area').style.display = 'block';
                document.getElementById('text-area').classList.remove('show');
            }
            checkInput();
        }
        
        function handleFile(input) {
            if (input.files && input.files[0]) {
                selectedFile = input.files[0];
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview-img').src = e.target.result;
                };
                reader.readAsDataURL(selectedFile);
                
                document.getElementById('file-name').textContent = selectedFile.name;
                document.getElementById('file-size').textContent = formatSize(selectedFile.size);
                document.getElementById('file-preview').classList.add('show');
                
                checkInput();
            }
        }
        
        function removeFile(e) {
            e.stopPropagation();
            selectedFile = null;
            document.getElementById('file-input').value = '';
            document.getElementById('file-preview').classList.remove('show');
            checkInput();
        }
        
        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
            return (bytes/(1024*1024)).toFixed(1) + ' MB';
        }
        
        function checkInput() {
            const btn = document.getElementById('analyze-btn');
            if (currentType === 'text') {
                btn.disabled = document.getElementById('text-input').value.length < 10;
            } else {
                btn.disabled = !selectedFile;
            }
        }
        
        async function analyze() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('result-card').classList.remove('show');
            document.getElementById('feedback-thanks').classList.remove('show');
            
            const formData = new FormData();
            formData.append('content_type', currentType);
            
            if (currentType === 'text') {
                formData.append('text', document.getElementById('text-input').value);
            } else {
                formData.append('file', selectedFile);
            }
            
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                
                lastResult = data;
                
                // Update result display
                const score = data.score;
                const scoreCircle = document.getElementById('score-circle');
                const verdict = document.getElementById('verdict');
                
                document.getElementById('score-value').textContent = score.toFixed(1);
                document.getElementById('verdict').textContent = data.verdict;
                document.getElementById('confidence-badge').textContent = (data.confidence || 'Medium') + ' Confidence';
                
                // Set colors based on score
                scoreCircle.classList.remove('real', 'ai', 'uncertain');
                verdict.classList.remove('real', 'ai', 'uncertain');
                
                if (score < 4) {
                    scoreCircle.classList.add('real');
                    verdict.classList.add('real');
                } else if (score > 6) {
                    scoreCircle.classList.add('ai');
                    verdict.classList.add('ai');
                } else {
                    scoreCircle.classList.add('uncertain');
                    verdict.classList.add('uncertain');
                }
                
                document.getElementById('result-card').classList.add('show');
                
            } catch (e) {
                alert('Error: ' + e.message);
            }
            
            document.getElementById('loading').classList.remove('show');
        }
        
        async function submitFeedback(isCorrect) {
            if (!lastResult) return;
            
            try {
                const formData = new FormData();
                formData.append('is_correct', isCorrect ? 'true' : 'false');
                formData.append('content_type', currentType);
                formData.append('score', lastResult.score);
                formData.append('verdict', lastResult.verdict);
                
                await fetch('/feedback', { method: 'POST', body: formData });
                
                document.getElementById('feedback-thanks').textContent = 
                    isCorrect ? "Thanks! Your feedback helps us improve!" : "Thanks for the correction! We'll learn from this.";
                document.getElementById('feedback-thanks').classList.add('show');
                
            } catch (e) {
                console.error('Error:', e);
            }
        }
        
        // Drag and drop
        const uploadArea = document.getElementById('upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files[0]) {
                handleFile({ files: [e.dataTransfer.files[0]] });
            }
        });
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
        if content_type == "image":
            if not file:
                raise HTTPException(status_code=400, detail="File required")
            
            file_ext = file.filename.split(".")[-1].lower() if file.filename else "jpg"
            filename = f"{uuid.uuid4()}.{file_ext}"
            filepath = UPLOAD_DIR / filename
            
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            result = await content_detector.analyze_image(filepath)
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
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_data = json.load(f)
        
        correct = is_correct.lower() == 'true'
        
        feedback_entry = {
            "content_type": content_type,
            "predicted_score": score,
            "predicted_verdict": verdict,
            "was_correct": correct
        }
        
        if correct:
            feedback_data["correct"].append(feedback_entry)
            message = "Thanks! We'll use this to improve."
        else:
            feedback_data["incorrect"].append(feedback_entry)
            message = "Thanks for the correction! We'll learn from this."
        
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
