from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import torch
import time
import base64
import io
from PIL import Image
import mediapipe as mp
from build_model.model import DermatologyClassifier
from torchvision import transforms
from face_scanner import FaceScanner
import logging
import os
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SkinSight API",
    description="Real-time skin condition analysis using computer vision and deep learning",
    version="1.0.0"
)

# Global scanner instance
scanner: FaceScanner | None = None
device = None

# Pydantic models for API requests/responses
class ScanRequest(BaseModel):
    image_data: str  # Base64 encoded image
    scan_duration: Optional[float] = 3.0  # Duration in seconds

class BatchScanRequest(BaseModel):
    image_data_list: List[str]  # List of base64 encoded images
    scan_duration: Optional[float] = 3.0  # Duration in seconds

class ScanResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# Initialize scanner on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the face scanner and device on application startup"""
    global scanner, device
    
    try:
        # Instantiate a single FaceScanner (handles device, model, transforms)
        model_path = os.environ.get(
            "MODEL_PATH",
            "prod_model/best_model_twohead_0922_031215.pth"
        )
        scanner = FaceScanner(model_path=model_path)
        device = scanner.device
        logger.info("FaceScanner initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise e

# Utility function
def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def analyze_skin_condition(face_crops: List[np.ndarray]) -> Dict[str, Any]:
    """Delegate analysis to FaceScanner for consistency"""
    if scanner is None:
        raise HTTPException(status_code=500, detail="Scanner not initialized")
    try:
        return scanner.analyze_skin_condition(face_crops)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Skin analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        with open("web_interface.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>SkinSight API</h1>
        <p>API is running! Web interface not found.</p>
        <p>Available endpoints:</p>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
            <li>POST /analyze - Single image analysis</li>
            <li>POST /analyze-batch - Batch image analysis</li>
        </ul>
        </body></html>
        """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if scanner is not None else "unhealthy",
        model_loaded=scanner is not None,
        device=str(device) if device else "unknown"
    )

@app.post("/analyze", response_model=ScanResponse)
async def analyze_skin(request: ScanRequest):
    """Analyze skin condition from uploaded image"""
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        
        # Detect and crop face using FaceScanner helper for consistency
        face_crop = scanner.detect_and_crop_face(image) if scanner else None
        
        if face_crop is None:
            return ScanResponse(
                success=False,
                message="No face detected or multiple faces detected. Please ensure exactly one face is visible.",
                processing_time=time.time() - start_time
            )
        
        # Analyze skin condition
        results = analyze_skin_condition([face_crop])
        
        processing_time = time.time() - start_time
        
        return ScanResponse(
            success=True,
            message="Analysis completed successfully",
            results=results,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_skin: {e}")
        return ScanResponse(
            success=False,
            message=f"Unexpected error: {e}",
            processing_time=time.time() - start_time
        )

@app.post("/analyze-batch", response_model=ScanResponse)
async def analyze_skin_batch(request: BatchScanRequest):
    """Analyze skin condition from multiple images (proper batching)"""
    start_time = time.time()
    
    try:
        if not request.image_data_list:
            return ScanResponse(
                success=False,
                message="No images provided for batch analysis",
                processing_time=time.time() - start_time
            )
        
        # Decode all images and detect faces
        face_crops = []
        for i, image_data in enumerate(request.image_data_list):
            try:
                image = decode_base64_image(image_data)
                face_crop = scanner.detect_and_crop_face(image) if scanner else None
                
                if face_crop is not None:
                    face_crops.append(face_crop)
                else:
                    logger.warning(f"No face detected in image {i+1}")
                    
            except Exception as e:
                logger.warning(f"Error processing image {i+1}: {e}")
                continue
        
        if not face_crops:
            return ScanResponse(
                success=False,
                message="No valid faces detected in any of the provided images",
                processing_time=time.time() - start_time
            )
        
        # Analyze skin condition with proper batching via FaceScanner
        results = analyze_skin_condition(face_crops)
        
        processing_time = time.time() - start_time
        
        return ScanResponse(
            success=True,
            message=f"Batch analysis completed successfully on {len(face_crops)} face crops",
            results=results,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_skin_batch: {e}")
        return ScanResponse(
            success=False,
            message=f"Batch analysis error: {e}",
            processing_time=time.time() - start_time
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)