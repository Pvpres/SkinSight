"""
Face scanning functionality for SkinSight
This module provides functions to scan faces and analyze skin conditions
"""
import cv2
import numpy as np
import torch
import time
import mediapipe as mp
from build_model.model import DermatologyClassifier
from torchvision import transforms
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceScanner:
    """Face scanning and skin analysis class"""
    
    def __init__(self, model_path: str = "build_model/models/best_model_twohead_0922_031215.pth"):
        """Initialize the face scanner with model"""
        self.model_path = model_path
        self.model = None
        self.device = None
        self.transform = None
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and device"""
        try:
            # Setup device - prioritize GPU for cloud deployment
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA for inference (GPU: {torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) for inference")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
            
            # Load model
            self.model = DermatologyClassifier(num_classes=5, use_two_heads=True)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225))
            ])
            
            logger.info("Face scanner model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize face scanner model: {e}")
            raise e
    
    def detect_and_crop_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face in image and return cropped face region"""
        try:
            with self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            ) as face_detection:
                
                # Convert BGR to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)
                
                if results.detections and len(results.detections) == 1:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    
                    h, w, c = image.shape
                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Ensure coordinates are within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    width = min(width, w - x_min)
                    height = min(height, h - y_min)
                    
                    # Crop face
                    face = image[y_min:y_min+height, x_min:x_min+width]
                    
                    if face.size > 0:
                        return face
                
                return None
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None
    
    def analyze_skin_condition(self, face_crops: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze skin condition from face crops"""
        if not face_crops:
            raise ValueError("No valid face crops to analyze")
        
        try:
            # Preprocess all face crops
            processed_faces = []
            for face in face_crops:
                if face.size != 0:
                    try:
                        face_tensor = self.transform(face)
                        processed_faces.append(face_tensor)
                    except Exception as e:
                        logger.warning(f"Error processing face crop: {e}")
                        continue
            
            if not processed_faces:
                raise ValueError("No valid face crops after preprocessing")
            
            # Batch all processed faces
            batch_tensor = torch.stack(processed_faces).to(self.device)
            
            with torch.no_grad():
                # Forward pass through model
                binary_output, disease_output = self.model(batch_tensor)
                
                # Get probabilities
                binary_probs = torch.softmax(binary_output, dim=1)
                disease_probs = torch.softmax(disease_output, dim=1)
                
                # Average predictions across all frames for more stable results
                avg_binary_probs = binary_probs.mean(dim=0)
                avg_disease_probs = disease_probs.mean(dim=0)
                
                # Extract results
                healthy_prob = avg_binary_probs[0].item()
                disease_prob = avg_binary_probs[1].item()
                
                # Disease-specific probabilities
                disease_class_labels = ["acne", "dry", "eczema", "oily"]
                disease_results = {}
                for i, label in enumerate(disease_class_labels):
                    disease_results[label] = {
                        "probability": avg_disease_probs[i].item(),
                        "percentage": round(avg_disease_probs[i].item() * 100, 1)
                    }
                
                # Determine most likely condition
                if disease_prob > healthy_prob:
                    pred_idx = avg_disease_probs.argmax().item()
                    pred_condition = disease_class_labels[pred_idx]
                    confidence = avg_disease_probs[pred_idx].item()
                    is_healthy = False
                else:
                    pred_condition = "healthy"
                    confidence = healthy_prob
                    is_healthy = True
                
                return {
                    "overall_assessment": {
                        "is_healthy": is_healthy,
                        "healthy_probability": round(healthy_prob, 3),
                        "healthy_percentage": round(healthy_prob * 100, 1),
                        "disease_probability": round(disease_prob, 3),
                        "disease_percentage": round(disease_prob * 100, 1)
                    },
                    "condition_analysis": disease_results,
                    "prediction": {
                        "condition": pred_condition,
                        "confidence": round(confidence, 3),
                        "confidence_percentage": round(confidence * 100, 1)
                    },
                    "analysis_metadata": {
                        "frames_analyzed": len(processed_faces),
                        "model_version": "two_head_v1",
                        "device_used": str(self.device)
                    }
                }
                
        except Exception as e:
            logger.error(f"Skin analysis error: {e}")
            raise e
    
    def scan_face_from_camera(self, duration: float = 3.0) -> Dict[str, Any]:
        """Scan face from camera for specified duration and return analysis"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video stream")
        
        predictions = []
        start_time = time.time()
        scanning = False
        
        try:
            with self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            ) as face_detection:
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        continue
                    
                    # Get frame dimensions
                    h, w, c = frame.shape
                    
                    # Convert to RGB for mediapipe
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                    
                    if results.detections and len(results.detections) == 1:
                        detection = results.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        
                        x_min = int(bboxC.xmin * w)
                        y_min = int(bboxC.ymin * h)
                        width = int(bboxC.width * w)
                        height = int(bboxC.height * h)
                        
                        face = frame[y_min:y_min+height, x_min:x_min+width]
                        predictions.append(face)
                        
                        # Start scanning when first face is detected
                        if not scanning:
                            scanning = True
                            start_time = time.time()
                            logger.info("Face detected! Starting scan...")
                        
                        # Draw detection box
                        self.mp_drawing.draw_detection(frame, detection)
                        
                        # Show scanning progress
                        elapsed = time.time() - start_time
                        remaining = max(0, duration - elapsed)
                        progress = min(1.0, elapsed / duration)
                        
                        # Progress bar
                        bar_width = 200
                        bar_height = 20
                        bar_x = (w - bar_width) // 2
                        bar_y = h - 50
                        
                        # Background
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                        # Progress
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                        
                        # Text (adjust coordinates for horizontal flip)
                        text_x = w - bar_x - 200
                        cv2.putText(frame, f"Scanning... {remaining:.1f}s", 
                                   (text_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Frames: {len(predictions)}", 
                                   (text_x, bar_y + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (255, 255, 255), 2)
                                   
                    else:
                        scanning = False
                        cv2.putText(frame, "Position your face in the camera", (w - 400, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "Make sure only 1 face is visible", (w - 450, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Show window (flip only for display)
                    display_frame = cv2.flip(frame, 1)
                    cv2.imshow("Skin Condition Scanner", display_frame)
                    
                    # Break on ESC key
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    
                    # Stop after duration seconds (only if scanning has started)
                    if scanning and time.time() - start_time >= duration:
                        break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Collected {len(predictions)} face crops")
        
        if len(predictions) > 0:
            return self.analyze_skin_condition(predictions)
        else:
            raise ValueError("No face crops collected during scan")

# Convenience function for easy usage
def scan_and_analyze(duration: float = 3.0, model_path: str = "build_model/models/best_model_twohead_0922_031215.pth") -> Dict[str, Any]:
    """
    Convenience function to scan face and analyze skin condition
    
    Args:
        duration: Duration to scan in seconds (default: 3.0)
        model_path: Path to the model file
    
    Returns:
        Dictionary containing analysis results
    """
    scanner = FaceScanner(model_path)
    return scanner.scan_face_from_camera(duration)

if __name__ == "__main__":
    # Example usage
    try:
        results = scan_and_analyze(duration=3.0)
        print("Analysis Results:")
        print(f"Overall Assessment: {'Healthy' if results['overall_assessment']['is_healthy'] else 'Has Condition'}")
        print(f"Most Likely Condition: {results['prediction']['condition']}")
        print(f"Confidence: {results['prediction']['confidence_percentage']}%")
    except Exception as e:
        print(f"Error: {e}")

