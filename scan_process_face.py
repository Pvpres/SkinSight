import mediapipe as mp
import cv2
import numpy as np
import time
from build_model.model import DermatologyClassifier
import torch
from torchvision import transforms


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for inference")
else:
    device = torch.device("cpu")
    print("Using CPU for inference")

model = DermatologyClassifier(num_classes=5, use_two_heads=True)  # Instantiate your model architecture
checkpoint = torch.load("build_model/models/best_model_twohead_0922_031215.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])
class_labels = ["acne", "dry", "eczema", "healthy", "oily"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

predictions = []
SCAN_DURATION = 3 # seconds
start_time = time.time()
scanning = False

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections and len(results.detections) == 1:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
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
                print("Face detected! Starting 3-second scan...")

            # Draw detection box
            mp_drawing.draw_detection(frame, detection)
            
            # Show scanning progress
            elapsed = time.time() - start_time
            remaining = max(0, SCAN_DURATION - elapsed)
            progress = min(1.0, elapsed / SCAN_DURATION)
            
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
            text_x = w - bar_x - 200  # Mirror the x coordinate
            cv2.putText(frame, f"Scanning... {remaining:.1f}s", 
                       (text_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {len(predictions)}", 
                       (text_x, bar_y + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
                       
        else:
            scanning = False
            # Adjust text coordinates for horizontal flip
            cv2.putText(frame, "Position your face in the camera", (w - 400, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Make sure only 1 face is visible", (w - 450, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show window (flip only for display, not for text rendering)
        display_frame = cv2.flip(frame, 1)
        cv2.imshow("Skin Condition Scanner", display_frame)

        # Break on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Stop after SCAN_DURATION seconds (only if scanning has started)
        if scanning and time.time() - start_time >= SCAN_DURATION:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Collected {len(predictions)} face crops")

# Process all collected frames in batches
if len(predictions) > 0:
    print("Processing collected face crops...")
    
    # Preprocess all face crops
    processed_faces = []
    for face in predictions:
        if face.size != 0:  # Make sure face crop is valid
            try:
                face_tensor = transform(face)
                processed_faces.append(face_tensor)
            except Exception as e:
                print(f"Error processing face crop: {e}")
                continue
    
    if len(processed_faces) > 0:
        # Batch all processed faces
        batch_tensor = torch.stack(processed_faces).to(device)
        print(f"Processing batch of {batch_tensor.shape[0]} faces...")
        
        with torch.no_grad():
            # Forward pass through model
            binary_output, disease_output = model(batch_tensor)
            
            # Get probabilities
            binary_probs = torch.softmax(binary_output, dim=1)
            disease_probs = torch.softmax(disease_output, dim=1)
            
            # Average predictions across all frames for more stable results
            avg_binary_probs = binary_probs.mean(dim=0)
            avg_disease_probs = disease_probs.mean(dim=0)
            
            print("\n" + "="*50)
            print("SKIN CONDITION ANALYSIS RESULTS")
            print("="*50)
            print(f"Analyzed {len(processed_faces)} frames over {SCAN_DURATION} seconds")
            print()
            
            # Binary classification results
            healthy_prob = avg_binary_probs[0].item()
            disease_prob = avg_binary_probs[1].item()
            print(f"Overall Health Assessment:")
            print(f"  Healthy: {healthy_prob:.3f} ({healthy_prob*100:.1f}%)")
            print(f"  Has Condition: {disease_prob:.3f} ({disease_prob*100:.1f}%)")
            print()
            
            if disease_prob > healthy_prob:
                # Show disease-specific probabilities
                disease_class_labels = ["acne", "dry", "eczema", "oily"]
                print("Condition Analysis:")
                for i, label in enumerate(disease_class_labels):
                    prob = avg_disease_probs[i].item()
                    print(f"  {label.capitalize()}: {prob:.3f} ({prob*100:.1f}%)")
                
                # Get most likely condition
                pred_idx = avg_disease_probs.argmax().item()
                pred_condition = disease_class_labels[pred_idx]
                confidence = avg_disease_probs[pred_idx].item()
                
                print()
                print(f"Most Likely Condition: {pred_condition.capitalize()}")
                print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            else:
                print("Assessment: Healthy skin detected")
                print(f"Confidence: {healthy_prob:.3f} ({healthy_prob*100:.1f}%)")
            
            print("="*50)
            
            # Individual frame analysis (optional - for debugging)
            print("\nIndividual Frame Analysis:")
            for i in range(min(5, len(processed_faces))):  # Show first 5 frames
                frame_binary = binary_probs[i]
                frame_disease = disease_probs[i]
                
                if frame_binary.argmax() == 0:
                    frame_pred = "healthy"
                    frame_conf = frame_binary[0].item()
                else:
                    frame_pred_idx = frame_disease.argmax().item()
                    frame_pred = disease_class_labels[frame_pred_idx]
                    frame_conf = frame_disease[frame_pred_idx].item()
                
                print(f"  Frame {i+1}: {frame_pred} ({frame_conf:.3f})")
    
    else:
        print("No valid face crops to process")
else:
    print("No face crops collected")

    