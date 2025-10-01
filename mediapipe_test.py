from sys import stderr
import mediapipe as mp
import cv2
import numpy as np
import torch
from torchvision import transforms
# Load your trained model (example with PyTorch)
from build_model.model import DermatologyClassifier

# Setup device
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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Example preprocessing pipeline (adjust to your training setup)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

class_labels = ["acne", "dry", "eczema", "healthy", "oily"]

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

            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Crop face
            face = frame[y_min:y_min+height, x_min:x_min+width]

            if face.size != 0:
                # Preprocess
                face_tensor = transform(face).unsqueeze(0).to(device)  # add batch dim and move to device

                with torch.no_grad():
                    binary_output, disease_outputs = model(face_tensor) # forward pass

                    # Get predictions
                    binary_probs = torch.softmax(binary_output, dim=1)
                    disease_probs = torch.softmax(disease_outputs, dim=1)
                    
                    # Assuming:
                    # binary_probs -> tensor([P(healthy), P(disease)])
                    # disease_probs -> tensor of probabilities for each disease type

                    # Print binary classification probs
                    print(f"Healthy probability: {binary_probs[0][0].item():.4f}")
                    print(f"Disease probability: {binary_probs[0][1].item():.4f}")

                    if binary_probs.argmax(dim=1)[0] == 0:
                        pred_class = "healthy"
                        confidence = binary_probs[0][0].item()
                    else:
                        disease_class_labels = ["acne", "dry", "eczema", "oily"]

                        # Print detailed disease probabilities
                        for i, label in enumerate(disease_class_labels):
                            print(f"{label} probability: {disease_probs[0][i].item():.4f}")

                        pred_idx = disease_probs.argmax(dim=1)[0].item()
                        pred_class = disease_class_labels[pred_idx]
                        confidence = disease_probs[0][pred_idx].item()

                    print(f"\nPredicted class: {pred_class} (confidence: {confidence:.4f})")

                # Show prediction
                cv2.putText(frame, f"{pred_class} ({confidence:.2f})", 
                            (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0,255,0), 2)

            # Draw face box
            mp_drawing.draw_detection(frame, detection)
        else:
            cv2.putText(frame, "Waiting for 1 face...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Skin Condition Scanner", cv2.flip(frame, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
