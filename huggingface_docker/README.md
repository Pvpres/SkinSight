# SkinSight API
A comprehensive dermatological image classification system, delivered as a containerized API for easy deployment and use.

---

## 🚀 Quick Start: Deploying to Hugging Face

This project is designed to be deployed as a Docker container on **Hugging Face Spaces**.

### 1. Create a New Hugging Face Space
- Select **Docker** as the Space SDK  
- Choose the **Blank template**  
- Use the **free CPU basic hardware**

### 2. Clone the Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
3. Add Project Files
Copy the following files into the cloned folder:

Dockerfile

requirements_api.txt

app.py

face_scanner.py

model.py

Crucially, create a prod_model/ directory and place your trained model file (.pth) inside it.

4. Push to Deploy
bash
Copy code
git add .
git commit -m "Initial application deployment"
git push
Hugging Face will automatically build the Docker image and launch your API.

📁 Project Structure
graphql
Copy code
SkinSight/
├── Dockerfile              # Instructions to build the Docker image
├── requirements_api.txt    # Python dependencies
├── app.py                  # The main FastAPI application logic and API endpoints
├── face_scanner.py         # Handles face detection and skin analysis
├── model.py                # Defines the PyTorch model architecture
└── prod_model/
    └── best_model_...pth   # The trained model weights (You must upload this!)
🔧 Features
Core Functionality
Real-time Skin Analysis via simple REST API endpoints

High-Accuracy Face Detection using MediaPipe

Deep Learning Classification of skin conditions (Acne, Dryness, etc.)

Batch Processing Endpoint for analyzing multiple images in one request

Deployment & Infrastructure
Containerized with Docker for portability

Optimized for CPU inference (free/low-cost deployment friendly)

FastAPI Backend with interactive API docs

🔌 API Endpoints & Usage
The API will be available at your Hugging Face Space URL.

Health Check
GET /health
Checks if the model is loaded and API is ready.

Response:

json
Copy code
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
Single Image Analysis
POST /analyze
Analyzes a single image for skin conditions.

Request Body:

json
Copy code
{
  "image_data": "BASE64_ENCODED_IMAGE_STRING"
}
Example Usage (curl):

bash
Copy code
curl -X POST "YOUR_SPACE_URL/analyze" \
-H "Content-Type: application/json" \
-d '{"image_data": "/9j/4AAQSkZJRgABAQ..."}'
🔍 Troubleshooting
Application Fails to Build on Hugging Face

Check build logs in your Space

Ensure prod_model/your_model.pth exists

Verify filenames in code match uploaded files

"No Face Detected" Error

API requires exactly one clear face

Try a better lit, direct-angle image

Connection Errors

Ensure app is running and not crashed

Check runtime logs in Hugging Face Space

🤝 Contributing
Fork the repository

Create a feature branch

Add tests for new functionality

Ensure all tests pass

Submit a pull request

📄 License
This project is licensed under the MIT License.
