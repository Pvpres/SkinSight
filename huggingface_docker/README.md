SkinSight
A comprehensive dermatological image classification system, delivered as a containerized API for easy deployment and use.
🚀 Quick Start: 
Deploying to Hugging FaceThis project is designed to be deployed as a Docker container on Hugging Face Spaces.
Create a New Hugging Face Space:Select Docker as the Space SDK.
Choose the Blank template.Use the free CPU basic hardware.
Clone the repository:git clone [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
cd YOUR_SPACE_NAME
Add all project files to the cloned folder:Dockerfile, requirements_api.txt, app.py, face_scanner.py, model.pyCrucially, create a prod_model/ directory and place your trained model file (.pth) inside it.Push to deploy:git add .
git commit -m "Initial application deployment"
git push
Hugging Face will automatically build the Docker image and launch your API.📁 Project StructureSkinSight/
├── Dockerfile              # Instructions to build the Docker image
├── requirements_api.txt     # Python dependencies
├── app.py                  # The main FastAPI application logic and API endpoints
├── face_scanner.py         # Handles face detection and skin analysis
├── model.py                # Defines the PyTorch model architecture
└── prod_model/
    └── best_model_...pth   # The trained model weights (You must upload this!)
🔧 FeaturesCore FunctionalityReal-time Skin Analysis via simple REST API endpoints.High-Accuracy Face Detection using MediaPipe to isolate the region of interest.Deep Learning Classification to determine if skin is healthy or has a condition, and to identify the specific condition (Acne, Dryness, etc.).Batch Processing Endpoint for analyzing multiple images in a single request.Deployment & InfrastructureContainerized with Docker for a portable, consistent, and isolated environment.Optimized for CPU Inference, making it suitable for free and low-cost deployment tiers.FastAPI Backend providing high performance and automatic interactive API documentation.🔌 API Endpoints & UsageThe API will be available at your Hugging Face Space URL.Health CheckGET /healthChecks if the model is loaded and the API is ready.Response:{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
Single Image AnalysisPOST /analyzeAnalyzes a single image for skin conditions.Request Body:{
  "image_data": "BASE64_ENCODED_IMAGE_STRING"
}
Example Usage (curl):curl -X POST "YOUR_SPACE_URL/analyze" \
-H "Content-Type: application/json" \
-d '{"image_data": "/9j/4AAQSkZJRgABAQ..."}'
🔍 TroubleshootingCommon IssuesApplication Fails to Build on Hugging Face:Check the build logs in your Space. The most common error is a missing prod_model/your_model.pth file. Ensure it has been uploaded.Verify that all filenames in your code (e.g., the model path in face_scanner.py) exactly match the files you've uploaded."No Face Detected" Error:The API requires exactly one clear face to be visible in the image. Try using a different image with better lighting and a more direct angle.Connection Errors:Ensure your application is running and has not crashed. Check the runtime logs on your Hugging Face Space.🤝 ContributingFork the repositoryCreate a feature branchAdd tests for new functionalityEnsure all tests passSubmit a pull request📄 LicenseThis project is licensed under the MIT License.
