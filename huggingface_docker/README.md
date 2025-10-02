SkinSight API: Skin Condition AnalysisThis repository contains the source code for the SkinSight API, a real-time skin condition analysis application powered by a deep learning model. The application is built with FastAPI and is containerized using Docker for easy deployment, including on Hugging Face Spaces.🚀 Tech StackBackend: FastAPIML Framework: PyTorchCV Libraries: OpenCV, MediaPipeContainerization: DockerDeployment: Hugging Face Spaces📂 Project Structure.
├── 📄 Dockerfile              # Instructions to build the Docker image
├── 📄 requirements_api.txt     # Python dependencies
├── 🐍 app.py                  # The main FastAPI application logic and API endpoints
├── 🐍 face_scanner.py         # Handles face detection and skin analysis
├── 🐍 model.py                # Defines the PyTorch model architecture
└── 📁 prod_model/
    └── 📦 best_model_...pth   # The trained model weights (You must upload this!)
🐳 Understanding the Docker SetupThe Dockerfile provides a recipe for creating a portable, self-contained environment for the application. Here's a step-by-step breakdown of what it does:FROM python:3.12-slimStarts with an official, lightweight Python 3.12 image. This keeps the final image size smaller.RUN apt-get update && apt-get install -y ...Installs system-level libraries (libgl1, libglib2.0-0) that are required by OpenCV to function correctly for image processing within the container.WORKDIR /appSets the working directory inside the container to /app. All subsequent commands will be run from this location.COPY requirements_api.txt .Copies only the requirements_api.txt file into the container.RUN pip install --no-cache-dir -r requirements_api.txtInstalls all the Python dependencies. This step is done before copying the application code to take advantage of Docker's layer caching. If you don't change your requirements, this layer won't be rebuilt, making future builds much faster.The requirements_api.txt is optimized to use a CPU-only version of PyTorch, which is ideal for cost-effective Hugging Face Spaces.COPY . .Copies the rest of the application code (like app.py, face_scanner.py, etc.) into the /app directory in the container.EXPOSE 7860Informs Docker that the application will listen on port 7860. Hugging Face Spaces will use this to route traffic to your application.CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]The command that runs when the container starts. It launches the uvicorn server to run the FastAPI application defined in app.py.🤗 How to Deploy on Hugging Face SpacesFollow these steps to deploy your application.Prerequisites:A Hugging Face account.Git installed on your machine.Step-by-Step InstructionsCreate a New Space:On the Hugging Face website, click on your profile picture and select "New Space".Configure Your Space:Owner & Space name: Choose a name for your project (e.g., SkinSight-App).License: Select a license (e.g., MIT).Select the Space SDK: Crucially, select Docker.Choose a template: Select Blank.Hardware: You can start with the free CPU basic hardware.Click "Create Space".Clone the Space Repository:Hugging Face will create a Git repository for your Space. Copy the command provided to clone it to your local machine.git clone [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
Add Your Project Files:Copy all your project files into the cloned repository folder:Dockerfilerequirements_api.txtapp.pyface_scanner.pymodel.pyIMPORTANT: You must also create the prod_model directory and add your trained model file (.pth) inside it. The application will fail to start without the model weights.Commit and Push Your Files:Use Git to upload your files to the Hugging Face repository.# Navigate into your repository
cd YOUR_SPACE_NAME

# Add all the files
git add .

# Commit the files
git commit -m "Initial application upload"

# Push to Hugging Face
git push
Watch it Build!Go back to your Space page on the Hugging Face website. You will see the build logs. Hugging Face automatically detects your Dockerfile and starts building the container image.This might take a few minutes. Once it's done, your application will be live and accessible! The page will show the running FastAPI application.⭐ Best Practice: Use a .dockerignore FileTo prevent sending unnecessary files (like __pycache__ or .git) into your Docker image, create a file named .dockerignore in your project's root directory with the following content:# .dockerignore

# Git files
.git
.gitignore

# Python specific
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment files
.env*

# IDE and OS files
.vscode/
.idea/
*.DS_Store
