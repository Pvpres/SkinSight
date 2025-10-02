# SkinSight API

A comprehensive dermatological image classification system, delivered as a containerized API for easy deployment and use.

---

## 🚀 Quick Start: Deploying to Hugging Face

This project is designed to be deployed as a Docker container on Hugging Face Spaces.

### 1. Create a New Hugging Face Space
- Select **Docker** as the Space SDK  
- Choose the **Blank template**  
- Use the **free CPU basic hardware**

### 2. Clone the repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

### 3. Add all project files to the cloned folder

Dockerfile, requirements_api.txt, app.py, face_scanner.py, model.py

Create a prod_model/ directory and place your trained model file (.pth) inside it
