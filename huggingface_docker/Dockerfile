# Dockerfile for the SkinSight FastAPI Application

# Step 1: Start with a lightweight Python base image.
FROM python:3.12-slim

# Step 2: Install system dependencies required by OpenCV.
# This fixes the "libGL.so.1" and "libgthread-2.0.so.0" errors.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set a working directory inside the container.
WORKDIR /app

# Step 4: Copy the requirements file first and install dependencies.
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Step 5: Copy the rest of your application code into the container.
COPY . .

# Step 6: Expose the port the application will run on.
EXPOSE 7860

# Step 7: Define the command to run your application.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]