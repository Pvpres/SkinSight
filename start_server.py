#!/usr/bin/env python3
"""
Startup script for the SkinSight FastAPI server
"""

import uvicorn
import sys
import os

def main():
    """Start the FastAPI server"""
    print("🚀 Starting SkinSight API Server")
    print("=" * 40)
    
    # Check if model file exists
    model_path = "build_model/models/best_model_twohead_0922_031215.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file exists before starting the server.")
        sys.exit(1)
    
    print(f"✅ Model file found: {model_path}")
    print("🌐 Starting server on http://localhost:8000")
    print("📖 API documentation available at http://localhost:8000/docs")
    print("🔍 Web interface available at http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

