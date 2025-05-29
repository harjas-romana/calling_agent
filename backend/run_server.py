#!/usr/bin/env python3
"""
Romana Restaurant AI Voice Agent - Development Server Startup Script
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
from main import app  # Import the FastAPI app from main.py

def check_requirements():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'requests', 'sounddevice', 
        'soundfile', 'numpy', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Please install requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not openrouter_key:
        print("⚠️  OPENROUTER_API_KEY not found in environment variables")
        openrouter_key = input("Enter your OpenRouter API key: ").strip()
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
        else:
            print("❌ OpenRouter API key is required")
            return False
    
    if not elevenlabs_key:
        print("⚠️  ELEVENLABS_API_KEY not found in environment variables")
        elevenlabs_key = input("Enter your ElevenLabs API key: ").strip()
        if elevenlabs_key:
            os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
        else:
            print("❌ ElevenLabs API key is required")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "static", "static/audio", "backups"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def start_backend_server():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    
    try:
        # Import and run the main app
        from main import app
        
        # Start server in a separate process for production
        # For development, we'll use the direct method
        print("🌐 Backend server starting at http://localhost:8000")
        print("📚 API Documentation available at http://localhost:8000/docs")

        app.run(
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error starting backend server: {str(e)}")
        return False
    
    return True

def show_startup_info():
    """Display startup information"""
    print("=" * 60)
    print("🍝 ROMANA RESTAURANT AI VOICE AGENT")
    print("=" * 60)
    print()
    print("🔧 BACKEND SETUP:")
    print("   • Backend Server: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print()
    print("💻 FRONTEND SETUP:")
    print("   • Navigate to your React project directory")
    print("   • Run: npm install")
    print("   • Run: npm run dev")
    print("   • Frontend will be available at: http://localhost:5173")
    print()
    print("🎤 FEATURES:")
    print("   • Voice Recognition & Speech Synthesis")
    print("   • Restaurant Reservations")
    print("   • Food Ordering System")
    print("   • Menu & Information Queries")
    print("   • Multi-step Conversation Flow")
    print()
    print("🔑 API ENDPOINTS:")
    print("   • POST /chat - Main chat interface")
    print("   • POST /voice-chat - Voice-focused chat")
    print("   • POST /reservations - Create reservations")
    print("   • POST /orders - Place food orders")
    print("   • GET /menu - Get restaurant menu")
    print("   • GET /restaurant-info - Get restaurant info")
    print()
    print("=" * 60)

def main():
    """Main startup function"""
    show_startup_info()
    
    print("🔍 Checking system requirements...")
    
    # Check if required files exist
    required_files = ["rag_layer.py", "voice_agent.py", "main.py"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are in the correct directory.")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    print("✅ All required packages are installed")
    
    # Check API keys
    if not check_api_keys():
        return
    
    print("✅ API keys configured")
    
    # Create directories
    create_directories()
    print("✅ Directories created")
    
    # Start the server
    print("\n" + "=" * 60)
    print("🚀 STARTING ROMANA RESTAURANT AI VOICE AGENT")
    print("=" * 60)
    
    try:
        start_backend_server()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that all files are in the correct directory")
        print("2. Verify API keys are correct")
        print("3. Ensure port 8000 is not in use")
        print("4. Check the logs directory for detailed error messages")

if __name__ == "__main__":
    main()
    