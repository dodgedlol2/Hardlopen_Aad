#!/usr/bin/env python3
"""
Simple script to run the Running Performance Dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'numpy',
        'requests'
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
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_app():
    """Run the Streamlit application"""
    if not check_requirements():
        sys.exit(1)
    
    print("🚀 Starting Running Performance Dashboard...")
    print("📊 The app will open in your web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Thanks for using the Running Dashboard!")
    except Exception as e:
        print(f"❌ Error running the application: {e}")
        print("💡 Make sure 'app.py' is in the same directory as this script")

if __name__ == "__main__":
    run_app()
