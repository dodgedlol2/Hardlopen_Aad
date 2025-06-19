#!/usr/bin/env python3
"""
Setup script for Running Performance Dashboard
"""

import os
import sys
import subprocess
import platform

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        '.streamlit',
        'data',
        'assets'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("💡 Please upgrade to Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def create_sample_data():
    """Create sample data file for testing"""
    sample_data = """# Sample Running Data
# This is just for reference - the app will use your Google Sheets data

Distance,Time,Date
100m,19.0,27-12-24
100m,19.5,2-4-25
200m,41.0,19-5-25
200m,41.0,14-4-25
"""
    
    with open('data/sample_data.txt', 'w') as f:
        f.write(sample_data)
    
    print("✅ Created sample data file in data/sample_data.txt")

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() == "Windows":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Running Dashboard.lnk")
            target = os.path.join(os.getcwd(), "run_app.py")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target}"'
            shortcut.WorkingDirectory = os.getcwd()
            shortcut.IconLocation = target
            shortcut.save()
            
            print("✅ Created desktop shortcut: Running Dashboard.lnk")
        except ImportError:
            print("💡 To create desktop shortcuts, install: pip install winshell pywin32")
        except Exception as e:
            print(f"⚠️  Could not create desktop shortcut: {e}")

def main():
    """Main setup function"""
    print("🏃‍♂️ Running Performance Dashboard Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        print("❌ Setup failed during package installation.")
        sys.exit(1)
    
    # Create sample data
    print("\n📄 Creating sample files...")
    create_sample_data()
    
    # Create desktop shortcut (Windows only)
    if platform.system() == "Windows":
        print("\n🔗 Creating desktop shortcut...")
        create_desktop_shortcut()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python run_app.py")
    print("2. Or run directly: streamlit run app.py")
    print("3. Open your browser to: http://localhost:8501")
    print("\n📊 Your running data will be loaded from Google Sheets automatically!")
    print("\n💡 Need help? Check the README.md file for detailed instructions.")

if __name__ == "__main__":
    main()
