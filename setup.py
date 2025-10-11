#!/usr/bin/env python3
"""
Setup script for LangChain Chat Application
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def setup_env_file():
    """Set up environment file"""
    env_file = ".env"
    template_file = "env_template.txt"
    
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    if os.path.exists(template_file):
        print(f"📝 Creating {env_file} from template...")
        with open(template_file, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print(f"✅ {env_file} created! Please edit it with your API keys.")
        return True
    else:
        print(f"❌ Template file {template_file} not found")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up LangChain Chat Application...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Setup environment file
    if not setup_env_file():
        print("❌ Setup failed during environment file creation")
        return
    
    print("=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your OpenAI API key")
    print("2. Run the basic app: streamlit run chat_app.py")
    print("3. Run the advanced app: streamlit run advanced_chat_app.py")
    print("\n🔑 Get your OpenAI API key from: https://platform.openai.com/api-keys")

if __name__ == "__main__":
    main()
