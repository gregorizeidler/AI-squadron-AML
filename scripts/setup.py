#!/usr/bin/env python3
"""
Setup script for AI Squadron AML Detection System

This script helps with initial setup and configuration of the AML system.
"""
import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data/storage/analyses",
        "data/storage/customers", 
        "data/storage/transactions",
        "data/storage/mappings",
        "data/exports",
        "data/imports",
        "config/environments",
        "deployment/terraform/environments",
        "deployment/docker",
        "deployment/kubernetes"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_sample_config():
    """Create sample configuration files"""
    
    # Create sample environment file
    env_content = """# AI Squadron AML Configuration
# Copy this file to .env and update with your values

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# LLM Provider Configuration
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Optional: OpenAI Configuration
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_MODEL=gpt-4

# Risk Thresholds
HIGH_RISK_THRESHOLD=75
MEDIUM_RISK_THRESHOLD=45
LOW_RISK_THRESHOLD=25

# System Configuration
LOG_LEVEL=INFO
DEBUG=false
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("✅ Created .env.example file")
    
    # Create development config
    dev_config = """# Development Environment Configuration
environment: development
debug: true

aws:
  region: us-east-1
  
llm:
  provider: mock
  temperature: 0.0
  
risk:
  high_risk_threshold: 75
  medium_risk_threshold: 45
  low_risk_threshold: 25
  
monitoring:
  enabled: true
  metrics_interval: 30
"""
    
    Path("config/environments").mkdir(parents=True, exist_ok=True)
    with open("config/environments/development.yaml", "w") as f:
        f.write(dev_config)
    print("✅ Created development configuration")


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "langgraph",
        "langchain", 
        "pydantic",
        "boto3",
        "pandas",
        "pyyaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main setup function"""
    print("🛡️  AI Squadron AML Detection System - Setup")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create sample configuration
    print("\n⚙️  Creating configuration files...")
    create_sample_config()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies before proceeding")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and update with your credentials")
    print("2. Run: python -m src.main --demo")
    print("3. Explore the system with different scenarios")


if __name__ == "__main__":
    main()