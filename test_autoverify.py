"""
Test script for AutoVerify system
================================

This script tests the basic functionality of the AutoVerify system
without requiring the full Streamlit interface.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from autoverify.core.base_agent import BaseAgent, AgentMessage
        print("[OK] Base agent imports successful")
    except ImportError as e:
        print(f"[ERROR] Base agent import failed: {e}")
        return False
    
    try:
        from autoverify.agents.generator_agent import GeneratorAgent
        print("[OK] Generator agent import successful")
    except ImportError as e:
        print(f"[ERROR] Generator agent import failed: {e}")
        return False
    
    try:
        from autoverify.agents.verifier_agent import VerifierAgent
        print("[OK] Verifier agent import successful")
    except ImportError as e:
        print(f"[ERROR] Verifier agent import failed: {e}")
        return False
    
    try:
        from autoverify.agents.correction_agent import CorrectionAgent
        print("[OK] Correction agent import successful")
    except ImportError as e:
        print(f"[ERROR] Correction agent import failed: {e}")
        return False
    
    try:
        from autoverify.agents.audit_agent import AuditAgent
        print("[OK] Audit agent import successful")
    except ImportError as e:
        print(f"[ERROR] Audit agent import failed: {e}")
        return False
    
    try:
        from autoverify.core.orchestrator import AutoVerifyOrchestrator
        print("[OK] Orchestrator import successful")
    except ImportError as e:
        print(f"[ERROR] Orchestrator import failed: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test that agents can be initialized."""
    print("\nTesting agent initialization...")
    
    try:
        from autoverify.agents.generator_agent import GeneratorAgent
        
        # Test with default config
        generator = GeneratorAgent()
        print("[OK] Generator agent initialized with default config")
        
        # Test with custom config
        custom_config = {
            "primary_model": "gpt-3.5-turbo",
            "temperature": 0.5
        }
        generator_custom = GeneratorAgent(config=custom_config)
        print("[OK] Generator agent initialized with custom config")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Agent initialization failed: {e}")
        return False

def test_orchestrator_initialization():
    """Test that the orchestrator can be initialized."""
    print("\nTesting orchestrator initialization...")
    
    try:
        from autoverify.core.orchestrator import AutoVerifyOrchestrator
        
        # Test with default config
        orchestrator = AutoVerifyOrchestrator()
        print("[OK] Orchestrator initialized with default config")
        
        # Test with custom config
        custom_config = {
            "generator": {
                "primary_model": "gpt-3.5-turbo",
                "temperature": 0.7
            },
            "verifier": {
                "confidence_threshold": 0.8
            }
        }
        orchestrator_custom = AutoVerifyOrchestrator(config=custom_config)
        print("[OK] Orchestrator initialized with custom config")
        
        # Test system status
        status = orchestrator.get_system_status()
        print(f"[OK] System status retrieved: {status['system_status']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Orchestrator initialization failed: {e}")
        return False

def test_api_key():
    """Test that API keys are available."""
    print("\nTesting API key availability...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("[OK] OpenAI API key found")
        return True
    else:
        print("[ERROR] OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        return False

def main():
    """Run all tests."""
    print("AutoVerify System Test")
    print("=" * 50)
    
    # Check API key first
    if not test_api_key():
        print("\n[WARNING] Please set your OpenAI API key in the .env file to continue testing.")
        return
    
    # Test imports
    if not test_imports():
        print("\n[ERROR] Import tests failed. Please check your installation.")
        return
    
    # Test agent initialization
    if not test_agent_initialization():
        print("\n[ERROR] Agent initialization tests failed.")
        return
    
    # Test orchestrator initialization
    if not test_orchestrator_initialization():
        print("\n[ERROR] Orchestrator initialization tests failed.")
        return
    
    print("\n[SUCCESS] All tests passed! AutoVerify system is ready to use.")
    print("\nTo run the web interface:")
    print("streamlit run autoverify_app.py")

if __name__ == "__main__":
    main()
