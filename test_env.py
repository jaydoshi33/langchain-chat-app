"""
Test script to verify environment variables are loaded correctly
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment Variables Test:")
print("=" * 40)
print(f"OPENAI_API_KEY found: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"Key starts with: {os.getenv('OPENAI_API_KEY', 'NOT_FOUND')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'NOT_FOUND'}")
print(f"LANGSMITH_API_KEY found: {bool(os.getenv('LANGSMITH_API_KEY'))}")
print(f"USER_AGENT: {os.getenv('USER_AGENT', 'NOT_SET')}")

# Test AutoVerify imports
try:
    from autoverify.core.orchestrator import AutoVerifyOrchestrator
    print("\nAutoVerify imports: SUCCESS")
    
    # Test initialization
    orchestrator = AutoVerifyOrchestrator()
    print("AutoVerify initialization: SUCCESS")
    
except Exception as e:
    print(f"\nAutoVerify error: {e}")
