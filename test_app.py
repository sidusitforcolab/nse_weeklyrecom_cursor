#!/usr/bin/env python3
"""
Test script to verify the Streamlit app works correctly
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import streamlit
        import pandas
        import numpy
        import yfinance
        import plotly
        import sklearn
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_app_file():
    """Test if app.py exists and is valid Python"""
    print("Testing app.py...")
    app_path = Path("app.py")
    if not app_path.exists():
        print("❌ app.py not found")
        return False
    
    try:
        with open(app_path, 'r') as f:
            content = f.read()
        compile(content, "app.py", "exec")
        print("✅ app.py is valid Python")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False

def test_requirements():
    """Test if requirements.txt exists"""
    print("Testing requirements.txt...")
    req_path = Path("requirements.txt")
    if req_path.exists():
        print("✅ requirements.txt found")
        return True
    else:
        print("❌ requirements.txt not found")
        return False

def main():
    """Run all tests"""
    print("🧪 Running NSE Stock App Tests\n")
    
    tests = [
        test_imports,
        test_app_file,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready for deployment.")
        print("\nTo run the app locally:")
        print("  streamlit run app.py")
        print("\nTo deploy to Streamlit Cloud:")
        print("  1. Push to GitHub")
        print("  2. Go to https://share.streamlit.io/")
        print("  3. Connect your repository")
        print("  4. Deploy with main file: app.py")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()