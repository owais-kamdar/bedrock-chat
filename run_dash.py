"""
Entry point for Dashboard app
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit with the app file
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/dashboard.py",
        "--server.port=8502",
        "--server.address=0.0.0.0"
    ]) 