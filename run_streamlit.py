"""
Entry point for Streamlit app
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Get port from environment variable (for App Runner)
    port = os.getenv("PORT", "8501")
    
    # Run streamlit with the app file
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/streamlit_app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]) 