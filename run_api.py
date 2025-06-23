"""
Entry point for Flask API
"""
import os

if __name__ == "__main__":
    from src.app.flaskapp import app
    # Use Render's PORT environment variable (defaults to 5003 for local dev)
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=False) 