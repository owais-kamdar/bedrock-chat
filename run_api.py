"""
Entry point for Flask API
"""

if __name__ == "__main__":
    from src.app.flaskapp import app
    app.run(host="0.0.0.0", port=5001, debug=True) 