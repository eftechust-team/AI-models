"""
Vercel serverless function wrapper for Flask app
Vercel automatically handles WSGI apps, so we just need to import and expose the app
"""
import sys
import os

# Add parent directory to path so we can import app.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent for relative file paths
os.chdir(parent_dir)

# Try to import opencv with fallback
try:
    import cv2
except ImportError:
    # If opencv fails to import, create a dummy module
    import types
    cv2 = types.ModuleType('cv2')
    # This will cause errors at runtime, but allows the app to load
    # The actual error will be more informative
    pass

# Import the Flask app - Vercel will automatically handle it as a WSGI app
try:
    from app import app
except ImportError as e:
    # If import fails, create a minimal error app
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return f"<h1>Import Error</h1><p>Failed to import app: {str(e)}</p>", 500

# Vercel expects the app to be available
# The @vercel/python builder automatically wraps WSGI apps

