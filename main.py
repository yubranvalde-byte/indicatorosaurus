"""
Root WSGI Entry Point Proxy.
Exposes the Flask `app` object from `api.main` for legacy/WSGI deployment configurations (e.g. `gunicorn main:app`).
"""
import sys
import os

# Ensure root and api directories are in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
