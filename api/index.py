"""
Vercel Serverless Function Entrypoint
Exposes the FastAPI application to Vercel's ASGI runtime.
"""

import sys
import os

import traceback

# Ensure root directory is in sys.path so app and its dependencies can be imported
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Top-level imports and exports required by Vercel's Python AST analyzer
from app import app

# Explicit top-level handler definitions recognized by Vercel Serverless Functions
handler = app
application = app
