"""
streamlit_app.py
----------------
Root entrypoint for Streamlit Community Cloud and other cloud hosts.
"""

import sys
import pathlib

# Ensure project root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Run the application
import app.app
