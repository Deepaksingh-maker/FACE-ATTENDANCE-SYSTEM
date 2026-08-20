"""
face_utils_stub.py
Stub version of face_utils for Python 3.13 compatibility.
This is a temporary workaround - install Python 3.11 to use the full face_recognition library.
"""

import numpy as np

def get_single_face_encoding(image_np):
    """
    Stub: Returns a dummy encoding for testing UI.
    Real version requires Python 3.11 + dlib.
    """
    # Return a dummy 128-element encoding (like face_recognition does)
    return np.random.rand(128), None

def get_all_faces(image_np):
    """
    Stub: Returns dummy face data for testing.
    Real version requires Python 3.11 + dlib.
    """
    # Return empty list (no faces detected in stub mode)
    return []

def match_face(unknown_encoding, known_students, tolerance=0.5):
    """
    Stub: No matching in stub mode.
    Real version requires Python 3.11 + dlib.
    """
    return None

def show_stub_warning():
    """
    Display a warning message that face recognition is not available.
    """
    return """
    ⚠️ **STUB MODE**: Face recognition is currently disabled.
    
    To enable facial recognition features:
    1. Install Python 3.11 from python.org
    2. Reinstall packages: `pip install -r requirements.txt`
    
    The app UI is available in stub mode for testing.
    """
