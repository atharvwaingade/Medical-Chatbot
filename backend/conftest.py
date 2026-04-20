"""
conftest.py
===========
Ensure the ``backend/`` package directory is on ``sys.path`` so that
``import app.*`` works regardless of the working directory, and set the
working directory to the repository root so that relative paths such as
``"backend/data/..."`` resolve correctly in every test file.
"""
import os
import sys

# Absolute path to backend/
_BACKEND_DIR = os.path.dirname(__file__)
# Absolute path to repo root (one level up from backend/)
_REPO_ROOT = os.path.dirname(_BACKEND_DIR)

# Add backend/ to Python path once
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Ensure relative paths in tests resolve from the repo root
os.chdir(_REPO_ROOT)
