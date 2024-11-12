"""Pytest executes this file automatically.
The file ensures that the 'src' folder is part of the Python path.
"""

import sys
import os

# Adding the 'src' folder to pythens Sys path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
