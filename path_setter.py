"""This file eddits the python path so you can import everything from src"""

import sys
import os

# Setze den PYTHONPATH, sodass Python im 'src' Verzeichnis nach Modulen sucht
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))  
if project_root not in sys.path:
    sys.path.append(project_root)
