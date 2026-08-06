"""Put ZeroER/ on sys.path so the tests import its top-level modules the same
way the notebooks do (``from module import ...``, cwd=ZeroER)."""
import os
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))
