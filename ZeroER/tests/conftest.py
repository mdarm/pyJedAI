"""Make the local ``zeroer`` package importable, as in the root notebooks."""
import os
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))
