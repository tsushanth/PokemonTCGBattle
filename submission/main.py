import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heuristic_agent import agent  # noqa: F401 -- Kaggle harness imports `agent` from this module
