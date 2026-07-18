"""Make ``axil_client`` importable when the suite is run in-tree (no install)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
