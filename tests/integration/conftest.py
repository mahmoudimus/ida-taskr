"""Pytest configuration for integration tests."""

import os
import sys
from pathlib import Path

import pytest

# Add tests/integration to path so anti_deob module can be imported
tests_integration_dir = Path(__file__).parent
if str(tests_integration_dir) not in sys.path:
    sys.path.insert(0, str(tests_integration_dir))
