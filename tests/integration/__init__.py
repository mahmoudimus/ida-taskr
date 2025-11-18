"""Integration tests for IDA Taskr with real IDA Pro.

These tests require pytest and a real IDA Pro environment.
They will be skipped if run via unittest discovery.
"""

import sys

# Check if we're being imported by unittest - if so, skip this entire module
if 'unittest' in sys.modules and 'pytest' not in sys.modules:
    import unittest
    raise unittest.SkipTest("Integration tests require pytest - use 'pytest tests/integration/' to run them")
