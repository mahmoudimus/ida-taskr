"""IDA Taskr test suite.

This test suite is organized into two categories:

- tests/unit/: Unit tests that use mocks and don't require IDA Pro or Qt
- tests/integration/: Integration tests that require a real IDA Pro environment

To run unit tests:
    python -m unittest discover -s tests/unit/

To run integration tests:
    pytest tests/integration/
"""
