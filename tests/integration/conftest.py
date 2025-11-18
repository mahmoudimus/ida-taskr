"""Pytest configuration for IDA Pro integration tests."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ida_available():
    """Check if IDA Pro is available in the environment."""
    try:
        import idaapi
        return True
    except ImportError:
        pytest.skip("IDA Pro not available in this environment")
        return False


@pytest.fixture(scope="session")
def qt_framework():
    """Determine which Qt framework is available (PyQt5 or PySide6).

    Only works when running in IDA GUI mode (idaapi.is_idaq()).
    Skips tests if Qt is not available or not in GUI mode.
    """
    # Check if we're in IDA GUI mode
    try:
        import idaapi
        if not idaapi.is_idaq():
            pytest.skip("Not running in IDA GUI mode (idaapi.is_idaq() == False)")
            return None
    except (ImportError, AttributeError):
        # idaapi not available or is_idaq doesn't exist
        pytest.skip("IDA API not available or cannot determine GUI mode")
        return None

    # Try to import Qt frameworks
    try:
        import PyQt5
        return "PyQt5"
    except (ImportError, NotImplementedError):
        pass

    try:
        import PySide6
        return "PySide6"
    except (ImportError, NotImplementedError):
        pass

    pytest.skip("No Qt framework available (PyQt5 or PySide6) or not in GUI mode")
    return None


@pytest.fixture(scope="session")
def test_binary():
    """Provide path to a test binary for IDA analysis."""
    resources_dir = Path(__file__).parent.parent / "_resources" / "bin"

    # Look for any suitable test binary
    binary_patterns = ["*.exe", "*.elf", "*.bin", "*.dll", "*.so"]
    for pattern in binary_patterns:
        binaries = list(resources_dir.glob(pattern))
        if binaries:
            return binaries[0]

    # If no binary found, skip tests that require it
    pytest.skip("No test binary found in _resources/bin")
    return None


@pytest.fixture
def temp_idb(test_binary):
    """Create a temporary IDA database from the test binary."""
    if test_binary is None:
        pytest.skip("No test binary available")
        return None

    # Create a temporary directory for the IDB
    temp_dir = tempfile.mkdtemp(prefix="ida_test_")
    temp_binary = Path(temp_dir) / test_binary.name

    try:
        # Copy binary to temp location (required for idalib)
        shutil.copy(test_binary, temp_binary)
        yield temp_binary
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def ida_database(temp_idb):
    """Open an IDA database and perform auto-analysis."""
    if temp_idb is None:
        pytest.skip("No temporary IDB available")
        return None

    try:
        # Try idalib (IDA 9.0+)
        import idalib
        import idaapi

        # Open the database
        db = idalib.open_database(str(temp_idb), run_auto_analysis=True)

        # Wait for analysis to complete
        idaapi.auto_wait()

        yield db

        # Close database
        db.close()
    except (ImportError, AttributeError):
        # Fallback for older IDA versions or when idalib is not available
        pytest.skip("idalib not available or database opening failed")
        yield None


@pytest.fixture(scope="session")
def output_dir():
    """Provide path to output directory for test artifacts."""
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    return out_dir
