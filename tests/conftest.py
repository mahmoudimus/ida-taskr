"""Root pytest configuration for all tests.

IMPORTANT: Import Qt BEFORE idapro to avoid "PySide6 can only be used from GUI" error.
idapro sets up an import hook that blocks PySide6 if imported after.
"""

import pytest

# Import Qt first - must happen before any idapro import
try:
    from PySide6 import QtCore
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt5 import QtCore
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        QtCore = None


@pytest.fixture(scope="session")
def qapp():
    """Create a QCoreApplication for tests that need Qt event loop."""
    if not QT_AVAILABLE or QtCore is None:
        pytest.skip("Qt not available")

    app = QtCore.QCoreApplication.instance()
    if app is None:
        import sys
        app = QtCore.QCoreApplication(sys.argv)
    yield app
    # Don't quit - other tests might need it


@pytest.fixture(scope="function")
def qapp_function(qapp):
    """Function-scoped fixture for tests that need a fresh Qt context."""
    yield qapp

