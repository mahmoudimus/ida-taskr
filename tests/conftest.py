"""Root pytest configuration for all tests.

IMPORTANT: Import Qt BEFORE idapro to avoid "PySide6 can only be used from GUI" error.
idapro sets up an import hook that blocks PySide6 if imported after.
"""

# Import Qt first - must happen before any idapro import
try:
    from PySide6 import QtCore
except ImportError:
    try:
        from PyQt5 import QtCore
    except ImportError:
        pass
