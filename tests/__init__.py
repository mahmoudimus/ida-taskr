import os
import pathlib
import sys
import unittest

import idapro

# Set framework path for IDA Pro
os.environ["DYLD_FRAMEWORK_PATH"] = (
    "/Applications/IDA Professional 9.1.app/Contents/Frameworks:"
    + os.environ.get("DYLD_FRAMEWORK_PATH", "")
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))
