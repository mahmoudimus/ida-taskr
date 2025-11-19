import unittest


class TestImports(unittest.TestCase):
    def test_taskrunner_import(self):
        try:
            from ida_taskr import TaskRunner

            print("✅ TaskRunner imported successfully")
        except ImportError:
            print("❌ TaskRunner import failed")
            self.fail("TaskRunner import failed")


if __name__ == "__main__":
    unittest.main()
