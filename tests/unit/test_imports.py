"""Test basic imports."""


class TestImports:
    def test_taskrunner_import(self):
        """Test that TaskRunner can be imported."""
        from ida_taskr import TaskRunner

        assert TaskRunner is not None

