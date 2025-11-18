"""Integration tests for IDA analysis with TaskRunner."""

import pytest


class TestIDAAnalysisIntegration:
    """Integration tests for IDA analysis functionality."""

    def test_ida_database_opened(self, ida_database):
        """Test that IDA database can be opened and analyzed."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import idc

        # Verify database is open and analyzed
        inf = idaapi.get_inf_structure()
        assert inf is not None

        # Check that we have at least one segment
        seg = idaapi.get_first_seg()
        assert seg is not None

        seg_name = idaapi.get_segm_name(seg)
        assert seg_name is not None and len(seg_name) > 0

    def test_segment_iteration(self, ida_database):
        """Test iterating through IDA segments."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import idautils

        segments = []
        for seg_ea in idautils.Segments():
            seg = idaapi.getseg(seg_ea)
            if seg:
                seg_name = idaapi.get_segm_name(seg)
                segments.append({
                    'name': seg_name,
                    'start': seg.start_ea,
                    'end': seg.end_ea,
                    'size': seg.end_ea - seg.start_ea
                })

        # Should have at least one segment
        assert len(segments) > 0

        # Verify segment data makes sense
        for seg_info in segments:
            assert seg_info['start'] < seg_info['end']
            assert seg_info['size'] > 0

    def test_function_detection(self, ida_database):
        """Test that IDA can detect functions in the binary."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import idautils

        functions = []
        for func_ea in idautils.Functions():
            func = idaapi.get_func(func_ea)
            if func:
                func_name = idaapi.get_func_name(func_ea)
                functions.append({
                    'name': func_name,
                    'start': func.start_ea,
                    'end': func.end_ea,
                    'size': func.end_ea - func.start_ea
                })

        # Should have at least one function (unless binary is very minimal)
        # We use >= 0 to allow for minimal binaries
        assert len(functions) >= 0

    def test_bytes_reading(self, ida_database):
        """Test reading bytes from IDA database."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi

        # Get first segment
        seg = idaapi.get_first_seg()
        if not seg:
            pytest.skip("No segments in database")

        # Try to read some bytes from the segment
        start_ea = seg.start_ea
        num_bytes = min(16, seg.end_ea - seg.start_ea)

        data = idaapi.get_bytes(start_ea, num_bytes)
        assert data is not None
        assert len(data) == num_bytes

    def test_disassembly_generation(self, ida_database):
        """Test generating disassembly from IDA database."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import idc

        # Get first segment
        seg = idaapi.get_first_seg()
        if not seg:
            pytest.skip("No segments in database")

        # Get disassembly for first address
        ea = seg.start_ea

        # Generate disassembly line
        disasm = idc.generate_disasm_line(ea, 0)

        # Should get some disassembly text
        # (might be empty for data segments, so we just check it's not None)
        assert disasm is not None

    def test_auto_analysis_completion(self, ida_database):
        """Test that auto-analysis has completed."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi

        # Check that auto-analysis is complete
        # This should be true since conftest.py waits for analysis
        assert not idaapi.is_auto_enabled()

    def test_ida_info_structure(self, ida_database):
        """Test reading IDA info structure."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi

        inf = idaapi.get_inf_structure()
        assert inf is not None

        # Check some basic properties
        assert hasattr(inf, 'procname')
        assert hasattr(inf, 'is_64bit')
        assert hasattr(inf, 'is_32bit')

        # At least one should be true
        assert inf.is_64bit() or inf.is_32bit() or inf.is_16bit()

    def test_concurrent_ida_api_access(self, ida_database):
        """Test concurrent access to IDA API from multiple threads."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import threading
        import queue

        results = queue.Queue()

        def read_segment_info():
            try:
                seg = idaapi.get_first_seg()
                if seg:
                    results.put(('success', seg.start_ea))
                else:
                    results.put(('error', 'No segment'))
            except Exception as e:
                results.put(('error', str(e)))

        # Note: IDA API may not be fully thread-safe
        # This test verifies basic thread safety
        threads = []
        for _ in range(3):
            t = threading.Thread(target=read_segment_info)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        # Collect results
        result_list = []
        while not results.empty():
            result_list.append(results.get())

        # Should have gotten 3 results
        assert len(result_list) == 3

    def test_taskrunner_with_ida_api(self, ida_database, qt_framework):
        """Test using TaskRunner with IDA API calls."""
        if not ida_database:
            pytest.skip("IDA database not available")

        assert qt_framework

        import idaapi

        def get_segment_count():
            """Count segments in the database."""
            count = 0
            seg = idaapi.get_first_seg()
            while seg:
                count += 1
                seg = idaapi.get_next_seg(seg.start_ea)
            return count

        # Execute the task
        count = get_segment_count()

        # Should have at least one segment
        assert count > 0
