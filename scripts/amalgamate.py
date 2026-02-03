#!/usr/bin/env python3
"""
Generate a single-file amalgamated version of ida-taskr.

This script combines all the ida-taskr modules into a single Python file
that can be easily distributed and used without installing the package.

Usage:
    python scripts/amalgamate.py [--output PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Order matters - modules are combined in this sequence
MODULES = [
    "helpers",
    "utils",
    "protocols",
    "qt_compat",
    "qtasyncio",
    "worker",
    "launcher",
    "task_runner",
]

HEADER = '''\
"""
ida-taskr - Amalgamated Single-File Version

A Qt-integrated task worker framework for IDA Pro and standalone Python applications.
Combines: {modules}

Usage:
    from ida_taskr_amalgamated import (
        TaskRunner, WorkerLauncher, WorkerBase, ThreadExecutor,
        ProcessPoolExecutor, InterpreterPoolExecutor, ...
    )
"""

from __future__ import annotations

# =============================================================================
# CONSOLIDATED IMPORTS
# =============================================================================
import asyncio
import atexit
import collections
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import math
import multiprocessing
import multiprocessing.connection
import multiprocessing.shared_memory
import os
import pathlib
import pickle
import select
import stat
import sys
import threading
import time
import typing
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

'''

# Exports to include at the end
EXPORTS = [
    # Helpers
    "is_ida",
    "get_logger",
    "configure_logging",
    "MultiprocessingHelper",
    # Utils
    "humanize_bytes",
    "emit",
    "reify",
    "EventEmitter",
    "AsyncEventEmitter",
    "log_execution_time",
    "Range",
    "IntervalSet",
    "resolve_overlaps",
    "PatchManager",
    "DeferredPatchOp",
    "make_chunks",
    "shm_buffer",
    "execute_chunk_with_shm_view",
    "DataProcessorCore",
    # Protocols
    "WorkerProtocol",
    "MessageEmitter",
    # Qt compatibility
    "QtCore",
    "Signal",
    "Slot",
    "QT_API",
    "QT_VERSION",
    "QT_AVAILABLE",
    "QProcessEnvironment",
    "get_qt_api",
    "get_qt_version",
    "QT_ASYNCIO_AVAILABLE",
    # QtAsyncio
    "QAsyncioEventLoop",
    "QAsyncioEventLoopPolicy",
    "set_event_loop_policy",
    "run",
    "Task",
    "FutureWatcher",
    "ThreadExecutor",
    "QThreadPoolExecutor",
    "ThreadPoolExecutorSignals",
    "ProcessPoolExecutor",
    "QProcessPoolExecutor",
    "ProcessPoolExecutorSignals",
    "InterpreterPoolExecutor",
    "QInterpreterPoolExecutor",
    "InterpreterPoolExecutorSignals",
    "INTERPRETER_POOL_AVAILABLE",
    # Worker
    "ConnectionContext",
    "WorkerController",
    "WorkerBase",
    "QTASYNCIO_ENABLED",
    # Launcher
    "TemporarilyDisableNotifier",
    "ConnectionReader",
    "QtListener",
    "WorkerLauncher",
    # Task Runner
    "TaskRunner",
]


def remove_module_docstring(source: str) -> str:
    """Remove the module-level docstring from source code."""
    lines = source.split('\n')
    result_lines = []
    i = 0

    # Skip leading whitespace/blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines):
        return source

    # Check for module docstring
    first_line = lines[i].strip()
    if first_line.startswith('"""') or first_line.startswith("'''"):
        quote = first_line[:3]
        # Check if single-line docstring
        if first_line.count(quote) >= 2:
            i += 1
        else:
            # Multi-line docstring - skip until closing quote
            i += 1
            while i < len(lines) and quote not in lines[i]:
                i += 1
            i += 1  # Skip the closing quote line

    # Return rest of the file
    return '\n'.join(lines[i:])


def remove_imports(source: str) -> str:
    """Remove top-level import statements from source code.

    Keeps imports inside try/except blocks (conditional imports) and
    imports inside functions/classes.
    """
    lines = source.split('\n')
    result_lines = []
    i = 0
    indent_stack = []  # Track indentation to detect top-level vs nested

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Calculate indentation level
        if stripped:
            indent = len(line) - len(line.lstrip())
        else:
            indent = 0

        # Check for import statements
        if stripped.startswith('from __future__'):
            # Always skip future imports
            i += 1
            continue
        elif stripped.startswith('import ') or stripped.startswith('from '):
            # Only remove top-level imports (no indentation)
            # Keep conditional imports (inside try/except) and nested imports
            if indent == 0:
                # Skip ida_taskr internal imports at top level
                if 'ida_taskr' in stripped:
                    i += 1
                    continue
                # Check for multi-line import
                if '(' in stripped and ')' not in stripped:
                    # Multi-line import - skip until closing paren
                    i += 1
                    while i < len(lines) and ')' not in lines[i]:
                        i += 1
                i += 1
                continue
            # Keep indented imports (they're conditional or nested)

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines)


def process_module(module_path: Path) -> str:
    """Process a single module file, removing imports and module docstring."""
    source = module_path.read_text()

    # Remove module docstring
    source = remove_module_docstring(source)

    # Remove imports
    source = remove_imports(source)

    # Remove excessive leading blank lines but keep structure
    while source.startswith('\n\n\n'):
        source = source[1:]

    return source


def make_section(name: str, content: str) -> str:
    """Create a section with header."""
    header = f"""
# =============================================================================
# {name.upper()} MODULE
# =============================================================================
"""
    return header + content


def amalgamate(src_dir: Path) -> str:
    """Generate the amalgamated source."""
    parts = [HEADER.format(modules=", ".join(MODULES))]

    # Process each module
    for module_name in MODULES:
        module_path = src_dir / f"{module_name}.py"
        if not module_path.exists():
            print(f"Warning: {module_path} not found, skipping", file=sys.stderr)
            continue

        content = process_module(module_path)
        if content.strip():
            section_name = module_name.replace('_', ' ')
            parts.append(make_section(section_name, content))

    # Exports
    exports_section = """
# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
"""
    for export in EXPORTS:
        exports_section += f'    "{export}",\n'
    exports_section += "]\n"
    parts.append(exports_section)

    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate amalgamated ida-taskr")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("ida_taskr_amalgamated.py"),
        help="Output file path (default: ida_taskr_amalgamated.py)"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Source directory (default: auto-detect)"
    )
    args = parser.parse_args()

    # Find source directory
    if args.src_dir:
        src_dir = args.src_dir
    else:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent / "src" / "ida_taskr",
            script_dir / "src" / "ida_taskr",
            Path("src") / "ida_taskr",
        ]
        src_dir = None
        for candidate in candidates:
            if candidate.exists() and (candidate / "helpers.py").exists():
                src_dir = candidate
                break

        if not src_dir:
            print("Error: Could not find ida_taskr source directory", file=sys.stderr)
            sys.exit(1)

    print(f"Source directory: {src_dir}", file=sys.stderr)
    print(f"Output file: {args.output}", file=sys.stderr)

    result = amalgamate(src_dir)
    args.output.write_text(result)

    line_count = result.count('\n') + 1
    print(f"Generated {args.output} ({len(result)} bytes, {line_count} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
