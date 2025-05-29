"""Helper utilities for the worker manager."""

import functools
import logging
import multiprocessing
import os
import pathlib
import stat
import sys


def is_ida():
    """
    Crude check to see if running inside IDA.
    Returns True if running inside IDA Pro, else False.

    we do not use this check because there's a possibility that
    we're running inside a headless IDA mode and that uses an
    `idapro` library which will make this succeed.

    Even though that is technically "running" in IDA, we can
    still use the python interpreter that's executing the script.
    The reason we want to know if we're in the IDA application is
    because we want to find the python interpreter that IDA is using
    to execute the script and to not start a new IDA instance.

    Maybe this function can be called `is_ida_application` or something.

        try:
            import idaapi  # noqa
            return True
        except ImportError:
            return False
    """
    exec_name = pathlib.Path(sys.executable).name.lower()
    return exec_name.startswith(("ida", "idat", "idaw", "idag"))


# on windows, we need to set the encoding to utf-8 because it defaults to cp1252
# which does not support the emoji characters used in the logging
# or really any non-ascii characters
if not is_ida():
    # this works in non IDA and for python 3.7+
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
else:
    # IDA wraps sys.stdout and does not expose the `reconfigure` method
    # so we need to set the encoding manually
    sys.stdout.encoding = "utf-8"  # type: ignore


def configure_logging(
    log,
    level=logging.INFO,
    handler_filters=None,
    fmt_str="[%(name)s:%(levelname)s:%(process)d:%(threadName)s] @ %(asctime)s %(message)s",
):
    """Configure logging with proper formatting and filters."""
    log.propagate = False
    log.setLevel(level)
    formatter = logging.Formatter(fmt_str)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    if handler_filters is not None:
        for _filter in handler_filters:
            handler.addFilter(_filter)

    for handler in log.handlers[:]:
        log.removeHandler(handler)
        handler.close()

    if not log.handlers:
        log.addHandler(handler)


def get_logger(name=None, configurer=None, log_level=logging.INFO, custom_logger=None):
    """Get a configured logger instance."""
    if custom_logger:
        return custom_logger
    if not configurer:
        configurer = functools.partial(configure_logging, level=log_level)
    name = name or f"{"ida." if is_ida() else "worker."}{__name__}"
    logger = logging.getLogger(name)
    configurer(logger)
    return logger


class MultiprocessingHelper:
    """
    Static helper class for multiprocessing context and Python interpreter discovery.
    """

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get_python_interpreter():
        """
        Gets the path to a suitable Python interpreter.
        Ensures we find a standalone Python executable.
        >>> import pathlib, sys, stat
        >>> interp: pathlib.Path = MultiprocessingHelper.get_python_interpreter()
        ...
        >>>
        """
        logger = get_logger()

        if (
            hasattr(sys, "_base_executable")  # type: ignore
            and sys._base_executable  # type: ignore
            and "python" in pathlib.Path(sys._base_executable).name.lower()  # type: ignore
        ):
            logger.debug(f"Using _base_executable: {sys._base_executable}")  # type: ignore
            return pathlib.Path(sys._base_executable)  # type: ignore

        base_paths = [
            sys.prefix,
            sys.exec_prefix,
            sys.executable,
        ]
        exe_suffix = ".exe" if os.name == "nt" else ""
        python_name = f"python{exe_suffix}"

        def base_dirs():
            for dirname in map(pathlib.Path, base_paths):
                yield dirname
                yield dirname.parent
                yield dirname.parent.parent

        for dirname in base_dirs():
            for basename in ["", "bin", "python"]:
                interp_path = dirname / basename / python_name
                if not interp_path.exists():
                    continue

                if not (interp_path.is_file() or interp_path.is_symlink()):
                    continue

                st_mode = interp_path.stat().st_mode
                if not st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                    continue

                logger.debug(f"Found Python interpreter at: {interp_path}")
                return interp_path

        logger.warning(
            "Could not determine Python interpreter path, falling back to 'python' in PATH."
        )
        return pathlib.Path("python")

    @staticmethod
    def set_multiprocessing_context():
        """
        Sets up the multiprocessing context to use 'spawn' and sets the Python executable.
        """
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        multiprocessing.set_executable(
            str(MultiprocessingHelper.get_python_interpreter())
        )


# Initialize multiprocessing context
MultiprocessingHelper.set_multiprocessing_context()
