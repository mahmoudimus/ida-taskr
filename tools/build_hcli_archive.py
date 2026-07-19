#!/usr/bin/env python3
"""Build the source-only archive consumed by HCLI and Plugin Manager."""

from __future__ import annotations

import argparse
import pathlib
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_FILES = ("ida-plugin.json", "entry_stub.py", "LICENSE", "README.md")


def build_hcli_archive(output: pathlib.Path) -> pathlib.Path:
    """Write the minimal plugin archive to ``output`` and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in PLUGIN_FILES:
            archive.write(ROOT / name, name)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    print(build_hcli_archive(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
