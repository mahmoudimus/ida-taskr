#!/usr/bin/env python3
"""Keep Taskr package and HCLI versions synchronized without importing Qt."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "ida_taskr" / "__init__.py"
MANIFEST = ROOT / "ida-plugin.json"
PYPROJECT = ROOT / "pyproject.toml"


def package_version() -> str:
    """Read the literal package version without importing IDA or Qt modules."""
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"could not find __version__ in {INIT}")


def metadata_matches(version: str) -> bool:
    """Return whether package metadata and HCLI requirements use ``version``."""
    project = PYPROJECT.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    project_version = re.search(r'^version\s*=\s*"([^"]+)"$', project, re.M)
    return bool(
        project_version
        and project_version.group(1) == version
        and manifest["plugin"]["version"] == version
        and manifest["plugin"].get("pythonDependencies")
        == [f"ida-taskr=={version}"]
    )


def sync(version: str) -> None:
    """Write the package version into pyproject and HCLI manifest metadata."""
    project = PYPROJECT.read_text(encoding="utf-8")
    project, count = re.subn(
        r'^(version\s*=\s*")[^"]+("\s*)$',
        r"\g<1>" + version + r"\g<2>",
        project,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit("could not locate project version in pyproject.toml")
    PYPROJECT.write_text(project, encoding="utf-8")

    requirement = f"ida-taskr=={version}"
    manifest = MANIFEST.read_text(encoding="utf-8")
    manifest, count = re.subn(
        r'("version"\s*:\s*")[^"]*(")',
        r"\g<1>" + version + r"\g<2>",
        manifest,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not locate manifest version")
    manifest, count = re.subn(
        r'("pythonDependencies"\s*:\s*\[\s*")[^"]*(")',
        r"\g<1>" + requirement + r"\g<2>",
        manifest,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not locate manifest pythonDependencies")
    MANIFEST.write_text(manifest, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    version = package_version()

    if metadata_matches(version):
        return 0
    if args.check:
        print(
            "pyproject.toml or ida-plugin.json does not match "
            f"ida_taskr.__version__ {version!r}; run: "
            "python tools/sync_plugin_version.py",
            file=sys.stderr,
        )
        return 1
    sync(version)
    print(f"synced package and HCLI metadata to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
