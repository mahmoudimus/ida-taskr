"""Keep Taskr's package and HCLI metadata on one version."""

from __future__ import annotations

import ast
import json
import pathlib
import re
import runpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "ida_taskr" / "__init__.py"
MANIFEST = ROOT / "ida-plugin.json"
PYPROJECT = ROOT / "pyproject.toml"


def _package_version() -> str:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"could not find __version__ in {INIT}")


def _pyproject_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"$', PYPROJECT.read_text(encoding="utf-8"), re.M
    )
    assert match is not None
    return match.group(1)


def test_all_published_versions_match() -> None:
    version = _package_version()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert _pyproject_version() == version
    assert manifest["plugin"]["version"] == version
    assert manifest["plugin"]["pythonDependencies"] == [f"ida-taskr=={version}"]


def test_sync_tool_checks_all_published_versions() -> None:
    namespace = runpy.run_path(str(ROOT / "tools" / "sync_plugin_version.py"))

    assert namespace["main"](["--check"]) == 0
