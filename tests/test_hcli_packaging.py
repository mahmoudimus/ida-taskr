"""Regression tests for Taskr's HCLI source-only plugin archive."""

from __future__ import annotations

import ast
import json
import pathlib
import runpy
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "ida_taskr" / "__init__.py"
MANIFEST = ROOT / "ida-plugin.json"
HCLI_CATEGORIES = {
    "api-scripting-and-automation",
    "collaboration-and-productivity",
    "debugging-and-tracing",
    "decompilation",
    "deobfuscation",
    "disassembly-and-processor-modules",
    "file-parsers-and-loaders",
    "integration-with-third-parties-interoperability",
    "malware-analysis",
    "other",
    "ui-ux-and-visualization",
    "vulnerability-research-and-exploit-development",
}


def _package_version() -> str:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"could not find __version__ in {INIT}")


def _build_hcli_archive(output: pathlib.Path) -> pathlib.Path:
    namespace = runpy.run_path(str(ROOT / "tools" / "build_hcli_archive.py"))
    return namespace["build_hcli_archive"](output)


def test_manifest_uses_the_plugin_stub_and_exact_distribution() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["plugin"]["entryPoint"] == "entry_stub.py"
    assert manifest["plugin"]["pythonDependencies"] == [
        f"ida-taskr=={_package_version()}"
    ]


def test_entry_stub_loads_the_real_plugin_entry() -> None:
    assert (ROOT / "entry_stub.py").read_text(encoding="utf-8") == (
        '"""Load Taskr after HCLI installs its Python distribution."""\n\n'
        "from ida_taskr.taskr_plugin import PLUGIN_ENTRY\n"
    )


def test_manifest_uses_hcli_categories() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert set(manifest["plugin"]["categories"]) <= HCLI_CATEGORIES


def test_hcli_archive_contains_only_runtime_plugin_files(tmp_path: pathlib.Path) -> None:
    archive = _build_hcli_archive(tmp_path / "ida-taskr.zip")

    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == {
            "ida-plugin.json",
            "entry_stub.py",
            "LICENSE",
            "README.md",
        }
