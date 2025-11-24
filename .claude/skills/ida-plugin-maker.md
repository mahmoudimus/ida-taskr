# IDA Pro HCLI Plugin Packaging Skill

This skill guides you through packaging an IDA Pro plugin for the HCLI (Hex-Rays Command Line Interface) plugin infrastructure.

## Context

HCLI is IDA Pro's built-in plugin manager that allows users to discover, install, and update plugins easily. Properly packaged plugins can be installed with a single command: `ida-hcli plugin install plugin-name`.

## Prerequisites to Review

Before starting, review these resources:

1. **HCLI Documentation:**
   - Plugin manager docs: `https://github.com/HexRaysSA/ida-hcli/tree/main/docs/user-guide/plugin-manager.md`

2. **Example Repositories:**
   - Pure Python library: `https://github.com/mahmoudimus/ida-taskr`
   - Pure Python plugin: `https://github.com/mahmoudimus/ida-sigmaker`
   - Native plugin with build: `https://github.com/williballenthin/zydisinfo/blob/gha-hcli/.github/workflows/build.yml`

## Step 1: Analyze the Repository

### 1.1 Read Documentation
- Read `README.md` to understand what the plugin does
- Check for existing `pyproject.toml` or `setup.py`
- Look for `LICENSE` file
- Check for existing entry points or plugin files

### 1.2 Identify Plugin Type

Determine if this is:

**A. Pure Python Plugin/Library**
- No C/C++ code
- No compilation required
- May depend on IDA's built-in packages (Qt, etc.)

**B. Native Plugin**
- Contains C/C++ source code
- Requires compilation against IDA SDK
- May need platform-specific builds

### 1.3 Gather Metadata

Extract the following information:

```markdown
**Plugin Name:** (from repo name or README)
**Version:** (from pyproject.toml, __init__.py, or use format YYYY.M.DD)
**Description:** (single concise sentence from README)
**License:** (from LICENSE file - usually MIT, GPL, Apache, etc.)
**Author(s):** (from LICENSE or README)
**Repository URL:** (GitHub URL)
**Entry Point:** (main plugin file - usually src/package/__init__.py or src/package/plugin.py)
**IDA Compatibility:** (usually >=9.0 for modern plugins)
**Python Dependencies:** (ONLY from README/pyproject.toml - do NOT infer from code)
**Settings:** (any configuration values users need to set)
```

### 1.4 Choose Categories

Pick 1-2 primary categories from:

- `disassembly-and-processor-modules` - CPU architecture support, disassembly enhancements
- `file-parsers-and-loaders` - File format parsers, custom loaders
- `decompilation` - Decompiler enhancements, optimizations
- `debugging-and-tracing` - Debugging tools, trace analysis
- `deobfuscation` - Anti-obfuscation, unpacking, demangling
- `collaboration-and-productivity` - Team features, workflow improvements
- `integration-with-third-parties-interoperability` - External tool integration
- `api-scripting-and-automation` - Scripting libraries, automation tools
- `ui-ux-and-visualization` - UI enhancements, visualization tools
- `malware-analysis` - Malware-specific analysis tools
- `vulnerability-research-and-exploit-development` - Vuln research, exploit dev
- `other` - Doesn't fit other categories

### 1.5 Select Keywords

Choose 5-10 keywords that describe:
- Technology used (e.g., "multiprocessing", "shared-memory", "Qt")
- Features (e.g., "decorators", "async", "parallel-computing")
- Use cases (e.g., "background-tasks", "task-execution")

## Step 2: Create `ida-plugin.json`

Create this file in the repository root:

```json
{
  "IDAMetadataDescriptorVersion": 1,
  "plugin": {
    "name": "plugin-name",
    "entryPoint": "src/plugin_package/__init__.py",
    "version": "0.1.0",
    "description": "Single sentence describing what the plugin does",
    "idaVersions": ">=9.0",
    "license": "MIT",
    "urls": {
      "repository": "https://github.com/username/repo"
    },
    "authors": [
      {
        "name": "Author Name",
        "email": "optional@email.com"
      }
    ],
    "pythonDependencies": [],
    "settings": [],
    "categories": [
      "api-scripting-and-automation"
    ],
    "keywords": [
      "keyword1",
      "keyword2",
      "keyword3"
    ]
  }
}
```

### Field Guidelines:

**`pythonDependencies`:**
- ONLY include dependencies explicitly mentioned in README or pyproject.toml
- Do NOT infer from import statements
- For plugins using IDA's built-in Qt, use empty array: `[]`
- Format: `["package>=version"]`
- Example with conditions: `["PyQt5>=5.15.0; ida_version < '9.2'", "PySide6>=6.0.0; ida_version >= '9.2'"]`

**`settings`:**
- Only include if plugin requires user configuration
- Format:
  ```json
  {
    "key": "api_key",
    "type": "string",
    "required": true,
    "default": "",
    "name": "API Key",
    "documentation": "Your service API key for authentication",
    "validation_pattern": "^[A-Za-z0-9_-]+$"
  }
  ```
- Omit if no settings needed: `"settings": []`

## Step 3: Create `setup.py` (Backwards Compatibility)

Create this file in the repository root:

```python
"""Setup script for backwards compatibility.

Modern packaging is defined in pyproject.toml.
This file exists for backwards compatibility with older pip versions.
"""

from setuptools import setup

# All configuration is in pyproject.toml
setup()
```

This is required even though `pyproject.toml` exists, for pip versions < 21.3.

## Step 4: Create Release Workflow

### For Pure Python Plugins

Create `.github/workflows/release.yml`:

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-plugin:
    name: Build IDA Plugin Archive
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build wheel setuptools

      - name: Build Python package
        run: python -m build

      - name: Extract version from tag
        id: get_version
        run: |
          if [[ "${GITHUB_REF}" == refs/tags/* ]]; then
            VERSION=${GITHUB_REF#refs/tags/v}
          else
            VERSION="dev-$(date +%Y%m%d)"
          fi
          echo "version=${VERSION}" >> $GITHUB_OUTPUT
          echo "Version: ${VERSION}"

      - name: Create HCLI plugin archive
        run: |
          mkdir -p PLUGIN_NAME-plugin/PACKAGE_NAME

          # Copy source files
          cp -r src/PACKAGE_NAME/* PLUGIN_NAME-plugin/PACKAGE_NAME/

          # Copy metadata and documentation
          cp ida-plugin.json PLUGIN_NAME-plugin/
          cp LICENSE PLUGIN_NAME-plugin/
          cp README.md PLUGIN_NAME-plugin/

          # Create archive
          cd PLUGIN_NAME-plugin
          zip -r ../PLUGIN_NAME-${{ steps.get_version.outputs.version }}.zip .
          cd ..

          # Also create a latest version for convenience
          cp PLUGIN_NAME-${{ steps.get_version.outputs.version }}.zip PLUGIN_NAME-latest.zip

      - name: Upload plugin artifact
        uses: actions/upload-artifact@v4
        with:
          name: PLUGIN_NAME-plugin
          path: |
            PLUGIN_NAME-*.zip
          retention-days: 30

      - name: Create GitHub Release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: |
            PLUGIN_NAME-${{ steps.get_version.outputs.version }}.zip
            PLUGIN_NAME-latest.zip
            dist/*.whl
            dist/*.tar.gz
          generate_release_notes: true
          draft: false
          prerelease: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Publish to PyPI
        if: startsWith(github.ref, 'refs/tags/')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          skip-existing: true
```

**Important:** Replace `PLUGIN_NAME` and `PACKAGE_NAME` with actual values:
- `PLUGIN_NAME`: Kebab-case plugin name (e.g., `ida-taskr`)
- `PACKAGE_NAME`: Python package name (e.g., `ida_taskr`)

### For Native Plugins

Use the pattern from `zydisinfo`:
- Build matrix for (Windows, Linux, macOS) × (IDA 9.0, 9.1, 9.2)
- Download IDA SDK using HCLI credentials
- CMake configuration and compilation
- Collect platform-specific binaries
- Create separate archives per platform

See: `https://github.com/williballenthin/zydisinfo/blob/gha-hcli/.github/workflows/build.yml`

## Step 5: Update README.md

Add installation section at the beginning of the README:

```markdown
## Installation

### Via HCLI (Recommended for IDA Pro 9.0+)

The easiest way to install PLUGIN_NAME is through IDA's built-in plugin manager:

```bash
# From IDA Pro's HCLI
ida-hcli plugin install PLUGIN_NAME
```

Or download from GitHub releases and install manually:
```bash
ida-hcli plugin install path/to/PLUGIN_NAME-X.X.X.zip
```

### Via pip

```bash
# Install from PyPI
pip install PLUGIN_NAME

# Or install from source
pip install -e .
```

### Manual Installation

Download the latest release and extract to your IDA plugins directory:
- Linux/macOS: `~/.idapro/plugins/`
- Windows: `%APPDATA%\Hex-Rays\IDA Pro\plugins\`
```

## Step 6: Verify Repository Structure

Ensure the repository follows the ida-reloader pattern:

```
plugin-name/
├── src/
│   └── package_name/       # Main package
│       ├── __init__.py     # Public API exports
│       └── *.py            # Implementation files
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests (optional)
├── examples/              # Example scripts (optional)
├── .github/
│   └── workflows/
│       ├── python.yml     # Test workflow (may exist)
│       └── release.yml    # Release workflow (NEW)
├── ida-plugin.json        # HCLI metadata (NEW)
├── pyproject.toml         # Modern Python packaging
├── setup.py               # Backwards compatibility (NEW)
├── README.md              # Documentation (UPDATED)
├── LICENSE                # License file
└── .gitignore
```

## Step 7: Commit and Create Release

### Commit Changes

```bash
git add ida-plugin.json setup.py .github/workflows/release.yml README.md
git commit -m "Add HCLI plugin packaging support

- Add ida-plugin.json with plugin metadata
- Add setup.py for pip backwards compatibility
- Add .github/workflows/release.yml for automated releases
- Update README with HCLI installation instructions

Users can now install with: ida-hcli plugin install PLUGIN_NAME"
```

### Create Release

```bash
# Tag the version
git tag v0.1.0

# Push to remote
git push origin main v0.1.0
```

The GitHub Actions workflow will automatically:
1. Build the plugin archive
2. Create a GitHub Release
3. Upload artifacts
4. Publish to PyPI (if PYPI_API_TOKEN configured)

## Common Patterns

### Pattern A: Library Plugin (No UI)

**Characteristics:**
- Entry point: `src/package/__init__.py`
- No standalone UI/hotkey
- Users import functionality: `from package import Feature`
- Categories: `api-scripting-and-automation`

**Example:** ida-taskr

### Pattern B: Standalone Plugin (With UI)

**Characteristics:**
- Entry point: `src/package/plugin.py` with `PLUGIN_ENTRY()`
- Has UI elements, hotkeys, menu items
- Users activate via hotkey or menu
- Categories: Based on functionality (e.g., `decompilation`, `malware-analysis`)

**Example:** ida-sigmaker

### Pattern C: Hybrid Plugin

**Characteristics:**
- Library + optional UI plugin
- Entry point can be either `__init__.py` or `plugin.py`
- Exports both library functions and plugin interface
- Multiple categories

## Troubleshooting

### Issue: "No module named 'package'"
**Cause:** Entry point path incorrect in `ida-plugin.json`
**Solution:** Verify path matches actual file structure: `src/package/__init__.py`

### Issue: Workflow fails to create archive
**Cause:** Package name mismatch in workflow
**Solution:** Update `PLUGIN_NAME` and `PACKAGE_NAME` in release.yml

### Issue: Dependencies not installing
**Cause:** Dependencies not listed in `ida-plugin.json`
**Solution:** Add to `pythonDependencies` array (only if documented in README)

### Issue: Plugin doesn't load in IDA
**Cause:** Entry point doesn't define required plugin interface
**Solution:**
- For standalone plugins: Ensure `PLUGIN_ENTRY()` function exists
- For library plugins: Ensure `__init__.py` exports public API

## Quick Reference Commands

```bash
# Create skills directory (if needed)
mkdir -p .claude/skills

# Create all required files
touch ida-plugin.json setup.py .github/workflows/release.yml

# Verify structure
tree -L 3 -I '__pycache__|*.pyc|.git'

# Test locally
python -m build
pip install -e .

# Create release
git tag v0.1.0
git push origin main --tags
```

## Checklist

Before creating a pull request or release, verify:

- [ ] `ida-plugin.json` created with complete metadata
- [ ] `setup.py` created for backwards compatibility
- [ ] `.github/workflows/release.yml` created
- [ ] `README.md` updated with HCLI installation section
- [ ] Repository structure matches pattern
- [ ] All paths in `ida-plugin.json` are correct
- [ ] Categories and keywords are relevant
- [ ] Version number is consistent
- [ ] License file exists and is referenced correctly
- [ ] Test workflow exists (`.github/workflows/python.yml` or similar)

## Example: Complete `ida-plugin.json` for Library Plugin

```json
{
  "IDAMetadataDescriptorVersion": 1,
  "plugin": {
    "name": "ida-taskr",
    "entryPoint": "src/ida_taskr/__init__.py",
    "version": "0.1.0",
    "description": "Pure Python library for IDA Pro parallel computing with Qt-based multiprocessing, decorators, and shared memory executors",
    "idaVersions": ">=9.0",
    "license": "MIT",
    "urls": {
      "repository": "https://github.com/mahmoudimus/ida-taskr"
    },
    "authors": [
      {
        "name": "Mahmoud Rusty Abdelkader"
      }
    ],
    "pythonDependencies": [],
    "settings": [],
    "categories": [
      "api-scripting-and-automation",
      "collaboration-and-productivity"
    ],
    "keywords": [
      "multiprocessing",
      "parallel-computing",
      "background-tasks",
      "shared-memory",
      "concurrent-futures",
      "qt-integration",
      "task-execution",
      "worker-processes",
      "decorators",
      "async"
    ]
  }
}
```

## Notes

- **Pure Python plugins** are easier to package (single platform)
- **Native plugins** require platform-specific builds (Windows/Linux/macOS)
- Always test installation locally before releasing
- PyPI publishing requires `PYPI_API_TOKEN` secret in repository settings
- HCLI will automatically handle dependency installation from `pythonDependencies`
- Keep descriptions concise (single sentence for `description` field)
- Use semantic versioning (major.minor.patch)

## Success Criteria

A successfully packaged plugin should:

1. ✅ Install with `ida-hcli plugin install plugin-name`
2. ✅ Load in IDA Pro without errors
3. ✅ Have all dependencies automatically installed
4. ✅ Be discoverable in IDA's plugin index
5. ✅ Include proper documentation in README
6. ✅ Have automated releases via GitHub Actions
7. ✅ Follow consistent naming conventions
8. ✅ Include proper licensing information

Once these criteria are met, the plugin is ready for distribution through HCLI!
