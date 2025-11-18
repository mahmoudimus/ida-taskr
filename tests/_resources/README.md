# Test Resources

This directory contains test resources for IDA Taskr integration tests.

## Structure

```
_resources/
├── bin/          # Test binaries for IDA analysis
└── README.md     # This file
```

## Adding Test Binaries

To run integration tests, you need to add test binaries to the `bin/` directory.

### Requirements

Test binaries should:
- Be small (< 1MB preferred)
- Be redistributable or self-created
- Contain analyzable code (functions, data segments)
- Be in a format supported by IDA Pro (PE, ELF, Mach-O, etc.)

### Recommended Test Binaries

You can use:

1. **Simple C programs** compiled for different architectures
2. **Example binaries** from reverse engineering practice sites
3. **Custom test binaries** created specifically for testing

### Example: Creating a Simple Test Binary

#### Linux/macOS (ELF)

```bash
cat > test_program.c << 'EOF'
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main() {
    int result = add(5, 3);
    result = multiply(result, 2);
    printf("Result: %d\n", result);
    return 0;
}
EOF

gcc -o test_binary.elf test_program.c
strip test_binary.elf  # Optional: remove debug symbols
mv test_binary.elf tests/_resources/bin/
```

#### Windows (PE)

```bash
# On Windows or with mingw-w64
x86_64-w64-mingw32-gcc -o test_binary.exe test_program.c
mv test_binary.exe tests/_resources/bin/
```

### Binary Format Support

IDA Pro supports many binary formats, including:

- **Windows**: PE (.exe, .dll, .sys)
- **Linux**: ELF (.elf, .so, no extension)
- **macOS**: Mach-O (.dylib, .app, no extension)
- **Raw binary**: (.bin)
- **And many more...**

### Security Considerations

**IMPORTANT**:
- Do NOT commit malware samples to this repository
- Do NOT commit copyrighted binaries
- Do NOT commit binaries containing sensitive information
- Always scan binaries with antivirus before committing

### .gitignore

Large binary files should be added to `.gitignore` to avoid bloating the repository:

```gitignore
# Add to .gitignore if needed
tests/_resources/bin/*.exe
tests/_resources/bin/*.elf
tests/_resources/bin/*.dll
```

For CI/CD, you may want to:
1. Generate test binaries during the CI workflow
2. Download them from a separate storage location
3. Use very small, purpose-built test binaries committed to the repo

## Using Test Binaries

The integration tests automatically discover binaries in this directory using the `test_binary` fixture in `conftest.py`.

No additional configuration is needed - just add binaries here and they'll be picked up by the test suite.

## Example Binary Properties

A good test binary should have:
- At least 1-2 functions
- A few basic blocks
- Some data references
- Simple control flow (if/else, loops)
- Size: 10KB - 100KB (smaller is better)

This allows IDA to perform meaningful analysis without taking too long.
