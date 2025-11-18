#!/bin/bash
# Generate a simple test binary for IDA Taskr integration tests

set -e

# Output directory
OUTPUT_DIR="tests/_resources/bin"
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for source files
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Generating test binary..."

# Create a simple C program
cat > "$TMP_DIR/test_program.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple arithmetic functions
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

// String manipulation function
char* reverse_string(const char* str) {
    int len = strlen(str);
    char* result = (char*)malloc(len + 1);

    for (int i = 0; i < len; i++) {
        result[i] = str[len - 1 - i];
    }
    result[len] = '\0';

    return result;
}

// Fibonacci function (recursive)
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Factorial function (iterative)
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Array sum function
int sum_array(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Main function
int main(int argc, char* argv[]) {
    printf("IDA Taskr Test Binary\n");
    printf("=====================\n\n");

    // Test arithmetic functions
    int a = 10, b = 5;
    printf("Addition: %d + %d = %d\n", a, b, add(a, b));
    printf("Subtraction: %d - %d = %d\n", a, b, subtract(a, b));
    printf("Multiplication: %d * %d = %d\n", a, b, multiply(a, b));
    printf("Division: %d / %d = %d\n", a, b, divide(a, b));

    // Test string reversal
    const char* test_str = "Hello, IDA!";
    char* reversed = reverse_string(test_str);
    printf("\nOriginal: %s\n", test_str);
    printf("Reversed: %s\n", reversed);
    free(reversed);

    // Test fibonacci
    int fib_n = 10;
    printf("\nFibonacci(%d) = %d\n", fib_n, fibonacci(fib_n));

    // Test factorial
    int fact_n = 5;
    printf("Factorial(%d) = %d\n", fact_n, factorial(fact_n));

    // Test array sum
    int numbers[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int array_size = sizeof(numbers) / sizeof(numbers[0]);
    printf("\nSum of array = %d\n", sum_array(numbers, array_size));

    return 0;
}
EOF

# Detect platform and compile accordingly
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - compile ELF binary
    echo "Compiling for Linux (ELF)..."
    gcc -o "$OUTPUT_DIR/test_binary.elf" "$TMP_DIR/test_program.c" -O1
    strip "$OUTPUT_DIR/test_binary.elf"
    echo "✓ Created: $OUTPUT_DIR/test_binary.elf"

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - compile Mach-O binary
    echo "Compiling for macOS (Mach-O)..."
    gcc -o "$OUTPUT_DIR/test_binary" "$TMP_DIR/test_program.c" -O1
    strip "$OUTPUT_DIR/test_binary"
    echo "✓ Created: $OUTPUT_DIR/test_binary"

    # Also try to compile for Linux if cross-compiler available
    if command -v x86_64-linux-gnu-gcc &> /dev/null; then
        echo "Cross-compiling for Linux (ELF)..."
        x86_64-linux-gnu-gcc -o "$OUTPUT_DIR/test_binary.elf" "$TMP_DIR/test_program.c" -O1
        x86_64-linux-gnu-strip "$OUTPUT_DIR/test_binary.elf"
        echo "✓ Created: $OUTPUT_DIR/test_binary.elf"
    fi
fi

# Try to compile Windows binary if mingw is available
if command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo "Cross-compiling for Windows (PE)..."
    x86_64-w64-mingw32-gcc -o "$OUTPUT_DIR/test_binary.exe" "$TMP_DIR/test_program.c" -O1
    x86_64-w64-mingw32-strip "$OUTPUT_DIR/test_binary.exe"
    echo "✓ Created: $OUTPUT_DIR/test_binary.exe"
fi

# List created binaries
echo ""
echo "Test binaries created in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Done! Test binaries are ready for integration tests."
