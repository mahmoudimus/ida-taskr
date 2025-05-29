#!/bin/bash
# Test runner script for anti-deobfuscation tests
# 
# This script sets up the necessary environment variables to run the tests
# with IDA Pro's PyQt5 framework on macOS.
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh test_event_emitter # Run specific test file
#   ./run_tests.sh TestMessageEmitter # Run specific test class
#   ./run_tests.sh --help            # Show this help message

# Set up the framework path for IDA Pro's PyQt5
export DYLD_FALLBACK_FRAMEWORK_PATH="/Applications/IDA Professional 9.1.app/Contents/Frameworks"

# Function to show help
show_help() {
    echo "🔬 Anti-Deobfuscation Test Runner"
    echo "================================="
    echo
    echo "📖 Usage:"
    echo "   ./run_tests.sh                    # Run all tests"
    echo "   ./run_tests.sh test_event_emitter # Run specific test file"
    echo "   ./run_tests.sh TestMessageEmitter # Run specific test class" 
    echo "   ./run_tests.sh --help            # Show this help message"
    echo
    echo "🎯 For anti-deobfuscation tests, you can also set:"
    echo "   export TEST_ROUTINE_ADDR=0x141887cbd  # hex format"
    echo "   export TEST_ROUTINE_ADDR=5394431165   # decimal format"
    echo
    echo "📋 Available test files:"
    echo "   test_anti_deob        # Anti-deobfuscation algorithm tests"
    echo "   test_event_emitter    # MessageEmitter functionality tests"
    echo "   test_task_runner      # TaskRunner functionality tests"
    echo
    echo "📋 Available routine addresses:"
    echo "   5368727905 (0x140004961) - 110 bytes"
    echo "   5368729905 (0x140005131) - 160 bytes"
    echo "   5369441646 (0x1400B2D6E) - 160 bytes"
    echo "   5376900737 (0x1407CFE81) - 160 bytes"
    echo "   5376975793 (0x1407E23B1) - 160 bytes"
    echo "   5390373031 (0x1414A90A7) - 160 bytes"
    echo "   5394431165 (0x141887CBD) - 1325 bytes"
}

echo "🔬 Anti-Deobfuscation Test Runner"
echo "================================="
echo

# Get the test name from command line argument
TEST_NAME="$1"

# Handle help requests
case "$TEST_NAME" in
    --help|-h|help)
        show_help
        exit 0
        ;;
esac

# Check if a specific test is requested
if [ -n "$TEST_NAME" ]; then
    echo "🎯 Running specific test: $TEST_NAME"
    echo "   Command: python -m unittest tests.$TEST_NAME -v"
    TEST_COMMAND="python -m unittest tests.$TEST_NAME -v"
else
    # Check if TEST_ROUTINE_ADDR is set for anti-deob tests
    if [ -n "$TEST_ROUTINE_ADDR" ]; then
        echo "🎯 Running tests for specific address: $TEST_ROUTINE_ADDR"
        echo "   Command: python -m unittest tests.test_anti_deob -v"
        TEST_COMMAND="python -m unittest tests.test_anti_deob -v"
    else
        echo "🔄 Running all tests"
        echo "   To run tests for a specific address, set TEST_ROUTINE_ADDR:"
        echo "   export TEST_ROUTINE_ADDR=0x141887cbd  # hex format"
        echo "   export TEST_ROUTINE_ADDR=5394431165   # decimal format"
        echo
        echo "   To run a specific test file:"
        echo "   ./run_tests.sh test_event_emitter"
        echo "   ./run_tests.sh test_task_runner"
        echo
        echo "   For help: ./run_tests.sh --help"
        TEST_COMMAND="python -m unittest discover tests"
    fi
fi

echo
echo "📋 Available routine addresses:"
echo "   5368727905 (0x140004961) - 110 bytes"
echo "   5368729905 (0x140005131) - 160 bytes"
echo "   5369441646 (0x1400B2D6E) - 160 bytes"
echo "   5376900737 (0x1407CFE81) - 160 bytes"
echo "   5376975793 (0x1407E23B1) - 160 bytes"
echo "   5390373031 (0x1414A90A7) - 160 bytes"
echo "   5394431165 (0x141887CBD) - 1325 bytes"
echo

echo "🚀 Starting tests..."
echo "===================="

# Run the tests and capture the exit code
$TEST_COMMAND
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Tests completed successfully!"
else
    echo "❌ Tests failed with exit code: $EXIT_CODE"
    if [ -n "$TEST_NAME" ]; then
        echo
        echo "💡 Tip: Make sure the test name is correct. Available tests:"
        echo "   test_anti_deob, test_event_emitter, test_task_runner"
        echo "   Or run './run_tests.sh --help' for more information."
    fi
fi

exit $EXIT_CODE 