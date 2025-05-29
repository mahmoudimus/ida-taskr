#!/bin/bash
# Test runner script for anti-deobfuscation tests
# 
# This script sets up the necessary environment variables to run the tests
# with IDA Pro's PyQt5 framework on macOS.

# Set up the framework path for IDA Pro's PyQt5
export DYLD_FALLBACK_FRAMEWORK_PATH="/Applications/IDA Professional 9.1.app/Contents/Frameworks"

echo "🔬 Anti-Deobfuscation Test Runner"
echo "================================="
echo

# Check if TEST_ROUTINE_ADDR is set
if [ -n "$TEST_ROUTINE_ADDR" ]; then
    echo "🎯 Running tests for specific address: $TEST_ROUTINE_ADDR"
    echo "   Command: python -m unittest tests.test_anti_deob -v"
else
    echo "🔄 Running tests for all routine addresses"
    echo "   To run tests for a specific address, set TEST_ROUTINE_ADDR:"
    echo "   export TEST_ROUTINE_ADDR=0x141887cbd  # hex format"
    echo "   export TEST_ROUTINE_ADDR=5394431165   # decimal format"
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

# Run the tests
python -m unittest tests.test_anti_deob -v

echo
echo "✅ Tests completed!" 