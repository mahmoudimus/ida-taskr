# ida-taskr

![CI Status](https://github.com/mahmoudimus/ida-taskr/actions/workflows/python.yml/badge.svg)

## Overview

IDA Taskr is a pure Python library for IDA Pro related parallel computing. It lets you use the power of Qt (built-in to IDA!) and Python's powerful multiprocessing and asyncio systems to quickly process computationally intensive tasks without freezing IDA Pro. Oh, and it's super fast too.

## Testing 🧪

`ida-taskr` is thoroughly tested to ensure reliability.

### Unit Tests

Run unit tests locally:

```bash
# Run all unit tests
python3 -m unittest discover -s tests/

# Or use the test runner script
./run_tests.sh
```

### Integration Tests

Integration tests verify IDA Taskr works with real IDA Pro installations, supporting:
- **IDA Pro 9.1** with PyQt5 ✅
- **IDA Pro 9.2** with PySide6 ✅

Run integration tests using Docker:

```bash
# Run tests for IDA 9.1 (PyQt5)
docker compose run --rm idapro-tests

# Run tests for IDA 9.2 (PySide6)
docker compose run --rm idapro-tests-9.2
```

For more details, see [Integration Test Documentation](tests/integration/README.md).

You'll see detailed output confirming the functionality of each component. ✅

## Contributing 🤝

We welcome contributions to `ida-taskr`! Whether it's bug fixes, new features, or documentation improvements, your help is appreciated. Here's how to contribute:

1. **Fork the Repository** and clone it locally. 🍴
2. **Make Your Changes** in a new branch. 🌿
3. **Run Tests** to ensure everything works (`python3 -m unittest discover -s tests/`). 🧪
4. **Submit a Pull Request** with a clear description of your changes. 📬

Please follow the coding style and include tests for new functionality. Let's make `ida-taskr` even better together! 💪

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 📄

## Contact 📧

Have questions, suggestions, or need support? Open an issue on GitHub or reach out to [mahmoudimus](https://github.com/mahmoudimus). I'm happy to help! 😊
