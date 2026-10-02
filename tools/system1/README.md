# System1 experiment scripts

The [root README](../../README.md) has installation and run commands for the full MNIST test set and Q*bert level one. The [Q*bert reproduction guide](../../docs/system1/qbert-reproduction.md) records model hashes, emulator settings, and trace packaging.

The active scripts use the server's [`/v1/decisions` endpoint](../../docs/system1/system1-endpoints.md). The server owns prompt caching; the Python scripts assemble requests and record results.
