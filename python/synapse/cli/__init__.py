"""CLI entry points for the Synapse suite.

Provides two top-level entry points registered in ``pyproject.toml``:

- ``synapse``       -> :func:`main_synapse`
- ``synapse-bench`` -> :func:`main_bench`

Both support a ``--json`` global flag that switches all stdout output
to machine-readable JSON (no mixed log noise), and propagate W3C
``traceparent`` from the environment for distributed tracing.
"""

from synapse.cli._app import main_bench, main_synapse

# Legacy compat: ``python -m synapse.cli`` still works
main = main_synapse

__all__ = ["main_synapse", "main_bench", "main"]
