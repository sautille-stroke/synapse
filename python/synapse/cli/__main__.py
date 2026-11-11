"""Main CLI entrypoint for ``python -m synapse.cli``."""

import sys

from synapse.cli import main_synapse

if __name__ == "__main__":
    sys.exit(main_synapse())
