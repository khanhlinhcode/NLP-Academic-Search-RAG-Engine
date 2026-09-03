"""Stable Streamlit entry point for local and container execution."""

import sys
from importlib import import_module
from pathlib import Path

# Streamlit Community Cloud can install the lightweight requirements next to
# this entrypoint without installing the backend's ML dependency set.
source_root = Path(__file__).resolve().parents[1] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

main = import_module("nlp_academic_search.ui.app").main

if __name__ == "__main__":
    main()
