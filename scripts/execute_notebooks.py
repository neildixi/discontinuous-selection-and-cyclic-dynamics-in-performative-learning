#!/usr/bin/env python
"""Execute all notebooks in notebooks/ from clean kernels."""

from __future__ import annotations

from dynafront.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["notebooks"]))
