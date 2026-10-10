#!/usr/bin/env python
"""Compare a summaries directory with results/cached/expected_results.json."""

from __future__ import annotations

import sys

from dynafront.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare", *sys.argv[1:]]))
