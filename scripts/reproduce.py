#!/usr/bin/env python
"""Thin wrapper around ``python -m dynafront.cli reproduce``."""

from __future__ import annotations

import sys

from dynafront.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce", *sys.argv[1:]]))
