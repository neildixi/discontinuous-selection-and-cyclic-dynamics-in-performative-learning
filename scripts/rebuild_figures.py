#!/usr/bin/env python
"""Rebuild every public figure from cached summaries. No simulations."""

from __future__ import annotations

from dynafront.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["figures"]))
