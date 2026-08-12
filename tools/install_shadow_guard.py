#!/usr/bin/env python3
"""Thin wrapper: `python tools/install_shadow_guard.py [--status|--uninstall]`.

The real implementation is `magcalc/_shadow_guard_install.py`, so that
`magcalc guard` works for every install -- `tools/` is not shipped by a
non-editable `pip install`. Kept because it works without the CLI on PATH.
"""
from magcalc._shadow_guard_install import main

if __name__ == "__main__":
    raise SystemExit(main())
