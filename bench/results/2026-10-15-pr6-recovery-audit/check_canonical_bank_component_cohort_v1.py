#!/usr/bin/env python3
"""Main-only ordered component qualification. Stop on the first failed arm."""
from check_canonical_bank_component_v1 import MODES, main

for mode in MODES:
    main(mode)
