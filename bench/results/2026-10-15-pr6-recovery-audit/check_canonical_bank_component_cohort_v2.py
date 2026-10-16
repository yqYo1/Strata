#!/usr/bin/env python3
"""Main ordered corrected pack64 component gates; stop on first failed arm."""
from check_canonical_bank_component_v2 import MODES, main

for mode in MODES:
    main(mode)
