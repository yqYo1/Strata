#!/usr/bin/env python3
"""Six declared independent samples, serial and stop on first failure."""
import sys
sys.dont_write_bytecode=True
from run_native_direct_clean_v1 import main
for arm,repetition in [(0,0),(1,0),(1,1),(0,1),(0,2),(1,2)]:
    main(arm,repetition)
