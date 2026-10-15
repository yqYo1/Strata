#!/usr/bin/env python3
"""Emit JSON cases only. Main launches each arm with its owned serial controller."""
import json
import sys
full = len(sys.argv) > 1 and sys.argv[1] == 'full'
cases = []
for t in [18, 22, 21, 23, 20, 42]:
    for m in [1, 3, 4, 8, 9]:
        cases.append(dict(type=t,role='gu' if t in [18,22,21,23] else 'down',m=m,n=32,k=256))
if full:
    for t in [18, 22, 21, 23, 20, 42]:
        for m in [4, 8, 16, 32, 64, 128, 1280]:
            gu = t in [18,22,21,23]
            cases.append(dict(type=t,role='gu' if gu else 'down',m=m,n=1280 if gu else 2560,k=2560 if gu else 640))
print(json.dumps(dict(status='NOT_RUN',cases=cases),indent=2))
