#!/usr/bin/env python3
"""Private FP32-output GDN candidate on the selected native source; MIT."""
import prefill_strategy_sources as original

P, B = original.P, original.B
require = original.require
BASE = 'f6cea2f63315e2598d63cb04d5d0b6443dc21a24'
IMPLEMENTATION = 'e618f3466432dca7ee7435a399e7eb92ecfc06c3'
CHANGED = ('src/prefill/sycl/vllm_xe2/native16.hpp',
           'src/prefill/sycl/vllm_xe2/native16.cpp')


def sources():
    full, native = original.sources()
    blob = original.original.original.split.blob
    for name in CHANGED:
        require(native[name] == blob(BASE, name), 'projection source basis differs: ' + name)
        native[name] = blob(IMPLEMENTATION, name)
    require(len(native) == 555, 'unexpected FP32 projection source count')
    return full, native
