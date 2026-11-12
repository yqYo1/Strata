#!/usr/bin/env python3
"""Qualify the coherent full engine with optional CPU caching disabled.

Main-authored adapter; repository LICENSE. Preserve the failed cached screen.
This admits state handoff,not clean performance or full physical context.
"""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_compact_cpu_model as device
import coherent_prefill_model as previous

P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B / 'compact-cpu-cache0-model-v1', 'compact_cpu_cache0_model_v1'
THIS = 'tools/sycl/compact_cpu_cache0_model.py'
OLD_OBSERVER = 'tools/sycl/compact_cpu_model_entry.py'
require, sha = device.require, device.sha
original_native, original_route = previous.native, previous.route


def native(expected=None, *, observers=()):
    return original_native(expected, observers={OLD_OBSERVER, *observers})


def route(text):
    value = original_route(text)
    require(not re.search(r'^CPU_COMPACT_GU_(BEGIN|PREPARED|READY)\b', text, re.M),
        'cache0 must not allocate or bind the optional CPUcache')
    return dict(value, optional_CPUcache_disabled=True)


def memory_admission():
    available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
        if s.startswith('MemAvailable:'))
    require(available >= 60 << 30, 'cache0 model requires60GiB currently available')


device.context = device.inherited.context
previous.device = device
previous.ROOT, previous.KEY, previous.THIS = ROOT, KEY, THIS
previous.native, previous.route = native, route


def prepare():
    memory_admission()
    previous.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment']['STRATA_CPU_COMPACT_GU_MIB'] = '0'
    row['budgets']['rss_cap'] = 80 << 30
    row.pop('previous_preparation', None)
    row.pop('previous_control_failure', None)
    row['plan'] = ('Independent cache0functional admission for the fresh89be7coherent engine. '
        'Keep the original32769/82/32769thinking inputs,32outputs each,spec4,5workers+host, '
        'static1256slots,8192prefill,logical262144/resident20480INT8KV and qualified runtime. '
        'Require all original route/state,MTP restore,pre-main,asset,quiet-validation,normal0 '
        'and closed-owner/no-fault gates plus absence of optional CPUcache preparation. '
        'Stop first failure;no retry or hash refresh. The prior cache-on selected-shape screen '
        'remains failed. This does not qualify cached math or performance.')
    row['retention'] = ('Keep compact functional evidence for the imminent coherent fallback '
        'migration/comparison. Retire normal logs/cache after closure. Keep14508fallback '
        'and89be7candidate pending matched speed,task and physical262144/RESTORE gates.')
    previous.context().context(row['native_observer_admission']).save(row)


if __name__ == '__main__':
    require(sys.executable == previous.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in previous.ACTIONS, 'qualified Python;prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            if sys.argv[1] == 'run':
                memory_admission()
            previous.context().execute(sys.argv[1])
