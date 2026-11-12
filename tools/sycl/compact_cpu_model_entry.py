#!/usr/bin/env python3
"""Owned cache-admission and model-handoff diagnostic,not clean speed evidence.

Main-authored adapter to the maintained native model controller;repo LICENSE.
Native environments are fixed before exec. Keep physical-model/task gates open.
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
ROOT, KEY = B / 'compact-cpu-model-entry-v1', 'compact_cpu_model_entry_v1'
THIS = 'tools/sycl/compact_cpu_model_entry.py'
require, sha = device.require, device.sha
MIN_AVAILABLE = 77 << 30
original_route = previous.route


def memory_admission():
    values = {s.split(':', 1)[0]: int(s.split()[1]) * 1024
        for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')}
    require(values['MemAvailable'] >= MIN_AVAILABLE,
        'model/cache diagnostic requires77GiB currently available;no native launch')


def route(text):
    value = original_route(text)
    ready = re.findall(r'^CPU_COMPACT_GU_READY phase=decode nt=1 max_jobs=6 '
        r'entries=(\d+) known_bytes=(\d+)$', text, re.M)
    prepared = re.findall(r'^CPU_COMPACT_GU_PREPARED .* setup_s=([0-9.]+)$', text, re.M)
    require(len(ready) == len(prepared) == 1 and int(ready[0][0]) == 8704
        and 0 < int(ready[0][1]) <= 16384 << 20, 'actual complete CPUcache admission required')
    layers = {0, 7, 12, 14, 16, 18, 20, 22, 30, 31, 33, 37, 38, 41, 42, 43, 46}
    calls = [(int(l), int(n)) for l, n in re.findall(
        r'^pool trace: layer (\d+) run (\d+)$', text, re.M)]
    eligible = sum(l in layers and n in (3, 6) for l, n in calls)
    require(eligible > 0, 'diagnostic must encounter selected batch shapes')
    value['CPU_compact_admission'] = dict(entries=int(ready[0][0]),
        known_bytes=int(ready[0][1]), setup_s=float(prepared[0]),
        observed_supported_layer_3or6_calls=eligible,
        scope='Prepared ownership and possible batch exposure,not per-job NT1/kernel-hit proof. '
              'No clean-speed,task,whole physical-context or model adoption claim.')
    return value


# Reuse the tested protocol/state readers with explicit new build/controller
# scope. Python itself does not initialize the native runtime/provider.
device.context = device.inherited.context
previous.device = device
previous.ROOT, previous.KEY, previous.THIS = ROOT, KEY, THIS
previous.route = route


def prepare():
    memory_admission()
    previous.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_CPU_COMPACT_GU_MIB='16384',
        STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
        STRATA_POOL_TRACE='1', STRATA_CPU_WORK_CENSUS='1')
    row['budgets']['rss_cap'] = 80 << 30
    row.pop('previous_preparation', None)
    row.pop('previous_control_failure', None)
    row['plan'] = ('Candidate-only functional diagnostic with16GiB optional CPUcache budget. '
        'Preserve actual32769/82/32769thinking inputs,32outputs each,spec4,5workers+host, '
        'static1256slots,8192prefill and logical262144/resident20480INT8KV. '
        'Require complete8704-entry cache preparation and at least one supported-layer3/6job '
        'pool call,plus all maintained route/state/MTP restore,asset,pre-main,normal-exit '
        'and fault checks. Pool trace/CPUwork/audit are charged diagnostics,not speed. '
        'The previous model peak was52.016GiB;allow80GiB RSS with77GiB MemAvailable admission. '
        'After this,collect explicit cached-math coverage and prospectively matched cache0/on '
        'long/agent and thinking long-generation evidence before model adoption.')
    row['limits'] += (' Complete allocation and pool-shape exposure do not alone prove actual '
        'per-job cached arithmetic;that coverage remains an adoption prerequisite. '
        'No sustained TG or physical262144model fill is qualified here.')
    row['retention'] = ('Keep compact admission/shape/setup/RSS/state/fault outcome for the '
        'imminent full-model comparison. Retire normal traces/cache after closure; '
        'preserve selected14508engine and current89be7candidate until comparison closes.')
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
