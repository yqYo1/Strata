"""Verify live RESTORE on the whole-request prefill build."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_restore as restore
import dflash2_prefill_ple_horizon_context as context

bank, require, phase, grouped = context.bank, context.require, context.phase, context.grouped
basis, cache = context.basis, context.cache
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_horizon_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', 'stock', 'dedicated')
PLAN = dict(order=list(ORDER), prefix=restore.PREFIX, fresh_suffix=restore.SUFFIX,
    scope='Park 16385 tokens, switch to a 33-token conversation, restore the first conversation from '
          'RAM and append 118785 fresh tokens. Both arms group all suffix chunks on the measured build.',
    gates='Vary only the dedicated cache. Require the original snapshot, output IDs, active state, '
          'mode-16 direct stores and complete per-layer KV ledger. Verify that the cache selection '
          'is phase-local and the whole window executes once, on the restored suffix.',
    limits='Live correctness with tracing and a nonzero prefix. Compare sustained decode and '
           'agent-session performance separately.')


def prerequisites():
    native, baseline, timing_sha = context.prerequisites(None)
    physical = basis.validation.committed(bank.RESULT, (context.KEY,))
    expected = dict(ple_horizon_configured='full', ple_horizon_executed=False,
                    prefill_cache_rows=cache.capacity.LARGE_ROWS,
                    ordinary_cache_rows=cache.capacity.DEFAULT_ROWS)
    require(context.wait.retained.closed(physical, 'closed')
            and physical['timing_sha256'] == timing_sha
            and all(physical['outcomes'][name][field] == value
                    for name in context.earlier.physical.ORDER
                    for field, value in expected.items()),
            'finish the physical context check on the same whole-request configuration')
    return native, baseline, timing_sha


def environment(row, case):
    env = restore.environment(row, dict(case, name='early'))
    require(not any(name in env for name in (phase.SELECTOR, grouped.SELECTOR))
            and env.get(grouped.geometry.SELECTOR, '0') == '0',
            'unexpected PLE cache selector or record observer')
    rows = cache.capacity.LARGE_ROWS if case['name'] == 'dedicated' else 0
    return dict(env, **{phase.SELECTOR: str(rows), grouped.SELECTOR: 'all'})


def inspect(peer, text, pid, arm, requests, row):
    # Validate the shared issue marker together with the grouped route below.
    issued = restore.timing.check.ISSUED
    reference_log = ''.join(line for line in text.splitlines(keepends=True)
                            if line.rstrip('\r\n') != issued)
    result = restore.inspect(peer, reference_log, pid, 'stock', requests, row)
    dedicated = arm == 'dedicated'
    ordinary_rows = cache.capacity.DEFAULT_ROWS
    active_rows = cache.capacity.LARGE_ROWS if dedicated else ordinary_rows
    expected = f'{phase.START}long_prefill={active_rows} ordinary={ordinary_rows}'
    require([line for line in text.splitlines() if line.startswith(phase.START)] == ([expected] if dedicated else [])
            and all(marker not in text for marker in (grouped.PRODUCER, grouped.CONSUMER, phase.TICKET)),
            'actual restore cache allocation or observers differ')

    end = 0
    routes = []
    for index, state in enumerate(re.finditer(r'^strata serve: STATE_HASH L=.*$', text, re.M)):
        frame = text[end:state.start()]
        end = state.end()
        phase.check_first_cache(frame, 'large' if dedicated else 'stock', int(index != 1))
        depth = (requests[index]['read'] - 2) // peer.capacity.CHUNK + 1 if index == 2 else 0
        window = f'{grouped.MARKER}tokens_per_chunk={peer.capacity.CHUNK} window_chunks={depth}'
        require([line for line in frame.splitlines() if line.startswith(grouped.MARKER)]
                == ([window] if depth else []) and frame.count(grouped.MARKER) == int(bool(depth))
                and frame.count(issued) == int(bool(depth)),
                'only the restored suffix should take the whole-request window')
        routes.append(depth)
    require(all(marker not in text[end:] for marker in (phase.FIRST, grouped.MARKER, issued)),
            'an unchecked PLE route appeared after the final request')
    return dict(result, cache_rows=active_rows, ordinary_cache_rows=ordinary_rows,
                record_window_chunks=routes)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a committed whole-request live RESTORE stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare, run = restore.experiment(key=KEY, root=ROOT, order=ORDER, plan=PLAN,
            physical_key=context.KEY, selector=phase.SELECTOR, prerequisites=prerequisites,
            launch_environment=environment, inspect_result=inspect,
            extra_sources=(THIS, context.THIS, cache.THIS, cache.horizon.THIS, grouped.THIS,
                           phase.THIS, phase.capacity.THIS, phase.geometry.THIS))
        prepare() if action == 'prepare' else run(action)
