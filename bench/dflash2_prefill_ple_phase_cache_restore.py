"""Verify live conversation RESTORE on the measured phase-local cache build."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_restore as earlier
import dflash2_prefill_ple_phase_cache as phase
import dflash2_prefill_ple_phase_cache_context as context

bank, basis, require = earlier.bank, earlier.basis, earlier.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_phase_cache_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', 'stock', 'dedicated')
PLAN = dict(order=list(ORDER), prefix=earlier.PREFIX, fresh_suffix=earlier.SUFFIX,
    scope='Park 16385 tokens, switch to a 33-token request, then restore and append 118785 fresh tokens. '
          'Keep early issue, the direct writer, the ordinary cache and the exact native build fixed.',
    gates='Observe each first-chunk cache choice on the initial prefix and restored suffix. Verify the same '
          'RAM snapshot, active state, emitted IDs, work counters and per-layer KV ledger and digest.',
    limits='Traced live correctness at a nonzero position. Sustained decode and agent performance '
           'require separate measurements.')


def prerequisites():
    native, baseline, speed_sha = context.prerequisites(None)
    physical = basis.validation.committed(bank.RESULT, (context.KEY,))
    expected = dict(prefill_rows=phase.capacity.LARGE_ROWS, ordinary_rows=phase.capacity.DEFAULT_ROWS,
                    observed_request_routes=1)
    require(phase.wait.retained.closed(physical, 'closed') and physical['timing_sha256'] == speed_sha
            and all(physical['outcomes'][name]['phase_ple_cache'] == expected for name in earlier.capacity.physical.ORDER),
            'finish the physical-context check with the dedicated cache active on both fixtures')
    return native, baseline, speed_sha


def environment(row, case):
    env = earlier.environment(row, dict(case, name='early'))
    require(phase.SELECTOR not in env and env.get('STRATA_PREFILL_PLE_RECORD_AUDIT', '0') == '0',
            'unexpected cache selector or record observer')
    return dict(env, **{phase.SELECTOR: str(phase.capacity.LARGE_ROWS if case['name'] == 'dedicated' else 0)})


def inspect(peer, text, pid, arm, requests, row):
    result = earlier.inspect(peer, text, pid, 'early', requests, row)
    dedicated = arm == 'dedicated'
    expected = f'{phase.START}long_prefill={phase.capacity.LARGE_ROWS} ordinary={phase.capacity.DEFAULT_ROWS}'
    require([line for line in text.splitlines() if line.startswith(phase.START)] == ([expected] if dedicated else [])
            and phase.TICKET not in text, 'actual restore cache startup or observer differs')
    routes = []
    begin = 0
    states = re.finditer(r'^strata serve: STATE_HASH L=.*$', text, re.M)
    for count, state in zip((1, 0, 1), states, strict=True):
        phase.check_first_cache(text[begin:state.start()], 'large' if dedicated else 'stock', count)
        routes.append(count)
        begin = state.end()
    require(phase.FIRST not in text[begin:], 'a cache route appeared after the final request')
    return dict(result, phase_ple_cache=dict(prefill_rows=phase.capacity.LARGE_ROWS if dedicated
                                             else phase.capacity.DEFAULT_ROWS,
                                             ordinary_rows=phase.capacity.DEFAULT_ROWS,
                                             request_route_counts=routes))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a committed dedicated-cache restore stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare, run = earlier.experiment(
            key=KEY, root=ROOT, order=ORDER, plan=PLAN, physical_key=context.KEY, selector=phase.SELECTOR,
            prerequisites=prerequisites, launch_environment=environment, inspect_result=inspect,
            extra_sources=(THIS, context.THIS, phase.THIS, phase.capacity.THIS, phase.geometry.THIS))
        prepare() if action == 'prepare' else run(action)
