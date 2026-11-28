"""Exercise the final physical slot with whole-request PLE selected."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_capacity as earlier
import dflash2_prefill_ple_horizon_cache as cache
import dflash2_prefill_ple_record_window as grouped

bank, wait, require, phase = grouped.bank, grouped.wait, grouped.require, grouped.phase
basis = earlier.basis
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_horizon_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
PLAN = dict(order=list(earlier.PLAN['order']), record_horizon='full',
    dedicated_prefill_rows=cache.capacity.LARGE_ROWS, ordinary_rows=cache.capacity.DEFAULT_ROWS,
    scope='Use the timed whole-request build. Select the dedicated prefill cache and full record horizon.',
    gates='Verify the same-build host bounds and the original mode-10 handoff. Consume all 262144 input '
          'slots on the mode-2 route with zero outputs. Verify both actual fallback routes, prefill-only '
          'cache capacity, output IDs, final commit, model hashes and native closure.',
    limits='The physical fallback and the validated mode-16 grouped path have distinct coverage. '
           'Live RESTORE, task quality and sustained speed have separate checks.')


def prerequisites(_reference_context):
    campaign = cache.experiment()
    baseline = campaign.baseline()
    timing = grouped.validate.committed(bank.RESULT, (cache.KEY,))
    require(wait.retained.closed(timing, 'closed')
            and timing['plan'] == campaign.protocol
            and timing['decision'] == cache.decision(timing['processes'])
            and timing['decision']['warm_cache_advances']
            and all(timing[k] == baseline[k] for k in (
                'binary', 'native_source_commit', 'native_build_sha256',
                'boot', 'driver', 'runtime_manifest', 'dflash2_assets', 'native_validation_probe')),
            'require the exact measured whole-request cache configuration')
    native = bank.previous.read(grouped.BUILD_ROOT / 'record.json')
    require(native['binaries']['strata'] == timing['binary']
            and all(name in native['binaries'] for name in (
                'strata_full_context_diagnostic', 'full_context_diagnostic_host_test')),
            'use the physical model and host fixture from the measured build')
    return native, baseline, bank.signature(timing)


def experiment():
    load = basis.core.source.device.load_module
    standalone = load('whole_ple_capacity', earlier.__file__)
    standalone.ROOT, standalone.KEY, standalone.PLAN = ROOT, KEY, PLAN
    standalone.prerequisites = prerequisites
    ctx = standalone.experiment()
    current_base, save_base = ctx.current, ctx.save
    env_base, evidence_base, pins_base = ctx.environment, ctx.evidence, ctx.source_pins

    def environment(row, case):
        env = env_base(row, case)
        require(not any(key in env for key in (phase.SELECTOR, grouped.SELECTOR))
                and env.get(grouped.geometry.SELECTOR, '0') == '0',
                'unexpected cached or instrumented PLE environment')
        return dict(env, **{phase.SELECTOR: str(cache.capacity.LARGE_ROWS), grouped.SELECTOR: 'all'})

    def launch_hashes(row):
        return {p['name']: bank.signature(environment(row, p)) for p in row['cases']}

    def save(row):
        if row['stage'] == 'prepared':
            row['effective_launch_environment_sha256'] = launch_hashes(row)
        save_base(row)

    def current(row):
        current_base(row)
        require(row['effective_launch_environment_sha256'] == launch_hashes(row),
                'standalone effective launch environment changed')

    def evidence(stdout, stderr, row, case, pid):
        result = evidence_base(stdout, stderr, row, case, pid)
        expected = (f'{phase.START}long_prefill={cache.capacity.LARGE_ROWS} '
                    f'ordinary={cache.capacity.DEFAULT_ROWS}')
        require([line for line in stderr.splitlines() if line.startswith(phase.START)] == [expected]
                and not any(marker in stderr for marker in (
                    grouped.MARKER, grouped.PRODUCER, grouped.CONSUMER, phase.TICKET))
                and result['window']['mode'] == (2 if case['full'] else 10),
                'actual physical route or dedicated cache setting changed')
        phase.check_first_cache(stderr, 'large', 1)
        require(not case['full'] or result['batched_rows'] == 262143
                and result['generated_ids'] == 0 and result['final_slot_consumed'],
                'the actual final physical input was not consumed')
        return dict(result, ple_horizon_configured='full', ple_horizon_executed=False,
                    prefill_cache_rows=cache.capacity.LARGE_ROWS,
                    ordinary_cache_rows=cache.capacity.DEFAULT_ROWS)

    def pins(revision):
        files = pins_base(revision)
        for name in (THIS, cache.THIS, cache.horizon.THIS, grouped.THIS, phase.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the whole-request capacity input: ' + name)
            files[name] = bank.sha(path)
        return files

    ctx.environment, ctx.evidence, ctx.source_pins = environment, evidence, pins
    ctx.save, ctx.current = save, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON
            and action in ('prepare', 'host', *earlier.physical.ORDER),
            'select a committed standalone capacity stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment()
        control.prepare() if action == 'prepare' else control.run(action)
