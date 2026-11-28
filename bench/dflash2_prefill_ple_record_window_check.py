"""Check grouped PLE native STOP, first-chunk use and direct-writer state."""

import fcntl
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_ple_early_check as earlier
import dflash2_prefill_ple_record_window as grouped

bank, require = grouped.bank, grouped.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
DEPTH = {'cancel': 2, 'one': 1, 'direct': 2}
FIRST_DIRECT_SUBMIT = 'strata trace: prefill PLE first direct read submitted'


def experiment(whole):
    depths = {'direct': 'all'} if whole else DEPTH
    key = grouped.FULL_SMOKE_KEY if whole else grouped.SMOKE_KEY
    load = grouped.wait.basis.core.source.device.load_module
    harness = load('record_window_native_check', earlier.__file__)
    if not whole:
        stop = load('record_window_stop_after_io_submit', earlier.__file__)
        stop.ISSUED = FIRST_DIRECT_SUBMIT
        harness.stop_at_layer_zero = stop.stop_at_layer_zero
    original = harness.build
    harness.ROOT, harness.KEY, harness.ORDER = bank.B / key.replace('_', '-'), key, tuple(depths)
    harness.build = SimpleNamespace(ROOT=grouped.BUILD_ROOT, wait=original.wait,
                                    source=original.source, builder=original.builder)
    original_inspect = harness.inspect
    source_context = grouped.build_context()

    def inputs(revision):
        native = bank.previous.read(grouped.BUILD_ROOT / 'record.json')
        host_proof = grouped.checked_host(native)
        plan, ref, environments, requests, expected = harness.inputs(revision, source_context)
        require((bank.P / THIS).read_bytes() == grouped.validate.git('show', revision + ':' + THIS),
                'commit the grouped STOP controller')
        for name, env in environments.items():
            require(grouped.SELECTOR not in env
                    and grouped.phase.SELECTOR not in env
                    and env.get(grouped.geometry.SELECTOR, '0') == '0',
                    'unexpected grouped reader, cache or legacy observer')
            env.update({grouped.SELECTOR: str(depths[name]), grouped.attribution.ple.SELECTOR: '1'})
            plan['cases'][name]['environment_sha256'] = bank.signature(env)
        scope = ('Verify whole-request direct-writer state, chunk publication and both host slots.' if whole else
                 'Cancel after the first DirectFile submission and run the next request on the same engine. '
                 'Check direct-writer state in fresh one- and two-chunk engines.')
        plan.update(host_fixture_sha256=host_proof, record_window_chunks=depths, scope=scope,
                    smoke_controller_sha256=bank.sha(bank.P / THIS))
        if not whole:
            plan['stop_marker'] = FIRST_DIRECT_SUBMIT
        return plan, ref, environments, requests, expected

    def inspect(text, pid, plan, name, responses, reference):
        requests = list(harness.tap.taps.REQUEST.finditer(text))
        require([tuple(map(int, match.groups())) for match in requests]
                == [(n, 0) for n in harness.tap.taps.PROMPTS], 'actual diagnostic prompts differ')
        middle = text[requests[1].start():requests[2].start()]
        expected = [(start, 8192, index & 1) for index, start in enumerate(range(0, 122880, 8192))]
        chunk_count = len(expected) if whole else depths[name]
        marker = f'{grouped.MARKER}tokens_per_chunk=8192 window_chunks={chunk_count}'
        require([line for line in middle.splitlines() if line.startswith(grouped.MARKER)] == [marker]
                and text.count(grouped.MARKER) == 1 and text.count(harness.ISSUED) == 1,
                'the long request did not execute the selected grouped reader')
        require(text.count(FIRST_DIRECT_SUBMIT) == middle.count(FIRST_DIRECT_SUBMIT) == 1,
                'did not observe the first accepted grouped DirectFile read')
        if name == 'cancel':
            require(grouped.CONSUMER not in text, 'a grouped chunk was uploaded on the cancelled request')
            outcome = original_inspect(text, pid, plan, name, responses, reference)
            return dict(outcome, first_direct_read_submitted=True)
        require(middle.count(harness.LAYER1) == 1
                and middle.index(marker) < middle.index(harness.LAYER1)
                and 'strata prefill dense: phase=restored residency=Full activations=48 ' in middle,
                'the grouped reader failed to restore dense weights')
        consumers = grouped.fields(middle, grouped.CONSUMER, grouped.CONSUMER_FIELDS)
        require(middle.index(harness.LAYER1) < middle.index(grouped.CONSUMER)
                and [tuple(map(int, row[:3])) for row in consumers] == expected
                and text.count(grouped.CONSUMER) == len(expected), 'actual grouped consumption differs')
        gated = harness.tap.inspect(text, pid, plan, 'direct', responses)
        outcome = dict(gated, requests=responses, record_window_chunks=chunk_count, consumed_chunks=len(consumers))
        require(all(outcome[k] == reference[k] for k in ('requests', 'states', 'tap_payloads', 'route')),
                'grouped reading altered output IDs, state, taps or work')
        return outcome

    harness.inspect = inspect
    return harness, inputs


if __name__ == '__main__':
    whole = sys.argv[1:2] == ['--whole']
    if whole:
        sys.argv.pop(1)
    action, = sys.argv[1:]
    require(sys.executable == grouped.wait.basis.reference.model.PYTHON, 'use the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        harness, inputs = experiment(whole)
        root, key, order = harness.ROOT, harness.KEY, harness.ORDER
        require(action in ('prepare', *order), 'select a committed grouped reader check stage')
        if action == 'prepare':
            require(not root.exists() and key not in grouped.validate.committed(bank.RESULT), 'fresh native check only')
            revision = grouped.validate.git('rev-parse', 'HEAD').decode().strip()
            plan, *_ = inputs(revision)
            for name in order:
                for folder in ('cache', 'tmp'):
                    (root / name / folder).mkdir(parents=True, mode=0o700)
            harness.save(dict(plan=plan, stage='prepared', outcomes={}, active=False, complete=False, passed=False,
                              ownership_closed=True, speed_eligible=False))
        else:
            row = bank.previous.read(root / 'record.json')
            require(row == grouped.validate.committed(bank.RESULT, (key,)) and not row['active']
                    and row['ownership_closed'] and tuple(row['outcomes']) == order[:order.index(action)]
                    and row['stage'] in ('prepared', 'partial'), 'commit the preceding successful native stage')
            plan, ref, environments, requests, expected = inputs(row['plan']['source_commit'])
            require(row['plan'] == plan, 'the pinned grouped reader check changed')
            harness.run(row, ref, action, environments[action], requests, expected)
