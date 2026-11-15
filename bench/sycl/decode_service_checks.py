#!/usr/bin/env python3
"""Fresh same-build admission for disjoint decode service attribution.

Main-authored; repository LICENSE. Reuse the maintained owned host/phase
runners without inheriting the retired engine's qualification. Main owns all
native execution; no model, clean-speed or hardware-saturation claim follows.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import build_decode_service_model as model
import iq3_host_v2 as host_template
import check_verify_window_phase as device_template

B, RESULT, base = model.B, model.RESULT, model.base
SELF = Path(__file__).resolve()
HOST_ROOT, HOST_KEY = B / 'decode-service-model-host-v1', 'decode_service_model_host_v1'
DEVICE_ROOT = B / 'decode-service-model-phase-device-v1'
DEVICE_KEY = 'decode_service_model_phase_device_v1'
require = model.require


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controls(revision, *, controller=None):
    require(controller in (None, SELF), 'only this admission adapter owns these gates')
    pins = model.controls(revision)
    # Both template roots and the bench/tools import boundary must be pinned.
    for entry in (SELF, Path(host_template.__file__), Path(device_template.__file__)):
        pins.update(base.controls(revision, controller=entry))
    return pins


def controlled_base():
    # Do not mutate the shared module used by the native build's admission.
    return SimpleNamespace(**dict(vars(base), controls=controls))


def built():
    native = model.context().verify()
    checked = native['diagnostic_native_recipe_checks']
    require(checked['passed'] is True and checked['actual_profile_and_wait_bridge_headers'] is True
        and native['selected_basis']['centering_selected'] is False
        and native['selected_basis']['host_contract']['diagnostic_headers_and_fixture_match'] is True,
        'complete diagnostic build and matching schema2 host contracts required')
    return native


def host_context():
    ctx = load('strata_decode_service_host', host_template.__file__)
    ctx.ROOT, ctx.KEY, ctx.SELF = HOST_ROOT, HOST_KEY, SELF
    ctx.base = controlled_base()
    ctx.build = SimpleNamespace(ROOT=model.ROOT, component=model.previous.original.component)
    ctx.proof = built
    prepare = ctx.prepare

    def prepare_host():
        prepare()
        row = json.loads((HOST_ROOT / 'record.json').read_text())
        row['cases'] += [dict(kind='pool', arm=arm, observer=1,
            target='cpu_service_pool_test', args=['1']) for arm in (0, 1)]
        row['plan'] = ('Fresh diagnostic same-build exact0,pool0,exact1,pool1 children,'
            'then pool0/service1,pool1/service1;'
            '384single/80pair bitwise GGML/canary controls,80production dispatch calls,'
            'decode/prefill routing and28real-pool cases with the original checksum.'
            'PAIR0/1 and CPU_SERVICE0/1 are immutable before each exec;common AUDIT1,SINGLE_EXACT0.'
            'Private1MiB output scope,60s/512MiB per child,stop first failure,no retry.'
            'The separately passed seven-ledger/five-bridge contract matches these'
            ' projected headers. No GPU/model/speed qualification.')
        row['qualification_basis'] = dict(build_key=model.KEY,
            host_contract='decode_service_host_contract_v2',
            scope='Exact current five-output build;unchanged CPU arithmetic/pool and decode-only centering0;no inherited native qualification.')
        ctx.save(row)

    ctx.prepare = prepare_host
    ctx.run = lambda: run_host(ctx)
    return ctx


def run_host(ctx):
    row = json.loads((HOST_ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (HOST_ROOT / 'commands').exists(),
            'one frozen six-child same-build host attempt')
    scope, checksums = None, []

    def persist():
        model.write(HOST_ROOT / 'execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))

    try:
        ctx.unchanged(row)
        scope = base.supervision.Owner(HOST_ROOT / 'commands', persist=persist)
        deadline = time.monotonic() + 420
        for case in row['cases']:
            observer = case.get('observer', 0)
            env = dict(row['environment'], STRATA_IQ3_PAIR=str(case['arm']),
                STRATA_CPU_SERVICE=str(observer))
            label = case['kind'] + '-' + str(case['arm']) + '-service' + str(observer)
            entry, out, err = scope.run(label,
                [row['binaries'][case['target']]['path'], *case['args']], env, HOST_ROOT,
                **dict(row['budgets'], wall=min(60, deadline - time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code'] == 0
                and err.stat().st_size == 0, 'same-build host child failed: ' + label)
            text = out.read_text().strip()
            if case['kind'] == 'exact':
                expected = ('IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries\n'
                    f'IQ3 production dispatch PASS arm={case["arm"]} calls=80 nt1_rows=20480 pair_rows={case["arm"] * 10240} prefill_pair_rows=0')
                require(text == expected, 'complete same-build exact/dispatch footer required')
            else:
                match = re.fullmatch(r'cpu_service_pool PASS observer=' + str(observer) +
                    r' formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})', text)
                require(match is not None, 'complete same-build pool/observer footer required')
                checksums.append(match[1])
            row['results'].append(dict(kind=case['kind'], arm=case['arm'], observer=observer,
                passed=True, exit_code=0, stdout_sha256=model.digest(out), ownership_closed=True))
        require(checksums == [model.previous.original.component()['correctness']['pool_checksum']] * 4,
                'same-build pool bits differ across pairing or observation modes')
        ctx.unchanged(json.loads((HOST_ROOT / 'record.json').read_text()))
        row.update(stage='closed', passed=True, complete=True, pool_checksum=checksums[0])
    except BaseException as error:
        row.update(stage='failed' if scope else 'failed-admission',
            passed=False, complete=False, error=repr(error))
        raise
    finally:
        row.update(commands=scope.commands if scope else [], active=bool(scope and scope.active),
            ownership_closed=scope is None or (scope.active is None and
                all(base.supervision.closed(v) for v in scope.commands)))
        ctx.save(row)
    print('Six same-build arithmetic/pool/service children passed;no GPU/model qualification.', flush=True)


def qualified_host(native):
    ctx = host_context()
    row = json.loads((HOST_ROOT / 'record.json').read_text())
    ctx.unchanged(row)
    require(row['stage'] == 'closed' and row['passed'] is True and row['complete'] is True
        and row['ownership_closed'] is True and row['active'] is False
        and row['GPU_executed'] is False and row['native_rebuilt'] is False
        and row['binaries'] == native['binaries']
        and row['native_source_commit'] == native['source_commit']
        and [(v['kind'], v['arm'], v['observer'], v['passed'], v['exit_code']) for v in row['results']] ==
            [('exact', 0, 0, True, 0), ('pool', 0, 0, True, 0),
             ('exact', 1, 0, True, 0), ('pool', 1, 0, True, 0),
             ('pool', 0, 1, True, 0), ('pool', 1, 1, True, 0)]
        and row['pool_checksum'] == model.previous.original.component()['correctness']['pool_checksum'],
        'fresh complete same-build exact arithmetic and actual pool gate required')


def device_context():
    ctx = load('strata_decode_service_phase', device_template.__file__)
    native_context = model.context()
    ctx.base = controlled_base()
    ctx.build = SimpleNamespace(ROOT=model.ROOT, KEY=model.KEY,
        compact=native_context.compact, inputs=model.inputs, selected=model.template.selected)
    ctx.ROOT, ctx.KEY, ctx.SELF = DEVICE_ROOT, DEVICE_KEY, SELF
    original_built = ctx.built

    def qualified_build():
        native = original_built()
        require(native['diagnostic_native_recipe_checks']['passed'] is True,
                'complete native diagnostic header/flag gate required')
        qualified_host(native)
        return native

    ctx.built = qualified_build
    return ctx


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ('host', 'device')
        and sys.argv[2] in ('prepare', 'run'), 'use host|device prepare|run')
    ctx = host_context() if sys.argv[1] == 'host' else device_context()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': ctx.prepare, 'run': ctx.run}[sys.argv[2]]()
