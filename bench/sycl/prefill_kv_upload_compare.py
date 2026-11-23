"""Compare upload scheduling and attention output on the measured KV/gather route."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_gather_store as gather
import prefill_kv_upload_build as build
import prefill_split_upload_compare as previous
from engine_once import native_environment

bank, basis, require = previous.bank, previous.basis, previous.require
READ = previous.READ
MODES = ('demand', 'batched')
BLOCKS = (MODES, MODES[::-1], MODES)
KEYS = {'admit': 'prefill_kv_upload_admission_v2', 'time': 'prefill_kv_upload_speed_v2',
        'store-admit': 'prefill_kv_upload_store_admission_v1', 'store-time': 'prefill_kv_upload_store_speed_v1'}
UPLOAD_TIMING = '297adab4dfbd5da2d3a433962174238badd255f8'
STORE_FIXTURE_KEY = 'prefill_kv_upload_store_component_v1'
STORE_FIXTURE_ROOT = bank.B / STORE_FIXTURE_KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def admission(traced, runs):
    original, prior, providers, extra, measured = gather.measured_route(READ)
    native = bank.previous.read(build.ROOT / 'record.json')
    build.context().check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['parent_binary'] == original['binaries']['strata']
            and native['driver'] == prior['driver'], 'complete the coherent KV/upload build and host controls')
    device_fixture = native['binaries']['sycl_expert_bank_test']
    earlier = bank.previous.historic(build.UPLOAD_EVIDENCE, 'prefill_split_upload_native_v1')
    require(device_fixture['sha256'] == earlier['binaries']['sycl_expert_bank_test']['sha256']
            and device_fixture['bytes'] == earlier['binaries']['sycl_expert_bank_test']['bytes'],
            'this plan requires the exact device-tested expert upload fixture')

    control = measured['processes'][gather.speed.ORDER.index(gather.SELECTION[READ])]
    expected = [previous.split.work(dict(route, requested_bank_H2D_bytes=
                   route['fast_window']['requested_bank_H2D_bytes']), sample)
                for route, sample in zip(control['routes'][:runs], control['samples'][:runs], strict=True)]
    require(len(expected) == runs and [w['request'] for w in expected] == measured['request_hashes'][str(READ)][:runs]
            and all(w['reads'] == READ and w['store']['gpu_rows'] == READ - 1 and w['store']['host_rows'] == 0
                    and w['store']['direct'] == 0 and w['store']['copy_bytes'] == 115495403520 for w in expected),
            'recover the exact all-GPU copy-control workloads')
    proof = dict(reference_configuration=measured['configuration'], expected_work=expected,
                 build_sha256=bank.signature(bank.builder.compact(native)), traced=traced,
                 reference_sha256=bank.signature(control), device_fixture=device_fixture)
    if not traced:
        admitted = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
        require(gather.closed(admitted) and admitted['binary'] == native['binaries']['strata']
                and admitted['native_source_commit'] == native['native_source_commit']
                and admitted['decision']['admitted_modes'] == list(MODES)
                and admitted['decision']['matched_decode_loan']
                and admitted['observer_admission']['build_sha256'] == proof['build_sha256']
                and admitted['observer_admission']['expected_work'] == expected[:2],
                'complete both traced upload modes with verified cache handoff first')
        proof['traced_admission_sha256'] = bank.signature(admitted)
    prior['observer_admission'].update(proof)
    return native, prior, providers, extra


def store_reference():
    native, prior, providers, extra = admission(False, 4)
    timing = bank.previous.historic(UPLOAD_TIMING, KEYS['time'])
    decision = timing['decision']
    require(gather.closed(timing) and timing['binary'] == native['binaries']['strata']
            and timing['native_source_commit'] == native['native_source_commit']
            and timing['observer_admission'] == prior['observer_admission']
            and timing['boot'] == prior['boot'] and timing['driver'] == prior['driver']
            and decision['repeated_warm_gain']['batched/demand'] and decision['repeated_1000']['batched'],
            'recover the closed same-build batched upload winner')
    return native, prior, providers, extra, timing


def fixture_inputs(revision):
    native, prior, providers, _, timing = store_reference()
    comparison = experiment('time')
    live = bank.previous.read(comparison.ROOT / 'record.json')
    require(basis.compact(live) == timing and live['native_validation_probe'] == prior['probe']
            and live['boot'] == gather.fixture.attention.BOOT.read_text().strip()
            and all(bank.sha(path) == digest for path, digest in providers.items()),
            'recover the measured fixture launch environment and runtime providers')
    env = previous.model.ordinary_environment(live, str(STORE_FIXTURE_ROOT),
                                              prior['observer_admission']['ordinary_control'])
    env.update(TMPDIR=str(STORE_FIXTURE_ROOT / 'tmp'), XDG_CACHE_HOME=str(STORE_FIXTURE_ROOT / 'cache'))
    env = native_environment(live, env)
    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr')
    pins = comparison.pins(revision)
    pins.update(bank.stage.source.pin(bank.P, [gather.fixture.attention.CONTROLLER])['source_sha256'])
    fixture = native['binaries'][gather.fixture.FIXTURE]
    plan = dict(source_commit=revision, sources_sha256=bank.signature(pins), sources_count=len(pins),
                native_source_commit=native['native_source_commit'],
                native_build_sha256=prior['observer_admission']['build_sha256'],
                engine=native['binaries']['strata'], fixture=fixture,
                command=[fixture['path'], 'correctness'], budgets=gather.fixture.attention.BUDGET,
                environment_sha256=bank.signature(env), providers_sha256=bank.signature(providers),
                observer_reference=UPLOAD_TIMING, boot=live['boot'], driver=live['driver'],
                scope='Exact BF16 store bits, tails, guards and checked cancellation on the KV/upload engine.')
    return plan, env, prior['probe']


def fixture_context():
    ctx = basis.core.source.device.load_module('kv_upload_store_fixture', gather.fixture.__file__)
    ctx.THIS, ctx.KEY, ctx.ROOT, ctx.inputs = THIS, STORE_FIXTURE_KEY, STORE_FIXTURE_ROOT, fixture_inputs
    ctx.validate_runtime = gather.validate_traced_fixture
    return ctx


def store_basis(runs, traced):
    native, prior, providers, extra, timing = store_reference()
    component = basis.validation.committed(bank.RESULT, (STORE_FIXTURE_KEY,))
    plan = component['plan']
    observer = prior['observer_admission']
    require(gather.closed(component) and plan['engine'] == native['binaries']['strata']
            and plan['fixture'] == native['binaries'][gather.fixture.FIXTURE]
            and plan['native_source_commit'] == native['native_source_commit']
            and plan['native_build_sha256'] == observer['build_sha256']
            and plan['observer_reference'] == UPLOAD_TIMING and plan['boot'] == prior['boot']
            and plan['driver'] == prior['driver'] and plan['providers_sha256'] == bank.signature(providers)
            and component['result']['cases'] == 4 and not component['result']['fresh_faults'],
            'complete the same-build traced BF16 store, bounds and cancellation check')

    admitted = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
    copy = admitted['processes'][MODES.index('batched')]
    expected = observer['expected_work'][:runs]
    require(copy['mode'] == 'batched' and copy['passed'] and copy['ownership_closed']
            and copy['native_exit_code'] == 0 and not copy['fresh_faults']
            and [previous.split.work(route, sample) for route, sample in
                 zip(copy['routes'], copy['samples'], strict=True)] == expected[:2]
            and all(work['store']['pieces'] == 720 for work in expected),
            'recover the two matching traced batched copy requests')
    observer.update(traced=traced, timing_evidence=UPLOAD_TIMING, timing_sha256=bank.signature(timing),
                    direct_store_component_sha256=bank.signature(component),
                    expected_loaned_slots=[route['loaned_slots'] for route in copy['routes']],
                    expected_work=previous.direct_work(expected) if traced else expected)
    if not traced:
        gate = basis.validation.committed(bank.RESULT, (KEYS['store-admit'],))
        traced_proof = dict(observer, traced=True, expected_work=previous.direct_work(expected[:2]))
        require(gather.closed(gate) and gate['binary'] == native['binaries']['strata']
                and gate['native_source_commit'] == native['native_source_commit']
                and gate['plan']['direct_output'] and gate['observer_admission'] == traced_proof
                and gate['decision']['matched_decode_loan'],
                'complete the traced direct model route and decode loan check before timing')
        observer['direct_store_admission_sha256'] = bank.signature(gate)
    return native, prior, providers, extra


def request_route(text, plan):
    route = previous.request_route(text, plan)
    store = route['attention_store']
    require(store['gpu_rows'] == READ - 1 and store['host_rows'] == 0 and store['window_mode'] == 10,
            'the upload comparison requires the original all-GPU split workspace')
    return route


def experiment(phase):
    comparison = basis.core.source.device.load_module('kv_upload_comparison', previous.__file__)
    comparison.THIS, comparison.KEYS, comparison.build = THIS, KEYS, build
    comparison.MODES, comparison.BLOCKS, comparison.CONTRASTS = MODES, BLOCKS, (('batched', 'demand'),)
    comparison.admission, comparison.request_route = admission, request_route
    comparison.store_basis = store_basis
    ctx = comparison.experiment(phase)
    original_pins = ctx.pins

    def pins(revision):
        return original_pins(revision) | gather.controller_sources(revision)

    ctx.pins = pins
    return ctx


def main(phase, action):
    require(sys.executable == basis.reference.model.PYTHON and phase in (*KEYS, 'store-fixture'),
            'use the benchmark interpreter and select admit, time, store-fixture, store-admit or store-time')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if phase == 'store-fixture':
            require(action in ('prepare', 'run'), 'select prepare or run for the store fixture')
            fixture = fixture_context()
            {'prepare': fixture.prepare, 'run': fixture.run}[action]()
            return
        ctx = experiment(phase)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            require(action in map(str, range(len(ctx.ORDER))), 'select one planned process')
            basis.comparison.run(int(action), ctx)


if __name__ == '__main__':
    main(*sys.argv[1:])
