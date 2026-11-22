"""Compare demand and batched expert copies on the measured KV/gather route."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_gather_store as gather
import prefill_kv_upload_build as build
import prefill_split_upload_compare as previous

bank, basis, require = previous.bank, previous.basis, previous.require
READ = previous.READ
MODES = ('demand', 'batched')
BLOCKS = (MODES, MODES[::-1], MODES)
KEYS = {'admit': 'prefill_kv_upload_admission_v1', 'time': 'prefill_kv_upload_speed_v1'}
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
    ctx = comparison.experiment(phase)
    original_pins = ctx.pins

    def pins(revision):
        return original_pins(revision) | gather.controller_sources(revision)

    ctx.pins = pins
    return ctx


def main(phase, action):
    require(sys.executable == basis.reference.model.PYTHON and phase in KEYS,
            'use the benchmark interpreter and select admit or time')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
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
