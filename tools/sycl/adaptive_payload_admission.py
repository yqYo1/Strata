#!/usr/bin/env python3
"""Bind unchanged component evidence to the one-header payload successor.

Main-authored; repository LICENSE. Reuse establishes identity and unchanged
CPU/GPU components only. Native32 positive/refusal and model speed remain open.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys

sys.dont_write_bytecode = True
import adaptive_native_admission as previous
import build_adaptive_payload_model as build

P, B, RESULT, base = build.P, build.B, build.RESULT, build.base
SELF = Path(__file__).resolve()
require, digest, write = build.require, build.digest, build.write
closed = previous.closed


def compiled():
    row = build.factory().context().verify()
    checks = row['native_recipe_checks']
    require(checks['passed'] and checks['only_audit_header_changed'] and
        checks['same_five_fixture_bytes'] and checks['host_audit_cases'] == 57,
        'closed actual-source/recipe/five-fixture successor checks required')
    host = row.get('host_result', {})
    require(host and closed(host) and [(v['mode'], v['exit_code'], v['cases']) for v in host['cases']] ==
        [(0, 0, 28), (1, 0, 28)] and host['cases'][0]['checksum'] == host['cases'][1]['checksum'],
        'fresh same-build pool0/1 parity required')
    return row


def evidence(recover=False):
    row = compiled(); old, _ = build.basis()
    require(set(row['native_sources']) == set(old['native_sources']) and
        len(row['native_sources']) == 568 and
        {n for n in row['native_sources'] if row['native_sources'][n] != old['native_sources'][n]} ==
        {build.source.HEADER}, 'only the diagnostic audit header may differ from429b')
    names = set(old['binaries']) - {'strata'}
    require(len(names) == 5 and set(row['binaries']) == set(old['binaries']) and all(
        (row['binaries'][n]['sha256'], row['binaries'][n]['bytes']) ==
        (old['binaries'][n]['sha256'], old['binaries'][n]['bytes']) for n in names),
        'five byte-identical successor fixtures required')
    prior = old['component_reuse']
    require(prior['passed'] and prior['byte_identical_fixtures'] == 5 and
        prior['unchanged_native_files'] == 566 and prior['host_audit_header_cases'] == 41 and
        prior['original_device_record_sha256'] == previous.DEVICE_SHA,
        'committed original component chain changed')
    require(digest(previous.DEVICE) == previous.DEVICE_SHA, 'original device evidence changed')
    device = json.loads(previous.DEVICE.read_text())
    require(closed(device) and not device['fresh_faults'] and device['boot_unchanged'] and
        device['runtime_manifest'] == prior['runtime_manifest'] and
        [(v['name'], v['native_exit'], v['checks'], v['ownership_closed']) for v in device['results']] ==
        [('submission', 0, 10, True), ('q6-half', 0, 32, True), ('phase-cache', 0, 16, True)],
        'original component device closure changed')
    manifest = device['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'runtime manifest changed')
    providers = json.loads(Path(manifest['path']).read_text())
    require(len(providers) == manifest['files'] and all(digest(p) == h for p, h in providers.items()),
        'runtime provider changed')
    for value in device['native_validation_probe'].values():
        require(digest(value['path']) == value['sha256'], 'before-main validation probe changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == device['boot'] and
        base.guards.driver_guard() == device['driver'], 'boot/driver changed')
    if recover:
        # Before retirement,validate the complete prior native/runtime chain
        # against its still-present ELFs. Later readers verify the scoped saved
        # evidence,Git inputs and new ELFs,so deleting429b does not strand them.
        observed, prior_device, prior_providers = previous.native()
        require(observed == old and prior_device == device and prior_providers == providers,
            'original live component chain differs')
        for value, key in ((old, 'controller_inputs'), (device, 'controller_inputs')):
            recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                base.basis.build.blobs(P, value['source_commit'], value[key])}
            require(recovered == value[key], 'recover original component controls')
    consumed = dict(source_commit=old['source_commit'], native_sources=old['native_sources'],
        binaries=old['binaries'], recipe=old['recipe'], component_reuse=prior)
    proof = dict(passed=True, unchanged_native_files=567, byte_identical_fixtures=5,
        fixture_sha256={n: row['binaries'][n]['sha256'] for n in sorted(names)},
        predecessor_consumed_fields_sha256=build.signature(consumed),
        original_device_record_sha256=previous.DEVICE_SHA,
        original_component_controls_recovered=True, host_audit_header_cases=57,
        fresh_CPU_pool_cases_per_mode=28, device_checks=[10, 32, 16], runtime_manifest=manifest,
        boot_unchanged=True, driver_unchanged=True, GPU_executed=False, model_executed=False,
        scope='Unchanged components and actual five-fixture identity only. The successor header '
            'has57host cases;real32-copy publication/dispatch/occupied loan,checked second-record '
            'refusal,quality,speed,ordinary provider teardown and physical262144remain unqualified. '
            'Scoped predecessor metadata may gain retirement fields;its consumed fields must not change.')
    return row, device, providers, proof


def native():
    row, device, providers, proof = evidence()
    saved = row.get('component_reuse', {})
    require({n: v for n, v in saved.items() if n not in ('source_commit', 'controller_inputs')} == proof,
        'commit exact successor component reuse first')
    pins = base.controls(saved['source_commit'], controller=SELF)
    require(saved['controller_inputs'] == dict(files=len(pins), sha256=build.signature(pins)),
        'successor reuse controls changed')
    return row, device, providers


def reuse():
    marker = build.ROOT / 'component-reuse-attempt.json'
    require(not marker.exists(), 'one successor reuse attempt only')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = base.controls(revision, controller=SELF)
    with marker.open('x') as stream: stream.write('{}\n')
    try:
        row, _, _, proof = evidence(recover=True)
        proof.update(source_commit=revision,
            controller_inputs=dict(files=len(pins), sha256=build.signature(pins)))
        row['component_reuse'] = proof; build.save(row)
    except BaseException as error:
        write(marker, dict(passed=False, complete=False, error=repr(error))); raise
    write(marker, dict(passed=True, complete=True, GPU_executed=False))
    print('Successor component identity qualified;real native32 remains unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] == ['reuse'], 'use reuse')
    resource.setrlimit(resource.RLIMIT_CPU, (150, 150)); signal.alarm(180)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([]); reuse()
