#!/usr/bin/env python3
"""Prove unchanged component qualification for the new decode-gate engine.

Main-authored; repository LICENSE. Five actual fixture ELFs are byte-identical
to cb7. Reuse their limited qualification;the changed generator still needs
its own bounded model/gate check. No new native process is launched here.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import build_decode_profile_model as build
import profile_split as profile

P, B, RESULT, base = build.P, build.B, build.RESULT, build.base
SELF = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write
DEVICE = B / 'q6-native-half-device-v1/record.json'
COLLECTOR = build.OLD / 'collector-active-v1/record.json'
WINDOW = B / 'compat-collector-window-v2/checks-v1/record.json'


def closed(row):
    return row['passed'] and row['complete'] and row['ownership_closed'] and not row.get('active',False)


def compiled():
    """Execution uses closed compile evidence,actual ELFs and runtime providers.

    Rebuilding separately requires the complete compiler/package/source image.
    """
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[build.KEY] == build.compact(row) and
            row['stage'] == 'built-not-executed' and closed(row) and row['native_recipe_checks']['passed'] and
            all(base.supervision.closed(c) and c['exit_code'] == 0 for c in row['commands_executed']),
            'commit closed exact native build first')
    require(base.controls(row['source_commit'],controller=build.SELF) == row['controller_inputs'],
            'native build controls changed')
    _, _, values = build.projection()
    require(build.selected.pins(values) == row['native_sources'] and all(
        (build.ROOT / 'source' / n).read_bytes() == raw for n,raw in values.items()),
        'current568-source construction changed')
    for asset in row['binaries'].values():
        require(digest(asset['path']) == asset['sha256'] and Path(asset['path']).stat().st_size == asset['bytes'],
                'actual current engine/fixture changed')
    for key in ('compiled_inputs','compile_database'):
        require(digest(row[key]['path']) == row[key]['sha256'], 'closed compile evidence changed')
    actual = json.loads(Path(row['compiled_inputs']['path']).read_text())
    require(len(actual) == row['compiled_inputs']['files'] and
            actual.get(str(build.ROOT / 'source' / build.source.HEADER)) == build.source.HEADER_SHA,
            'actual dependency closure/header differs')
    return row


def recover(revision, pins):
    values = {n:hashlib.sha256(raw).hexdigest() for n,raw in base.basis.build.blobs(P,revision,pins)}
    require(values == pins, 'original component controls not recoverable')


def evidence(*, recover_controls=False):
    data, row = base.committed(RESULT), compiled()
    require(digest(build.OLD / 'record.json') == build.host.NATIVE_RECORD_SHA, 'original cb7 record changed')
    old = json.loads((build.OLD / 'record.json').read_text())
    require(data[build.selected.KEY] == build.selected.compact(old), 'original native result changed')
    unchanged = {n:h for n,h in old['native_sources'].items() if n != build.source.GENERATOR}
    require(len(unchanged) == 566 and all(row['native_sources'].get(n) == h for n,h in unchanged.items()) and
            set(row['native_sources']) - set(old['native_sources']) == {build.source.HEADER},
            'reuse requires unchanged CPU/GPU kernels,runtime,fixtures and build source')
    names = set(old['binaries']) - {'strata'}
    require(len(names) == 5 and set(row['binaries']) == set(old['binaries']) and all(
        (row['binaries'][n]['sha256'],row['binaries'][n]['bytes']) ==
        (old['binaries'][n]['sha256'],old['binaries'][n]['bytes']) for n in names),
        'all five actual same-build fixture ELFs must be byte-identical')
    host = old['host_result']
    require(closed(host) and [(v['mode'],v['exit_code'],v['cases']) for v in host['cases']] ==
            [(0,0,28),(1,0,28)] and host['cases'][0]['checksum'] == host['cases'][1]['checksum'],
            'original closed real-pool parity missing')
    device = json.loads(DEVICE.read_text())
    compact = {n:v for n,v in device.items() if n not in ('controller_inputs','commands')}
    compact['controller_inputs'] = dict(files=len(device['controller_inputs']),
        aggregate_sha256=build.signature(device['controller_inputs']))
    require(data['q6_native_half_model_device_v1'] == compact and closed(device) and
            device['binaries'] == old['binaries'] and device['native_source_commit'] == old['source_commit'] and
            device['build_record']['sha256'] == build.host.NATIVE_RECORD_SHA and
            not device['fresh_faults'] and device['boot_unchanged'] and
            [(v['name'],v['native_exit'],v['checks'],v['ownership_closed']) for v in device['results']] ==
            [('submission',0,10,True),('q6-half',0,32,True),('phase-cache',0,16,True)],
            'original exact-fixture device qualification differs')
    collector = json.loads(COLLECTOR.read_text())
    compact = {n:v for n,v in collector.items() if n not in ('sources','environment_template')}
    compact.update({n+'_sha256':build.signature(collector[n]) for n in ('sources','environment_template')})
    require(data['q6_native_half_collector_active_v1'] == compact and closed(collector) and
            collector['binary'] == old['binaries']['sycl_submission_test'] and
            collector['engine'] == old['binaries']['strata'] and len(collector['cases']) == 1 and
            collector['cases'][0]['native_exit_code'] == 0 and
            collector['cases'][0]['trace']['expected_kernels'] ==
            collector['cases'][0]['trace']['observed_kernels'] == 62 and
            collector['cases'][0]['trace']['native_host_flow_joins_unique'], 'original62-kernel collector differs')
    window = json.loads(WINDOW.read_text())
    compact = {n:v for n,v in window.items() if n not in ('sources','providers','environment_template')}
    compact['inputs'] = {n:dict(count=len(window[n]),sha256=build.signature(window[n]))
                         for n in ('sources','providers','environment_template')}
    require(data['compat_collector_window_check_v1'] == compact and closed(window) and
            [v['mode'] for v in window['cases']] == ['smoke','prefix','window'] and all(
                closed(v) and v['native_exit_code'] == 0 and v['numerical_guards_passed'] and
                not v['fresh_faults'] and v['boot_unchanged'] for v in window['cases']) and
            [v['trace']['recorded_kernels'] for v in window['cases'][1:]] == [4,10] and all(
                v['trace']['unique_native_host_flow_joins'] and v['trace']['gap_and_suffix_excluded']
                for v in window['cases'][1:]), 'original drained-window qualification differs')
    manifest = device['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'runtime manifest changed')
    files = json.loads(Path(manifest['path']).read_text())
    require(len(files) == manifest['files'] and window['providers'] == files and
            all(digest(p) == h for p,h in files.items()), 'runtime provider changed')
    for v in (collector,window):
        require(v['native_source_commit'] == old['source_commit'] and
                v['boot'] == device['boot'] and v['driver'] == device['driver'] and
                v['native_validation_probe'] == device['native_validation_probe'], 'component environment differs')
    for asset in device['native_validation_probe'].values():
        require(digest(asset['path']) == asset['sha256'], 'validation preload/provider changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == device['boot'] and
            base.guards.driver_guard() == device['driver'], 'boot/driver changed')
    assets = profile.collector_assets(data)
    require(assets == collector['assets'] == window['assets'], 'collector tool bytes or source changed')
    if recover_controls:
        recover(old['source_commit'],old['controller_inputs'])
        recover(device['source_commit'],device['controller_inputs'])
        recover(collector['source_commit'],collector['sources'])
        recover(window['source_commit'],window['sources'])
    proof = dict(passed=True,unchanged_native_files=566,byte_identical_fixtures=5,
        fixture_sha256={n:row['binaries'][n]['sha256'] for n in sorted(names)},
        original_native_record_sha256=build.host.NATIVE_RECORD_SHA,
        original_device_record_sha256=digest(DEVICE),original_collector_record_sha256=digest(COLLECTOR),
        original_window_record_sha256=digest(WINDOW),runtime_manifest=manifest,
        original_control_count=len(old['controller_inputs'])+len(device['controller_inputs'])+
            len(collector['sources'])+len(window['sources']),
        original_controls_recovered=True,CPU_pool_cases_per_mode=28,
        device_checks=[10,32,16],collector_kernels=62,drained_window_recorded_kernels=[4,10],
        boot_unchanged=True,driver_unchanged=True,GPU_executed=False,model_executed=False,
        scope='Execution reuse only for five byte-identical fixtures and unchanged native/runtime/collector inputs. Original outcomes remain original;no new component GPU execution,changed-generator/model gate,coverage,speed,physical saturation,teardown or full-context qualification.')
    return row,device,files,assets,proof


def native():
    row,device,files,assets,proof = evidence()
    saved = row.get('component_reuse',{})
    require({n:v for n,v in saved.items() if n not in ('source_commit','controller_inputs')} == proof,
            'commit the exact limited component-reuse proof first')
    pins = base.controls(saved['source_commit'],controller=SELF)
    require(saved['controller_inputs'] == dict(files=len(pins),aggregate_sha256=build.signature(pins)),
            'component-reuse controller changed')
    return row,device,files,assets


def reuse():
    marker = build.ROOT / 'component-reuse-attempt.json'
    require(not marker.exists(), 'one reuse attempt only')
    revision = base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    pins = base.controls(revision,controller=SELF)
    with marker.open('x') as stream: json.dump(dict(source_commit=revision,passed=False),stream)
    started = time.monotonic()
    try:
        row,device,files,assets,proof = evidence(recover_controls=True)
        require(time.monotonic()-started < 180, 'finite component recovery deadline')
        proof.update(source_commit=revision,
            controller_inputs=dict(files=len(pins),aggregate_sha256=build.signature(pins)))
        row['component_reuse'] = proof; build.save(row)
    except BaseException as error:
        write(marker,dict(source_commit=revision,passed=False,complete=False,error=repr(error)))
        raise
    write(marker,dict(source_commit=revision,passed=True,complete=True,GPU_executed=False))
    print('Exact five-fixture/runtime qualification reused;changed generator still needs bounded model gates.',flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] == ['reuse'], 'use reuse')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); base.guards.no_model_owners([])
        reuse()
