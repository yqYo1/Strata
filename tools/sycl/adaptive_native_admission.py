#!/usr/bin/env python3
"""Reuse exact unchanged fixture qualification for the adaptive audit engine.

Main-authored; repository LICENSE. The generator still needs its own model
checks. No GPU or native process is launched by this identity/recovery check.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys

sys.dont_write_bytecode = True
import build_adaptive_audit_model as build

P,B,RESULT,base = build.P,build.B,build.RESULT,build.base
SELF = Path(__file__).resolve()
DEVICE = B / 'q6-native-half-device-v1/record.json'
DEVICE_SHA = 'a174e7de45b8b2ffb5317592fb12606c934babcc184817973c0d46805b3dbce9'
require,digest,write = build.require,build.digest,build.write


def closed(value):
    return value['passed'] and value['complete'] and value['ownership_closed'] and not value.get('active',False)


def compiled():
    row=json.loads((build.ROOT/'record.json').read_text())
    require(base.committed(RESULT)[build.KEY]==build.compact(row) and closed(row) and
            row['stage']=='built-not-executed' and row['native_recipe_checks']['passed'] and
            all(base.supervision.closed(v) and v['exit_code']==0 for v in row['commands_executed']),
            'commit exact closed native audit build first')
    require(base.controls(row['source_commit'],controller=build.SELF)==row['controller_inputs'],
            'native build controls changed')
    original,values=build.basis()
    require(build.selected.pins(values)==row['native_sources'] and all(
        (build.ROOT/'source'/name).read_bytes()==raw for name,raw in values.items()), 'native568source changed')
    for value in row['binaries'].values():
        require(digest(value['path'])==value['sha256'] and Path(value['path']).stat().st_size==value['bytes'],
                'actual native engine/fixture changed')
    for key in ('compiled_inputs','compile_database'):
        require(digest(row[key]['path'])==row[key]['sha256'], 'closed compiler evidence changed')
    actual=json.loads(Path(row['compiled_inputs']['path']).read_text())
    require(len(actual)==row['compiled_inputs']['files'] and
            actual.get(str(build.ROOT/'source'/build.source.HEADER))==build.source.HEADER_SHA,
            'actual audit header/dependency closure changed')
    return row,original


def evidence(recover=False):
    data=base.committed(RESULT);row,old=compiled()
    unchanged={n:h for n,h in old['native_sources'].items() if n!=build.source.GENERATOR}
    require(len(unchanged)==566 and all(row['native_sources'].get(n)==h for n,h in unchanged.items()) and
            set(row['native_sources'])-set(old['native_sources'])=={build.source.HEADER}, 'native kernel/runtime drift')
    names=set(old['binaries'])-{'strata'}
    require(len(names)==5 and set(row['binaries'])==set(old['binaries']) and all(
        (row['binaries'][n]['sha256'],row['binaries'][n]['bytes'])==
        (old['binaries'][n]['sha256'],old['binaries'][n]['bytes']) for n in names), 'five exact fixture ELFs required')
    host=old['host_result']
    require(closed(host) and [(v['mode'],v['exit_code'],v['cases']) for v in host['cases']]==[(0,0,28),(1,0,28)] and
            host['cases'][0]['checksum']==host['cases'][1]['checksum'], 'original real-pool parity differs')
    require(digest(DEVICE)==DEVICE_SHA, 'original exact-fixture device record changed')
    device=json.loads(DEVICE.read_text())
    compact={n:v for n,v in device.items() if n not in ('controller_inputs','commands')}
    compact['controller_inputs']=dict(files=len(device['controller_inputs']),
        aggregate_sha256=build.signature(device['controller_inputs']))
    require(data['q6_native_half_model_device_v1']==compact and closed(device) and
            device['binaries']==old['binaries'] and device['native_source_commit']==old['source_commit'] and
            device['build_record']['sha256']==build.host.NATIVE_RECORD_SHA and not device['fresh_faults'] and
            device['boot_unchanged'] and [(v['name'],v['native_exit'],v['checks'],v['ownership_closed'])
                for v in device['results']]==[('submission',0,10,True),('q6-half',0,32,True),('phase-cache',0,16,True)],
            'original component device qualification differs')
    manifest=device['runtime_manifest']
    require(digest(manifest['path'])==manifest['sha256'], 'runtime manifest changed')
    providers=json.loads(Path(manifest['path']).read_text())
    require(len(providers)==manifest['files'] and all(digest(p)==h for p,h in providers.items()), 'runtime provider changed')
    for value in device['native_validation_probe'].values():
        require(digest(value['path'])==value['sha256'], 'pre-main validation probe changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==device['boot'] and
            base.guards.driver_guard()==device['driver'], 'boot/driver changed')
    if recover:
        for value,key in ((old,'controller_inputs'),(device,'controller_inputs')):
            pins={n:hashlib.sha256(raw).hexdigest() for n,raw in base.basis.build.blobs(P,value['source_commit'],value[key])}
            require(pins==value[key], 'original qualification controls not recoverable')
    proof=dict(passed=True,unchanged_native_files=566,byte_identical_fixtures=5,
        fixture_sha256={n:row['binaries'][n]['sha256'] for n in sorted(names)},
        original_native_record_sha256=build.host.NATIVE_RECORD_SHA,original_device_record_sha256=DEVICE_SHA,
        original_controls_recovered=True,original_control_count=len(old['controller_inputs'])+len(device['controller_inputs']),
        CPU_pool_cases_per_mode=28,device_checks=[10,32,16],host_audit_header_cases=41,runtime_manifest=manifest,
        boot_unchanged=True,driver_unchanged=True,GPU_executed=False,model_executed=False,
        scope='Exact fixture/unchanged kernel execution reuse only. New generator/audit dispatch,positive copies,loan/refill,model quality/speed,ordinary provider teardown and262144/RESTORE remain unqualified.')
    return row,device,providers,proof


def native():
    row,device,providers,proof=evidence();saved=row.get('component_reuse',{})
    require({n:v for n,v in saved.items() if n not in ('source_commit','controller_inputs')}==proof,
            'commit exact limited component reuse first')
    pins=base.controls(saved['source_commit'],controller=SELF)
    require(saved['controller_inputs']==dict(files=len(pins),sha256=build.signature(pins)), 'reuse source changed')
    return row,device,providers


def reuse():
    marker=build.ROOT/'component-reuse-attempt.json';require(not marker.exists(),'one reuse attempt only')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    pins=base.controls(revision,controller=SELF)
    with marker.open('x') as stream:stream.write('{}\n')
    try:
        row,device,providers,proof=evidence(recover=True)
        proof.update(source_commit=revision,controller_inputs=dict(files=len(pins),sha256=build.signature(pins)))
        row['component_reuse']=proof;build.save(row)
    except BaseException as error:write(marker,dict(passed=False,complete=False,error=repr(error)));raise
    write(marker,dict(passed=True,complete=True,GPU_executed=False))
    print('Five byte-identical fixtures reuse unchanged component qualification;new native audit still pending.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:]==['reuse'],'use reuse')
    resource.setrlimit(resource.RLIMIT_CPU,(150,150));signal.alarm(180)
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([]);reuse()
