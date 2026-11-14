#!/usr/bin/env python3
"""Fresh qualification for the host-only MoE census successor.

Main-authored from the owned native-gate controller;repository LICENSE applies.
No previous GPU result or model qualification is inherited.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import build_moe_census_model as build
import check_qsa_cached_model as legacy

P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=B/'moe-host-census-device-v1','moe_host_census_device_v1'
THIS='tools/sycl/check_moe_census_model.py'
STAGES=legacy.STAGES
original,component=legacy.original,legacy.component
require,sha=build.require,build.sha

def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers, 'this observer must be pinned')
    row = json.loads((build.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
        and row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'committed closed nine-target build required')
    pin = row['project']
    before = pin['source_sha256']
    after = original.common.source.pin(P, original.common.PREFIXES)['source_sha256']
    require(before == {p:hashlib.sha256(raw).hexdigest() for p,raw in
        original.common.source.blobs(P, pin['revision'], before)}
        and set(after)-set(before) == observers and not set(before)-set(after)
        and all(after[p] == h for p,h in before.items()),
        'only explicitly declared committed observers may extend compiled inputs')
    actual = row['actual_inputs']
    require(row['native_source_commit'] == pin['revision'] == actual['source_commit']
        and actual['translation_units'] == 132 and actual['uncovered'] == 0
        and len(row['planned_compilation']) == 133
        and row['duplicate_source_admission'] == build.base.DUPLICATE_TARGETS
        and set(row['binaries']) == set(build.TARGETS), 'exact 132-source/133-object nine-target census required')
    for dep in row['dependencies']:
        require(dep['source_sha256'] == {p:hashlib.sha256(raw).hexdigest() for p,raw in
            original.common.source.blobs(Path(dep['root']), dep['revision'], dep['source_sha256'])}
            and all(sha(Path(dep['root'])/p) == h for p,h in dep['source_sha256'].items()),
            'original dependency source changed')
    require(sha(build.context().SEED) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p,h in row['toolchain'].items())
        and all(sha(p) == h for p,h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'compiler/SDK/recipe inputs changed')
    seed = json.loads(build.context().SEED.read_text())
    recipe = copy.deepcopy(seed['recipe'])
    old = str(build.context().SEED.parent)
    recipe['configure'] = [str(build.ROOT)+v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S')+1] = str(P)
    recipe['environment'] = {k:v.replace(old,str(build.ROOT)) for k,v in recipe['environment'].items()}
    recipe['environment']['STRATA_SYCL_PREFILL_QSA_STREAM'] = '1'
    recipe['targets'] = recipe['build_targets'] = build.TARGETS
    python_seed = json.loads(original.candidate.SEED.read_text())
    require(recipe == row['recipe'] and original.common.HOST_PYTHON == python_seed['host_python']
        and sha(original.common.HOST_PYTHON) == python_seed['host_python_sha256'], 'exact seed relocation required')
    for item in (*row['preserved_binaries'], *row['binaries'].values()):
        require(sha(item['path']) == item['sha256']
            and Path(item['path']).stat().st_size == item['bytes'], 'retained or same-build ELF changed')
    require(original.candidate.probe_qualification() ==
        (row['native_validation_probe'],row['validation_qualification']), 'runtime probe qualification changed')
    require(all(sha(p) == h for p,h in (row['configured_files']|row['generated_inputs']).items())
        and sha(actual['path']) == actual['sha256'], 'configured/generated/census inputs changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled','generated','links')
        for p,h in census[family].items()), 'actual compiler/header/link input changed')
    require(build.base.component_reference() == row['producer_component'], 'original eligible component changed')
    proof = dict(native_source_commit=row['native_source_commit'],original_source_files=len(before),
        observer_source_files=len(after),observer_source_sha256=original.common.signature(after),
        added_observers={p:after[p] for p in sorted(observers)},exact_nine_ELFs_census_recipe=True,
        exact_seed_recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(),
        native_recompiled=False,GPU_result_inherited=False)
    require(expected is None or expected == proof, 'native observer admission changed')
    return dict(row,observer_admission=proof)

def context():
    spec=importlib.util.spec_from_file_location('moe_census_native_gate_driver',legacy.__file__)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.ROOT,driver.KEY,driver.THIS=ROOT,KEY,THIS
    driver.build,driver.native=build,native
    original_host=driver.host
    def host(device,ctx,row,scope,outcome):
        env=dict(row['environment']);env.pop('LD_PRELOAD',None)
        binary=row['binaries']['sycl_moe_upload_census_host_test']
        e,out,err=scope.run('census-host',[binary['path']],env,P,**row['host_budget'])
        expected='MoE census host controls passed: literal upload/hit/byte/format bins;invalid rows;empty frame;512expert bound. No GPU qualification.\n'
        require(original.owner.closed(e) and e['exit_code']==0 and out.read_text()==expected
            and not err.read_text(),'literal host-census contract failed')
        original_host(device,ctx,row,scope,outcome)
        outcome['host_census_controls']=dict(passed=True,native_exit_code=0,
            GPU_executed=False,source='tests/sycl/moe_upload_census_host.cpp',
            scope='Literal miss/hit/format/requested-byte bins,invalid rows,empty frame and512expert cap;not GPU/model correctness.')
    driver.host=host
    return driver


def prepare():
    driver=context();ctx=driver.context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    row['environment'].update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0')
    row['previous_census_failure']=dict(key='moe_job_geometry_v1',
        evidence_commit='237f6390fc64081d8a02bc4c6ba9f00f24807eee',
        native_exit_code=-6,watchdog_seconds=480,completed_requests=0,
        original_ownership_closed=False,physical_session_empty_observed=True,
        GPU_health_inherited=False)
    row['plan'].update(order=list(STAGES),
        host='First the new CPU-only literal census contract,then54current reader/census controls,11native CPU-only refusals,phase/shape/cache policy and exact selected decode module. No GPU in this stage.',
        submission='First GPU execution after the failed timed observer is one fresh bounded full-trace/parameter-validation submission control. Stop on any fault,owner,validation,budget or correctness error;no automatic retry or model execution.',
        attention='Fresh small/full numerical oracles including262144live cells;no old result inheritance.',
        producer='Fresh off/on small/full producer gates including last-cell overwrite,with immutable safety and startup settings.',
        next='Only after all gates pass,commit a separate small long/short/long host-census model check with LM_PROFILE=0. Then representative32K/64Kcounts. Do not rerun timing-enabled V1 or infer performance/health from compilation.')
    row['limits']='Fresh native/component qualification only. The earlier CPU spin has no proven root cause. Model phase preservation,full262144/RESTORE,task/agent,sustainedTG,clean PP and adoption remain open.'
    ctx.context(row['native_observer_admission']).save(row)
    print('Prepared host-only census native gates;commit before CPU/GPU execution.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare',*STAGES),'prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1] in original.STAGES:context().context().execute(sys.argv[1])
        else:context().extra(sys.argv[1])
