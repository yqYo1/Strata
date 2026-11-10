#!/usr/bin/env python3
"""Qualify the same-build cached-subgroup long-prefill candidate.

Main-authored; repository LICENSE applies. Reuse owned supervision, not old GPU
results. Keep this observer separate from the original compiled source.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_qsa_cached_model as build
import check_qsa16_stream_model as original
import qsa_cached_subgroup as component

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'qsa-cached-subgroup-model-device-v1', 'qsa_cached_subgroup_model_device_v1'
THIS = 'tools/sycl/check_qsa_cached_model.py'
STAGES = (*original.STAGES, 'qsa-small', 'qsa-full', 'producer-small', 'producer-full')
require, sha = build.require, build.sha


def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers, 'this observer must be pinned')
    row = json.loads((build.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
        and row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'committed closed eight-target build required')
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
        and actual['translation_units'] == 131 and actual['uncovered'] == 0
        and len(row['planned_compilation']) == 132
        and row['duplicate_source_admission'] == build.DUPLICATE_TARGETS
        and set(row['binaries']) == set(build.TARGETS), 'exact 131-source/132-object eight-target census required')
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
    require(recipe == row['recipe'] and original.common.HOST_PYTHON == seed['host_python']
        and sha(original.common.HOST_PYTHON) == seed['host_python_sha256'], 'exact seed relocation required')
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
    require(build.component_reference() == row['producer_component'], 'original eligible component changed')
    proof = dict(native_source_commit=row['native_source_commit'],original_source_files=len(before),
        observer_source_files=len(after),observer_source_sha256=original.common.signature(after),
        added_observers={p:after[p] for p in sorted(observers)},exact_eight_ELFs_census_recipe=True,
        exact_seed_recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(),
        native_recompiled=False,GPU_result_inherited=False)
    require(expected is None or expected == proof, 'native observer admission changed')
    return dict(row,observer_admission=proof)


def context():
    spec = importlib.util.spec_from_file_location('qsa_cached_native_checks', original.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.NATIVE, ctx.NATIVE_KEY, ctx.TARGETS = build.ROOT, build.KEY, build.TARGETS
    ctx.STAGES, ctx.native, ctx.host = STAGES, native, host
    return ctx


def host(device, ctx, row, scope, outcome):
    env = dict(row['environment']); env.pop('LD_PRELOAD',None)
    e,out,err = scope.run('reader-controls',[original.common.HOST_PYTHON,'-B','-m','unittest','-v',
        'tools.sycl.test_q6_dpas_component','tools.sycl.test_compile_census',
        'tools.sycl.test_q6_head_generation_speed'],env,P,**row['host_budget'])
    text = out.read_text()+err.read_text()
    require(original.owner.closed(e) and e['exit_code'] == 0 and 'Ran 54 tests' in text
        and '\nOK\n' in text and 'skipped=' not in text, 'all54 current reader/census controls required')
    cases = [('empty',['--host'],{},'HOST owner_negative_cases=3 GPU=0\nPACKED_HOST owner_negative_cases=1 GPU=0')]
    for label,changes,message in (
        ('flag3',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'3'},'must be 0 or 1'),
        ('empty-flag',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':''},'must be 0 or 1'),
        ('release0',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':'0'},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('release-absent',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':None},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('packed-release0',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':'0'},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('packed-release-absent',{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':None},'requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('backing3',{'STRATA_PREFILL_HEAD_HOST_BACKING':'3'},'STRATA_PREFILL_HEAD_HOST_BACKING must be 0 or 1'),
        ('backing-empty',{'STRATA_PREFILL_HEAD_HOST_BACKING':''},'STRATA_PREFILL_HEAD_HOST_BACKING must be 0 or 1'),
        ('backing-release0',{'STRATA_PREFILL_HEAD_HOST_BACKING':'1','STRATA_PREFILL_RELEASE_HEADS':'0'},'host backing requires STRATA_PREFILL_RELEASE_HEADS=1'),
        ('backing-release-absent',{'STRATA_PREFILL_HEAD_HOST_BACKING':'1','STRATA_PREFILL_RELEASE_HEADS':None},'host backing requires STRATA_PREFILL_RELEASE_HEADS=1')):
        cases.append((label,['--reject-switch',message],changes,'REJECT_SWITCH passed=1 GPU=0'))
    for label,args,changes,expected in cases:
        child = dict(env)
        for k,v in changes.items():
            if v is None: child.pop(k,None)
            else: child[k] = v
        e,out,err = scope.run(label,[row['binary']['path'],*args],child,ROOT,**row['host_budget'])
        require(original.owner.closed(e) and e['exit_code'] == 0 and out.read_text().strip() == expected
            and not re.search(r'--->\s*ur[A-Z]|<---\s*ur[A-Z]',err.read_text()),
            'CPU owner/switch refusal:'+label)
    e,out,err = scope.run('phase-shape-policy',[row['binaries'][original.incumbent.POLICY]['path']],
        env,P,**row['host_budget'])
    require(original.owner.closed(e) and e['exit_code'] == 0
        and out.read_text() == original.incumbent.POLICY_OUT, 'phase/shape/cache CPU policy failed')
    n = native(row['native_observer_admission'])
    reference = json.loads(original.candidate.SEED.read_text())
    images = {}
    for label,binary in (('engine',n['binaries']['strata']),('fixture',row['binary']),
            ('reference',reference['binaries']['strata'])):
        images[label] = original.common.structure(binary,original.canonical_image(Path(binary['path']).read_bytes()))
    require(images['engine'] == images['fixture'] == images['reference'], 'selected decode head module changed')
    outcome.update(passed=True,native_exit_code=0,GPU_executed=False,reader_cases=54,native_cases=len(cases),
        policy_check=dict(passed=True,GPU_executed=False,native_exit_code=0),
        selected_head_IR=dict(images['engine'],same_as_current_reference=True,engine_fixture_identical=True,
            scope='Exact selected linked-module structure,not final ISA or model speed.'),
        diagnostics_controls=component.diagnostic_controls())


def prepare():
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['plan'].update(order=list(STAGES),
        host='54 current CPU reader/census cases,11 native CPU-only refusal processes,phase/shape/cache policy,exact selected head module against independent decode.',
        attention='After the five owner/submission/phase/head stages,fresh traced numerical and benchmark smokes,then11 independent BF16 numerical/masking/guard cases including actual262144 live cells.',
        producer='Fresh off/on bulk-mirror processes for small producer and actual262144 full-fill plus last-cell overwrite;independent existing oracle,all5partial-page offsets and exact accepted bulk receipts.',
        next='Then separately committed actual long/short/long model handoff and same-binary32K/64K repeated PP/TTFT plus independent decode preservation. Component/full producer success is not physical262144 model/RESTORE/task or throughput qualification.')
    row['limits'] = 'Fresh same-build native/component evidence only. Full-model physical262144/RESTORE,task quality,matched PP/TG/TTFT and adoption remain unqualified.'
    ctx.context(row['native_observer_admission']).save(row)
    print('Prepared all9 same-build native/component stages;commit before host.',flush=True)


def extra(stage):
    ctx = context()
    row = json.loads((ROOT/'record.json').read_text()); head = ctx.context(row['native_observer_admission'])
    head.verify(row)
    require(row['stage'] == 'in_progress' and len(row['results']) == STAGES.index(stage)
        and all(v['passed'] for v in row['results']), 'fresh next stage only;no retry')
    root = ROOT/stage; root.mkdir(mode=0o700)
    for name in ('tmp','cache'): (root/name).mkdir(mode=0o700)
    scope = None
    def persist():
        original.common.write(root/'execution.json',dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    cursor = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
        text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor, 'kernel cursor required')
    result = dict(mode=stage,passed=False,complete=False,GPU_executed=False,checks=[])
    try:
        scope = original.owner.Owner(root/'owned',persist)
        if stage == 'qsa-small':
            jobs = [('numerical','sycl_qsa_bf16_prefix_cached_subgroup_test',['--smoke'],None),
                    ('fixture','sycl_qsa_bf16_prefix_cached_subgroup_bench',['--smoke'],None)]
        elif stage == 'qsa-full':
            jobs = [('numerical','sycl_qsa_bf16_prefix_cached_subgroup_test',['--component'],None)]
        else:
            jobs = [('producer-'+flag,'sycl_native16_qsa_test',
                ['--streamed-full-extent'] if stage == 'producer-full' else [],flag) for flag in ('0','1')]
        for name,target,args,flag in jobs:
            trace = stage in ('qsa-small','producer-small')
            env = dict(row['environment'],STRATA_TEST_MAIN_GATES='0',STRATA_TRACE='1',
                LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
                TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/(name+'-validation.log')))
            if not trace: env.pop('UR_LOG_TRACING',None)
            if flag is not None: env['STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR'] = flag
            result['native_execution_attempted'] = True
            e,out,err = scope.run(name,[row['binaries'][target]['path'],*args],env,P,
                wall=240,cpu=720,rss_cap=4<<30,text_cap=64<<20,file_cap=64<<20,total_cap=80<<20)
            text,errors = out.read_text(),err.read_text()
            proof = original.validation.preload_evidence(errors,e['pid'])
            component.diagnostics(errors,e['pid'])
            require(original.owner.closed(e) and e['exit_code'] == 0
                and proof['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'normal validated component closure required')
            log = root/(name+'-validation.log')
            require(not log.exists() or log.stat().st_size == 0, 'runtime parameter validation warning')
            if trace: require('urEnqueue' in errors, 'actual traced GPU submission missing')
            check = dict(name=name,exit_code=0,ownership_closed=True,validation=proof)
            if flag is None:
                require('fixture_status=PASS' in text, 'QSA fixture incomplete')
                cases = [s for s in text.splitlines() if s.startswith('case ')]
                if name == 'numerical':
                    require(sum(s.startswith('case cells=') for s in cases) == (2 if trace else 11),
                        'full independent numerical case count required')
                    if not trace: require(any(s.startswith('case cells=262144 live_cells=262144 ')
                        for s in cases), 'actual final physical cell not exercised')
                check['cases'] = cases
            else:
                marker = ('QSA16 streamed KV producer physical262144 component PASS; attention/model/RESTORE/performance unqualified'
                    if stage == 'producer-full' else
                    'QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified')
                require(text.splitlines() == [marker], 'complete independent producer oracle required')
                receipts = [tuple(map(int,v)) for v in re.findall(
                    r'^strata QSA mirror bulk: tokens=(\d+) pos0=(\d+) segments=(\d+) copies=(\d+) requested_D2H_bytes=(\d+) queue=same accepted=1 host_quantization=0$',errors,re.M)]
                require(bool(receipts) == (flag == '1'), 'actual bulk route differs')
                for tokens,pos,segments,copies,payload in receipts:
                    require(0 < tokens <= 8192 and pos+tokens <= 262144 and 0 < segments <= 5
                        and copies == 4*segments and payload == tokens*1056, 'accepted bulk extent differs')
                cells,copies,payload = (sum(v[i] for v in receipts) for i in (0,3,4))
                if stage == 'producer-full' and flag == '1':
                    require(cells == 262145 and len(receipts) == 33 and copies == 136
                        and payload == 262145*1056, 'full fill/final-cell overwrite receipt differs')
                check.update(flag=flag,physical_cells=262144 if stage == 'producer-full' else 24,
                    bulk_calls=len(receipts),accepted_copy_commands=copies,requested_D2H_bytes=payload)
            result['checks'].append(check)
        head.verify(row)
        result.update(passed=True,complete=True,native_exit_code=0,host_probe_accepted=True,GPU_executed=True)
    except BaseException as error:
        result.update(passed=False,error=repr(error))
    finally:
        original.validation.finish(result,scope,cursor,row['boot'])
        result['passed'] = bool(result['passed'] and result['complete']
            and original.guards.driver_guard() == row['driver'])
        row['results'].append(result); row['commands'] += scope.commands if scope else []
        row.update(stage=('closed' if stage == STAGES[-1] else 'in_progress') if result['passed'] else 'failed',
            active=bool(scope and scope.active),ownership_closed=result['ownership_closed'],GPU_executed=True)
        row['complete'] = row['stage'] == 'closed'; row['passed'] = row['complete'] and row['ownership_closed']
        head.save(row)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(result['passed'],result.get('error','same-build component gate failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare',*STAGES), 'prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] in original.STAGES: context().execute(sys.argv[1])
        else: extra(sys.argv[1])
