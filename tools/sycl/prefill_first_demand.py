#!/usr/bin/env python3
"""Measure exact prefix expert demand from an already completed prefill window.

Main-authored from the owned window build and request controller. Repository LICENSE.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import moe_upload_census as geometry
import prefill_window_build as builder
import prefill_workspace_admission as workspace

P, B, RESULT = basis.P, basis.B, basis.RESULT
BUILD_ROOT, BUILD_KEY = B/'prefill-first-demand-native-v1', 'prefill_first_demand_native_v1'
HOST_ROOT, HOST_KEY = B/'prefill-first-demand-host-v2', 'prefill_first_demand_host_v2'
ROOT, KEY = B/'prefill-first-demand-v1', 'prefill_first_demand_v1'
THIS = 'tools/sycl/prefill_first_demand.py'
OVERLAYS = ('include/strata/prefill/moe_upload_census.hpp', 'src/prefill/moe_window.inc',
            'tests/sycl/moe_upload_census_host.cpp')
HOST_TARGET = 'sycl_moe_upload_census_host_test'
TARGETS = [*builder.TARGETS, HOST_TARGET]
READS, RUNS = (32769, 65537), 2
source, write = builder.source, builder.write
require, sha, signature = basis.require, basis.sha, basis.signature
PYTHON = basis.reference.model.PYTHON


def private_module(original, name):
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    return ctx


def historical():
    native, proof, speed = basis.basis()
    prior = basis.validation.committed(RESULT, ('prefill_window_native_v2',))
    profile = basis.validation.committed(RESULT, ('eager_request_profile_v2',))
    staging = basis.validation.committed(RESULT, ('fullfit_attention_arena_source_v1',))
    build_summary = prior['source_files']
    require(prior['stage'] == 'built' and prior['ownership_closed'] and not prior['active']
        and prior['native_base_commit'] == basis.original.SOURCE
        and prior['binaries']['strata'] == profile['binary']
        and profile['native_source_commit'] == prior['source_commit']
        and profile['native_composition_sha256'] == build_summary['sha256']
        and profile['closed_capture_retirement']['complete']
        and staging['artifact_retirement']['complete']
        and staging['decision']['source_comparisons']['staged_vs_allocated']['eligible']
        and not staging['decision']['eligible_borrowed_layer_source'],
        'archive starting image and isolated staging decision required')
    return native, proof, speed, prior


def build_context():
    ctx = private_module(builder, 'first_demand_build')
    ctx.ROOT, ctx.KEY, ctx.TARGETS = BUILD_ROOT, BUILD_KEY, TARGETS
    original_save = ctx.save
    def save(row):
        if row['stage'] == 'configured':
            build = BUILD_ROOT/'build'
            paths = [build/name for name in ('CMakeCache.txt','build.ninja','compile_commands.json')]
            paths += sorted((build/'CMakeFiles').glob('*.ninja'))
            row['configured_files'] = {str(p):sha(p) for p in paths}
        original_save(row)
    ctx.check, ctx.save = check_build, save
    return ctx


def recover_composition(native, prior, revision):
    payload = dict(source.blobs(P, basis.original.SOURCE, native['project']['source_sha256']))
    overlays = dict(source.blobs(P, prior['source_commit'], prior['overlays']))
    require({p:hashlib.sha256(raw).hexdigest() for p,raw in overlays.items()} == prior['overlays'],
        'historic native overlays do not match')
    payload.update(overlays)
    hashes = {p:hashlib.sha256(raw).hexdigest() for p,raw in payload.items()}
    require(dict(files=len(hashes),sha256=signature(hashes)) == prior['source_files'],
        'recover the exact profiled 64K native base before applying observations')
    changed = dict(source.blobs(P, revision, OVERLAYS))
    require(set(changed) == set(OVERLAYS) and all(p in payload and payload[p] != raw
        and (P/p).read_bytes() == raw for p,raw in changed.items()), 'commit the observer and host control')
    payload.update(changed)
    return payload, {p:hashlib.sha256(raw).hexdigest() for p,raw in payload.items()}


def recipe(prior):
    old = str(Path(prior['binaries']['strata']['path']).parent.parent)
    original = prior['recipe']
    return dict(configure=[v.replace(old,str(BUILD_ROOT)) for v in original['configure']],
        environment={k:v.replace(old,str(BUILD_ROOT)) for k,v in original['environment'].items()},targets=TARGETS)


def check_build(row, committed=True, controller_revision=None):
    if committed:
        require(basis.validation.committed(RESULT, (BUILD_KEY,)) == build_context().compact(row),
            'commit the exact native build stage')
    native, proof, _, prior = historical()
    original_controls = row['controller_sources']
    expected = dict(original_controls)
    if controller_revision is not None:
        recovered = {p:hashlib.sha256(raw).hexdigest() for p,raw in
                     source.blobs(P,row['source_commit'],original_controls)}
        require(recovered==original_controls and THIS in expected, 'recover original native controller closure')
        raw = source.git(P,'show',controller_revision+':'+THIS)
        expected[THIS] = hashlib.sha256(raw).hexdigest()
        require(expected!=original_controls,'declare the changed controller-only admission')
    require(row['foundation'] == proof and row['native_base_receipt'] == signature(prior)
        and row['driver'] == builder.guards.driver_guard() and row['recipe'] == recipe(prior)
        and row['retained_decode_reference'] == builder.decode_reference()
        and source.pin(P,list(expected))['source_sha256'] == expected,
        'frozen native/archive/controller/phase reference changed')
    require(all(sha(BUILD_ROOT/'source'/p) == h for p,h in row['source_files'].items())
        and all(sha(Path(d['root'])/p) == h for d in native['dependencies'] for p,h in d['source_sha256'].items())
        and all(sha(p)==h for p,h in row.get('configured_files',{}).items()),
        'actual materialized source, dependency or configured input changed')
    return native


def prepare_build():
    require(not BUILD_ROOT.exists() and BUILD_KEY not in json.loads(RESULT.read_text()), 'fresh native root only')
    native, proof, _, prior = historical()
    revision = source.git(P,'rev-parse','HEAD').decode().strip()
    payload, hashes = recover_composition(native, prior, revision)
    decode = builder.decode_reference(recover=True)
    BUILD_ROOT.mkdir(mode=0o700)
    for name in ('source','tmp','cache','owned'):
        (BUILD_ROOT/name).mkdir(mode=0o700)
    for name,raw in payload.items():
        target = BUILD_ROOT/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    row = dict(stage='prepared',source_commit=revision,source_files=hashes,foundation=proof,
        controller_sources=source.pin(P,[*builder.PREFIXES,*OVERLAYS])['source_sha256'],
        native_base_commit=prior['source_commit'],native_base_receipt=signature(prior),
        overlays={p:hashes[p] for p in OVERLAYS},recipe=recipe(prior), driver=builder.guards.driver_guard(),
        external_inputs=native['external_inputs'],retained_decode_reference=decode,
        commands=[],active=False,ownership_closed=True,passed=False,complete=False,
        plan=dict(unit='Rebuild the exact archived 64K window image and change only the drained host MoE census printer and its control. The observer makes no additional GPU submission, transfer or wait.',
            retention='Temporary third build while both phase references remain available. Measure this decision, then verify native/source recovery and retire the build. No new adoption or speed qualification.'))
    check_build(row,False)
    build_context().save(row)


def save_host(row):
    write(HOST_ROOT/'record.json',row)
    data=json.loads(RESULT.read_text()); data[HOST_KEY]=row; write(RESULT,data)


def host_basis(revision):
    build = json.loads((BUILD_ROOT/'record.json').read_text())
    check_build(build,controller_revision=revision)
    failed, = json.loads((BUILD_ROOT/'host-ownership.json').read_text())
    require(build['stage']=='failed' and build['complete'] and not build['active']
        and not build['host_control']['passed'] and failed['error']=="RuntimeError('artifact cap')"
        and failed['session_empty'] and failed['direct_child_reaped'] and failed['exit_code']==0
        and not failed['survivors'] and not failed['errors'], 'preserve the original observed host-control failure')
    for key in ('compiled','generated','links','objects','produced'):
        require(all(sha(p)==h for p,h in build['actual_inputs'][key].items()),'exact native '+key+' changed')
    require(all(sha(v['path'])==v['sha256'] for v in build['binaries'].values()), 'same-build executable changed')
    return build


def host_prepare():
    require(not HOST_ROOT.exists() and HOST_KEY not in json.loads(RESULT.read_text()),'fresh host qualification only')
    revision=source.git(P,'rev-parse','HEAD').decode().strip()
    build=host_basis(revision)
    HOST_ROOT.mkdir(mode=0o700)
    save_host(dict(stage='prepared',controller_revision=revision,source_commit=build['source_commit'],
        binary=build['binaries'][HOST_TARGET],build_signature=signature(build_context().compact(build)),
        previous_failure='The 4 MiB output cap counted the existing build. The original child was reaped with an empty session, without an accepted output observation.',
        plan='Run the exact built CPU fixture with a separate small owned output root, retaining 4 MiB, and check the unchanged parser assertions.',
        passed=False,ownership_closed=True))


def host_native():
    row=json.loads((HOST_ROOT/'record.json').read_text())
    require(row['stage']=='prepared' and row==basis.validation.committed(RESULT,(HOST_KEY,)),'commit isolated host plan')
    build=host_basis(row['controller_revision']); binary=row['binary']
    require(binary==build['binaries'][HOST_TARGET] and row['build_signature']==signature(build_context().compact(build)),
        'same original compiled host input required')
    scope=builder.ownership.Owner(HOST_ROOT/'owned',lambda:write(HOST_ROOT/'execution.json',scope.commands))
    result,stdout,_ = scope.run('literal',[binary['path']],build['recipe']['environment'],P,
        wall=30,cpu=20,rss_cap=256<<20,text_cap=64<<10,file_cap=1<<20,total_cap=4<<20)
    passed = builder.ownership.closed(result) and result['exit_code']==0 and stdout.read_text().startswith('MoE census host controls passed:')
    if passed:
        try:
            demand_controls()
        except (RuntimeError, ValueError, KeyError) as error:
            row['host_parser_error'] = repr(error)
            passed = False
    row.update(stage='closed' if passed else 'failed',passed=passed,
        ownership_closed=builder.ownership.closed(result),native_gpu_executed=False,
        native_exit_code=result['exit_code'],native_observation_error=result.get('error'))
    save_host(row)
    require(passed,'literal native route and boundary controls failed')


def demand(text, rows, stride_by_layer):
    def frames(label):
        return [json.loads(raw,object_pairs_hook=geometry.unique_object) for raw in
                re.findall(r'^strata MoE '+label+r' census: (.+)$',text,re.M)]
    routes, uploads = frames('first-route'), frames('upload')
    require(len(stride_by_layer)==len(routes)==len(uploads)==48,'one complete routed and drained window per layer')
    points = list(range(8192, rows+1, 8192))
    require(points and points[-1]==rows,'aligned measurement extent required')
    unique, by_piece = [0]*len(points),[0]*len(points)
    total_full = sum(stride_by_layer)*512
    for layer,(route,upload,stride) in enumerate(zip(routes,uploads,stride_by_layer)):
        keys=('version','layer','first_row','rows')
        for frame in (route,upload):
            require(all(type(frame[k]) is int for k in keys)
                and tuple(frame[k] for k in keys)==(1,layer,0,rows),'actual window identity changed')
        pairs = route['experts']
        require(isinstance(pairs,list) and 10<=len(pairs)<=512,'empty or oversized expert demand')
        require(all(isinstance(p,list) and len(p)==2 and type(p[0]) is int and type(p[1]) is int
            and 0<=p[0]<512 and 0<=p[1]<rows for p in pairs),'invalid first-route expert or row')
        experts = [p[0] for p in pairs]
        require(experts==sorted(set(experts)) and sum(r==0 for _,r in pairs)==10,
                'duplicate routes or incorrect first-token fanout')
        require(all(type(upload[k]) is int for k in ('blob_bytes','accepted_uploads','cache_hits','requested_upload_bytes'))
            and upload['final_drain']=='checked' and upload['blob_bytes']==stride
            and upload['accepted_uploads']==len(pairs) and upload['cache_hits']==0
            and upload['requested_upload_bytes']==len(pairs)*stride,'full-fit selected payload not conserved')
        counts = [0]*len(points)
        for _,row in pairs:
            counts[row//8192]+=1
        cumulative=0
        for i,count in enumerate(counts):
            cumulative+=count
            unique[i]+=cumulative
            by_piece[i]+=cumulative*stride
    return dict(points=[dict(rows=p,experts_across_layers=n,requested_payload_bytes=b,
                            selected_payload_fraction=b/by_piece[-1])
                        for p,n,b in zip(points,unique,by_piece)],
        selected_payload_bytes=by_piece[-1],all_expert_payload_bytes=total_full,
        additional_all_expert_bytes=total_full-by_piece[-1],ordered_first_routes_sha256=signature(routes),
        layers=48,actual_early_dispatch=False)


def demand_controls():
    rows=16384
    strides=[1000]*24+[2000]*24
    def fixture(tail_row=8192, tail_id=511, bad_bytes=0):
        lines=[]
        for layer,stride in enumerate(strides):
            window=dict(version=1,layer=layer,first_row=0,rows=rows)
            for label,record in (
                ('first-route',dict(window,experts=[[e,0] for e in range(10)]+[[tail_id,tail_row]])),
                ('upload',dict(window,blob_bytes=stride,accepted_uploads=11,cache_hits=0,
                    requested_upload_bytes=11*stride+(bad_bytes if layer==0 else 0),final_drain='checked')),
            ):
                lines.append('strata MoE '+label+' census: '+json.dumps(record))
        return '\n'.join(lines)
    good=demand(fixture(),rows,strides)
    require([p['requested_payload_bytes'] for p in good['points']]==[720000,792000]
        and good['additional_all_expert_bytes']==501*72000,'prefix boundary or stride weighting changed')
    for text in (fixture(tail_row=rows),fixture(tail_id=9),fixture(bad_bytes=1)):
        try:
            demand(text,rows,strides)
        except RuntimeError:
            continue
        raise RuntimeError('invalid prefix-demand control was accepted')


def model_context():
    ctx = private_module(basis.core, 'first_demand_model')
    ctx.ROOT,ctx.KEY,ctx.THIS = ROOT,KEY,THIS
    ctx.READS,ctx.ORDER,ctx.RUNS,ctx.CLEAN_FAST = READS,[(n,1) for n in READS],RUNS,True
    def reference():
        host=basis.validation.committed(RESULT,(HOST_KEY,))
        build=host_basis(host['controller_revision'])
        require(host==json.loads((HOST_ROOT/'record.json').read_text()) and host['stage']=='closed'
            and host['passed'] and host['ownership_closed'] and host['native_exit_code']==0
            and host['build_signature']==signature(build_context().compact(build)),
            'closed same-build CPU controls required before model')
        _,_,speed,_ = historical()
        old = json.loads((basis.reference.model.ROOT/'record.json').read_text())
        require(old['models']==speed['model_assets'] and old['probe']==speed['native_validation_probe'],
            'coherent model assets/probe changed')
        old['observer_admission']=dict(native_build=signature(build_context().compact(build)),
            fast_configuration_diagnostic=dict(scope='Historical asset/runtime basis only; fresh model placement measured.'))
        native = dict(binaries=build['binaries'],native_source_commit=build['source_commit'])
        return native, old, json.loads(Path(speed['runtime_manifest']['path']).read_text()), None
    old_env, old_current = ctx.environment, ctx.current
    def environment(row,plan):
        env=old_env(row,plan)
        env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='512',
            STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
            STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1', STRATA_SYCL_PREFILL_GU_SWIGLU='large',
            STRATA_SYCL_PREFILL_MOE_CENSUS='1', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
            STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='jit',
            STRATA_TRACE='1')
        return env
    def current(row):
        require(row['stage'] in ('prepared','ready','running') and [p['read'] for p in row['plans']]==list(READS)
            and all(p['bank']==512 and p['window']==10 and p['trace_progress'] and p['upload']=='jit'
                and p['relocate']==0 for p in row['plans']),'frozen prefix demand plan differs')
        return old_current(row)
    def route(text,plan):
        result=workspace.request_route(text,plan)
        pack=Path(plan['native_layout']['path'])
        require(sha(pack)==plan['native_layout']['sha256'],'native model layout changed')
        strides=[int(line.split()[4]) for line in pack.read_text().splitlines()
                 if line and not line.startswith('#')]
        result['prefix_demand']=demand(text,plan['read']-1,strides)
        require(result['prefix_demand']['selected_payload_bytes']==result['requested_bank_H2D_bytes']
            and result['permanent_hits']==0, 'request payload and prefix census differ')
        return result
    def decision(processes):
        require([p['read'] for p in processes]==list(READS)
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                and p['native_exit_code']==0 and not p['fresh_faults'] and len(p['routes'])==RUNS for p in processes),
            'both complete diagnostic processes required')
        return dict(census_complete=True,requests=sum(len(p['routes']) for p in processes),
            workload_scope='Actual full-fit JIT routes at 32K and 64K. Early availability and hidden time require a separate scheduler comparison.',
            speedup_qualified=False)
    def sample(sample,count,read,minimum_ms=10000):
        basis.comparison.check_sample(sample,count,read,minimum_ms)
        require(sample['generated_tokens']==24,'match the profile request output extent')
    ctx.basis,ctx.environment,ctx.current,ctx.request_route = reference,environment,current,route
    ctx.decision,ctx.check_sample=decision,sample
    return ctx


def prepare_model():
    ctx=model_context()
    ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    args=row['configuration']['args']
    layout=Path(args[args.index('--pack')+1])/'native_experts.txt'
    require(sha(layout)==row['model_assets'][str(layout)]['sha256'],'actual pinned native expert layout required')
    for plan in row['plans']:
        plan.update(bank=512,window=10,relocate=0,trace_progress=True,upload='jit',
                    native_layout=dict(path=str(layout),sha256=sha(layout)))
        require(plan['environment_sha256']==basis.comparison.fingerprint(ctx.environment(row,plan)),
                'frozen environment changed')
    row['plan']=dict(question='What fraction of weight demand is exactly identified by the first 8192, 16384 and later prefill rows?',
        order='Fresh 32769-input diagnostic, then 65537-input diagnostic. Two independent nonce requests per process with thinking, no reuse and ordinary full-fit bank512 JIT.',
        checks='One stable expert/first local row pair per active expert and 48 drained windows per request. Match actual selected payload and model uploads. Validate observed source placement, typed QSA, handoff, bounded Level Zero progress and no fresh GPU fault.',
        timing='Observation includes observer expense. No speedup or saturation claim and no new GPU operation.',
        retention='Keep per-piece byte/expert totals, verified source identity and request samples. Retire raw route lists and temporary comparison build after recovery.')
    ctx.save(row)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2, 'use pinned Python and a declared stage')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='build-prepare': prepare_build()
        elif action in ('configure','compile'): build_context().execute(action)
        elif action=='host-prepare': host_prepare()
        elif action=='host-run': host_native()
        elif action=='prepare': prepare_model()
        elif action=='host': basis.comparison.host(model_context())
        elif action in ('0','1'): basis.comparison.run(int(action),model_context())
        else: raise RuntimeError('unknown stage')
