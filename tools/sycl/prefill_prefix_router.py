#!/usr/bin/env python3
"""Qualify the same-queue prefill MoE prefix producer before adding copy overlap.

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
import prefill_window_build as builder
import prefill_workspace_admission as workspace

P, B, RESULT = basis.P, basis.B, basis.RESULT
BUILD_ROOT, BUILD_KEY = B/'prefill-prefix-router-native-v1', 'prefill_prefix_router_native_v1'
ROOT, KEY = B/'prefill-prefix-router-model-v1', 'prefill_prefix_router_model_v1'
THIS = 'tools/sycl/prefill_prefix_router.py'
OBSERVER = ('include/strata/prefill/moe_upload_census.hpp', 'src/prefill/moe_window.inc',
            'tests/sycl/moe_upload_census_host.cpp')
OVERLAYS = ('include/strata/prefill/prefill.hpp', 'src/prefill/prefill.cpp',
            'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc')
HOST_TARGET = 'sycl_moe_upload_census_host_test'
TARGETS = [*builder.TARGETS, HOST_TARGET]
READ, RUNS = 32769, 2
ORDER = ('off', 'late', 'early')
MODE = 'STRATA_SYCL_PREFILL_MOE_PREFIX_ROUTER'
PREFIX_BYTES = 8192 * (2560*2 + 4*4 + 10*4 + 10*4 + 4 + 512*4)
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
    demand = basis.validation.committed(RESULT, ('prefill_first_demand_v1',))
    demand_build = basis.validation.committed(RESULT, ('prefill_first_demand_native_v1',))
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
    require(demand['stage']=='closed' and demand['ownership_closed'] and not demand['active']
        and demand['decision']['requests']==4 and demand['artifact_retirement']['complete']
        and demand_build['source_commit']==demand['native_source_commit']
        and demand_build['binaries']['strata']==demand['binary'],
        'closed measured prefix demand and original observer composition required')
    return native, proof, speed, prior, demand_build


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


def recover_composition(native, prior, demand_build, revision):
    payload = dict(source.blobs(P, basis.original.SOURCE, native['project']['source_sha256']))
    overlays = dict(source.blobs(P, prior['source_commit'], prior['overlays']))
    require({p:hashlib.sha256(raw).hexdigest() for p,raw in overlays.items()} == prior['overlays'],
        'historic native overlays do not match')
    payload.update(overlays)
    hashes = {p:hashlib.sha256(raw).hexdigest() for p,raw in payload.items()}
    require(dict(files=len(hashes),sha256=signature(hashes)) == prior['source_files'],
        'recover the exact profiled 64K native base before applying observations')
    recovered_observer = dict(source.blobs(P, demand_build['source_commit'], OBSERVER))
    require({p:hashlib.sha256(raw).hexdigest() for p,raw in recovered_observer.items()} ==
        demand_build['overlays'], 'restore exact previously measured observer sources')
    payload.update(recovered_observer)
    hashes = {p:hashlib.sha256(raw).hexdigest() for p,raw in payload.items()}
    require(dict(files=len(hashes),sha256=signature(hashes))==demand_build['source_files'],
        'restore exact route-observation build before adding scheduling')
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


def check_build(row, committed=True):
    if committed:
        require(basis.validation.committed(RESULT, (BUILD_KEY,)) == build_context().compact(row),
            'commit the exact native build stage')
    native, proof, _, prior, demand_build = historical()
    require(row['foundation'] == proof and row['native_base_receipt'] == signature(prior)
        and row['driver'] == builder.guards.driver_guard() and row['recipe'] == recipe(prior)
        and row['retained_decode_reference'] == builder.decode_reference()
        and source.pin(P,list(row['controller_sources']))['source_sha256'] == row['controller_sources']
        and row['observer_base_receipt']==signature(demand_build),
        'frozen native/archive/controller/phase reference changed')
    require(all(sha(BUILD_ROOT/'source'/p) == h for p,h in row['source_files'].items())
        and all(sha(Path(d['root'])/p) == h for d in native['dependencies'] for p,h in d['source_sha256'].items())
        and all(sha(p)==h for p,h in row.get('configured_files',{}).items()),
        'actual materialized source, dependency or configured input changed')
    return native


def prepare_build():
    require(not BUILD_ROOT.exists() and BUILD_KEY not in json.loads(RESULT.read_text()), 'fresh native root only')
    native, proof, _, prior, demand_build = historical()
    revision = source.git(P,'rev-parse','HEAD').decode().strip()
    payload, hashes = recover_composition(native, prior, demand_build, revision)
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
        observer_base_receipt=signature(demand_build),
        overlays={p:hashes[p] for p in OVERLAYS},recipe=recipe(prior), driver=builder.guards.driver_guard(),
        external_inputs=native['external_inputs'],retained_decode_reference=decode,
        commands=[],active=False,ownership_closed=True,passed=False,complete=False,
        plan=dict(unit='Reuse the exact previous prefix-demand observer source as the base. Build one binary with off, late and early producer timing. No expert upload policy or queue change.',
            retention='Temporary third build for the prefix equivalence and upcoming timing comparison. Preserve both phase references; retire or select after matched results.'))
    check_build(row,False)
    build_context().save(row)


def host_native():
    row = json.loads((BUILD_ROOT/'record.json').read_text())
    check_build(row)
    require(row['stage']=='built' and row['passed'] and row['ownership_closed'] and not row['active']
        and 'host_control' not in row, 'one owned native host control required')
    binary=row['binaries'][HOST_TARGET]
    require(sha(binary['path'])==binary['sha256'], 'same-build host executable differs')
    root=BUILD_ROOT/'host-control'; root.mkdir(mode=0o700)
    scope=builder.ownership.Owner(root/'owned',lambda:write(root/'execution.json',scope.commands))
    result,stdout,_=scope.run('literal',[binary['path']],row['recipe']['environment'],P,
        wall=30,cpu=20,rss_cap=256<<20,text_cap=64<<10,file_cap=1<<20,total_cap=4<<20)
    passed=builder.ownership.closed(result) and result['exit_code']==0 and stdout.read_text().startswith('MoE census host controls passed:')
    row['host_control']=dict(passed=passed,binary=binary,ownership_closed=builder.ownership.closed(result),
                            exit_code=result['exit_code'],error=result.get('error'))
    if not passed: row.update(stage='failed',passed=False)
    build_context().save(row)
    require(passed,'literal census and context-boundary CPU controls failed')


def router_evidence(text, plan, admitted):
    variant=plan['variant']
    admissions=workspace.records(text,'strata MoE prefix router: ')
    completions=workspace.records(text,'strata MoE prefix router complete: ')
    if variant=='off':
        require(not admissions and not completions,'original router unexpectedly retained a prefix')
    else:
        require(admissions==[dict(requested=variant,effective=variant,
            workspace_bytes=str(PREFIX_BYTES),decode_changed='0')]
            and completions==[dict(effective=variant,layers='48',rows='8192',final_drain='checked')]
            and admitted['fixed_including_headroom_bytes']+PREFIX_BYTES<=admitted['free_gpu_bytes']
            and PREFIX_BYTES<=admitted['allocation_limit_bytes'],'production prefix route did not admit or retire')
    frames=re.findall(r'^strata MoE router census: layer=(\d+) first=(\d+) rows=(\d+) '
                     r'ids=([0-9a-f]{16}) prefix_weights=([0-9a-f]{16})$',text,re.M)
    require(len(frames)==48 and all((int(layer),int(first),int(rows))==(i,0,plan['read']-1)
        for i,(layer,first,rows,_,__) in enumerate(frames)),'missing or inconsistent exact router digest')
    return dict(full_route_ids=signature([x[3] for x in frames]),
                prefix_weights=signature([x[4] for x in frames]),layers=len(frames),variant=variant)


def numerical_decision(processes):
    def observed(process):
        return [dict(ids=route['router']['full_route_ids'],weights=route['router']['prefix_weights'],
                     body=sample['request_sha256'],jobs=route['actual_GU_dispatch'],
                     uploads=route['uploads'],requested_bank_H2D_bytes=route['requested_bank_H2D_bytes'],
                     MTP_expert_restore_bytes=route['MTP_expert_restore_bytes'])
                for route,sample in zip(process['routes'],process['samples'],strict=True)]
    off, late, early=map(observed,processes)
    return dict(comparisons={'late_vs_off':late==off,'early_vs_late':early==late},
        router_equivalence=off==late==early,matched_requests=len(off),
        observer_speed_qualified=False,inference_adopted=False,
        next='If numerical and ownership gates pass, compare the storage/readback change and earlier compute placement separately using observer-free repeated 64K requests.')


def model_context():
    ctx=private_module(basis.core,'prefix_router_model')
    ctx.ROOT,ctx.KEY,ctx.THIS=ROOT,KEY,THIS
    ctx.READS,ctx.ORDER,ctx.RUNS,ctx.CLEAN_FAST=(READ,)*len(ORDER),[(READ,1)]*len(ORDER),RUNS,True
    def reference():
        native=json.loads((BUILD_ROOT/'record.json').read_text())
        check_build(native)
        require(native['stage']=='built' and native['passed'] and native['complete']
            and native['host_control']['passed'] and native['ownership_closed'] and not native['active'],
            'same-build CPU evidence required')
        require(all(sha(p)==h for k in ('compiled','generated','links','objects','produced')
            for p,h in native['actual_inputs'][k].items())
            and all(sha(v['path'])==v['sha256'] for v in native['binaries'].values()),'actual native build inputs differ')
        _,_,speed,_,_=historical()
        old=json.loads((basis.reference.model.ROOT/'record.json').read_text())
        require(old['models']==speed['model_assets'] and old['probe']==speed['native_validation_probe'],
            'coherent model assets or probe differs')
        old['observer_admission']=dict(native_build=signature(build_context().compact(native)),
            fast_configuration_diagnostic=dict(scope='Use immutable historical assets and providers. Qualify new order from fresh model requests.'))
        return dict(binaries=native['binaries'],native_source_commit=native['source_commit']), old, \
            json.loads(Path(speed['runtime_manifest']['path']).read_text()), None
    old_env,old_current=ctx.environment,ctx.current
    def environment(row,plan):
        env=old_env(row,plan)
        env.update({MODE:ORDER[plan['index']], 'STRATA_TRACE':'1',
            'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS':'512',
            'STRATA_SYCL_PREFILL_QSA_MATH':'bf16-prefix-ordered-exp',
            'STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR':'1',
            'STRATA_SYCL_PREFILL_GU_SWIGLU':'large', 'STRATA_SYCL_PREFILL_MOE_CENSUS':'1',
            'STRATA_PREFILL_LAYER_MAJOR_PROFILE':'0', 'STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING':'0',
            'STRATA_SYCL_PREFILL_FULLFIT_UPLOAD':'jit'})
        return env
    def current(row):
        require([p['variant'] for p in row['plans']]==list(ORDER)
            and all(p['bank']==512 and p['window']==10 and p['upload']=='jit' and p['relocate']==0
                and p['trace_progress'] and p['read']==READ for p in row['plans']),
            'frozen single-factor numerical plan differs')
        return old_current(row)
    def route(text,plan):
        result=workspace.request_route(text,plan)
        result['router']=router_evidence(text,plan,result['layout'])
        return result
    def sample(sample,count,read,minimum_ms=10000):
        basis.comparison.check_sample(sample,count,read,minimum_ms)
        require(sample['generated_tokens']==24,'match the same request output extent')
    ctx.basis,ctx.environment,ctx.current,ctx.request_route=reference,environment,current,route
    ctx.decision,ctx.check_sample=numerical_decision,sample
    return ctx


def prepare_model():
    ctx=model_context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    for plan,mode in zip(row['plans'],ORDER,strict=True):
        plan.update(variant=mode,bank=512,window=10,relocate=0,trace_progress=True,upload='jit')
        require(plan['environment_sha256']==basis.comparison.fingerprint(ctx.environment(row,plan)),
                'frozen runtime mode differs')
    row['plan']=dict(unit='Off versus late isolates a dedicated prefix buffer and split ID readback. Late versus early moves the same producer and readback before later attention. Same expert uploads, queue and post-prefix FFN.',
        body='Three owned processes off/late/early, two identical thinking-on 32769-input and 24-output benchmark bodies per process. Finite logs and normal cleanup. Existing SYCL safety flags.',
        gate='All 48 routed ID digests and first 8192-row FP32 gate-weight digests agree per request, along with executed job geometry, transfer totals and MTP handoff. Early and late must admit equal workspace, publish exactly once per layer and complete after checked drains.',
        limits='Numerical route/owner check only. Added copy, hash and trace observability is charged here. Subsequent native-order measurements require clean repeated 64K comparisons. Physical context and sustained decode are separate.',
        retention='Use this exact registered build for the imminent clean router overhead comparison if numerical qualification passes.')
    ctx.save(row)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2,'use the pinned Python interpreter and one declared action')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='build-prepare': prepare_build()
        elif action in ('configure','compile'): build_context().execute(action)
        elif action=='host-native': host_native()
        elif action=='prepare': prepare_model()
        elif action=='host': basis.comparison.host(model_context())
        elif action in ('0','1','2'): basis.comparison.run(int(action),model_context())
        else: raise RuntimeError('unknown stage')
