#!/usr/bin/env python3
"""Current corrected-engine117K extent check; repository LICENSE.

Private policy/parser contexts reuse the qualified32K entry. No native,benchmark,
shared context or process environment change. Logged lifecycle/extent evidence
is separate from clean repeated speed and sustained generation qualification.
"""
import builtins
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
import cache_map_model_entry as entry

P,B,RESULT=entry.P,entry.B,entry.RESULT
SOURCE='tools/sycl/cache_map_prefill.py'
ROOT=entry.model.ROOT/'prefill-read117489-v3'
KEY='cache_map_prefill_read117489_v3'
PARSER_KEY='cache_map_prefill_parser_v3'
FAILED_ROOT=entry.model.ROOT/'prefill-read117489-v2'
FAILED_KEY='cache_map_prefill_read117489_v2'
PRIOR_ADAPTER='tools/sycl/cache_map_generation.py'
PRIOR_ADAPTER_COMMIT='70805eb7bc432d2bba6eb446ef1b584786775499'
READ,ROWS,RUNS=117489,117488,2
require,digest=entry.require,entry.digest


def pinned_sources(revision):
    values=entry.pinned_sources(revision)
    values.update(entry.base.controls(revision,controller=Path(__file__).resolve()))
    return values


def basis(revision):
    prior=entry.context('zero');row=json.loads((prior.ROOT/'record.json').read_text());prior.current(row)
    require(row['stage']=='closed' and row['passed'] and row['complete'] and row['ownership_closed'] and
        not row['active'] and row['host_passed'] and row['outcome']['native_exit_code']==0 and
        not row['outcome']['fresh_faults'] and len(row['outcome']['samples'])==2,
        'closed current-engine zero-swap32K qualification required')
    recovered={n:hashlib.sha256(raw).hexdigest()for n,raw in prior.shared.gen.blobs(P,row['source_commit'],row['sources'])}
    current=pinned_sources(revision)
    prior_adapter=prior.shared.gen.git('show',PRIOR_ADAPTER_COMMIT+':'+PRIOR_ADAPTER)
    require(recovered==row['sources'] and all(current.get(n)==h for n,h in recovered.items()) and
        set(current)-set(recovered)=={SOURCE,PRIOR_ADAPTER} and
        current[PRIOR_ADAPTER]==hashlib.sha256(prior_adapter).hexdigest(),
        'unchanged original entry inputs plus this adapter and the recovered existing generation controller only')
    failed=entry.base.committed(RESULT)['cache_map_prefill_read117489_admission_v1']
    require(failed['stage']=='failed' and not failed['model_executed'] and failed['ownership_closed'] and
        failed['source_commit']=='11f912553c491d13e30491667f17c4d1043b1569',
        'preserve the closed pre-owner V1 admission failure')
    parser=entry.base.committed(RESULT)[PARSER_KEY]
    require(parser['passed'] and parser['complete'] and parser['ownership_closed'] and
        parser['GPU_executed'] is False and parser['source_sha256']==digest(__file__) and
        parser['requests']==2 and parser['preserved_failed_scope']==FAILED_KEY,
        'commit the current-source two-window parser replay before fresh model preparation')
    return row


def long_progress(text,common,comparison):
    actual=comparison.route(text,READ,1)
    require(actual['rows']==actual['residual_gpu_rows']==ROWS and actual['residual_host_bytes']==0 and
        actual['prefill_slots']==128 and actual['split'] and actual['split_guard_checked'] and
        actual['staged_bank_checked_drain'],'full long split/bank contract required')
    # The long-route parser above validates the real admission, tail bounds,
    # completion guard and drains. Only the legacy checker receives a private
    # compatibility copy; retain all of its checkpoint and owner checks.
    lines=[line for line in text.splitlines()if line.startswith('strata MoE split admission: ')]
    require(len(lines)==1 and 'active=1 'in lines[0],'one real active split admission required')
    result=common(text.replace(lines[0],lines[0].replace('active=1 ','active=0 ',1),1))
    require(result['jobs']==actual['jobs'] and result['windows']==actual['windows'],
        'long-route and checkpoint work differ')
    result.update(split_active=True,split_guard_checked=True,prefill_slots=actual['prefill_slots'],
        staged_bank_checked_drain=actual['staged_bank_checked_drain'],
        residual_gpu_bytes=actual['residual_gpu_bytes'],residual_host_bytes=actual['residual_host_bytes'],
        split_primary_rows=65536,split_tail_payload_bytes=(ROWS-65536)*2560*4)
    return result


def parser_controls(original,legacy,comparison):
    old=legacy.selfcheck()
    geometry=comparison.selfcheck()
    tail=(ROWS-65536)*2560*4
    lines=['strata prefill dense: phase=retired residency=Full',
        'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000',
        f'strata prefill heads: draft detached bytes={comparison.DRAFT_BYTES} retained-map-bytes={comparison.MAP_BYTES}',
        'strata prefill dense: phase=bank residency=Bank',
        f'strata MoE split admission: active=1 rows={ROWS} primary_rows=65536 tail_payload_bytes={tail} '
            f'extra_allocation_bytes={tail+256} prefill_slots=128 max_single_allocation_bytes={4<<30} decode_changed=0',
        f'strata MoE expert bank: requested_slots=384 effective_slots=128 staged=1 host_slab_bytes=1363148800 '
            f'device_bytes=340787200 residual_gpu_rows={ROWS} allocation_limit_bytes={4<<30} general_map=private_scratch_never_published',
        f'strata prefill: layer-major candidate, tokens={ROWS} residual_gpu_rows={ROWS} chunk=8192 residual_dtype=bf16 '
            f'residual_element_bytes=2 residual_total={ROWS*20480} residual_gpu={ROWS*20480} residual_host=0 '
            'residual_staging=0 layer_cache=340787200',
        f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={ROWS}']
    for layer in range(48):
        for phase in ('before_drain','after_drain'):
            lines.append(f'strata MoE progress: phase={phase} layer={layer} first=0 submitted=8 total=8 interval=8 queue=0x100')
    lines+=[f'strata MoE accumulator window: windows=48 jobs=384 dequants=48 source_H2D_bytes=1024 '
                f'GPU_resident_windows_only=1 operator_max=8192 maximum_rows={ROWS}',
        'strata MoE Down fusion: fused_jobs=2 fused_rows=4668 separate_jobs=382 min_rows=2334',
        'strata MoE expert bank complete: uploads=2 requested_H2D_bytes=1024 permanent_slots=1 total_slots=128 final_drain=checked',
        f'strata MoE split complete: rows={ROWS} primary_rows=65536 tail_bytes={tail} '
                'guard=checked final_drain=checked decode_changed=0',
        'strata prefill dense: phase=restored residency=Full activations=48']
    good='\n'.join(lines);v=original.progress(good)
    require(v['rows']==v['GPU_residual_rows']==ROWS and v['checkpoint_pairs']==48 and
        v['split_active'] and v['split_guard_checked'],'extended split/extent parser')
    bad=[good.replace(f'tokens={ROWS}', 'tokens=32768'),good.replace(f'residual_gpu_rows={ROWS}',f'residual_gpu_rows={ROWS-1}'),
        good.replace('jobs=384','jobs=385'),good+'\n'+good,
        good.replace('active=1 rows=','active=0 rows='),good.replace('primary_rows=65536','primary_rows=65535'),
        good.replace(f'tail_payload_bytes={tail}',f'tail_payload_bytes={tail-1}'),
        good.replace(f'extra_allocation_bytes={tail+256}',f'extra_allocation_bytes={tail}'),
        good.replace(f'max_single_allocation_bytes={4<<30}',f'max_single_allocation_bytes={tail}'),
        good.replace('prefill_slots=128','prefill_slots=384'),good.replace('guard=checked','guard=failed'),
        good.replace('final_drain=checked','final_drain=failed'),good.replace('decode_changed=0','decode_changed=1'),
        good.replace('residual_host=0','residual_host=1'),good.replace('residual_element_bytes=2','residual_element_bytes=4'),
        good.replace('separate_jobs=382','separate_jobs=381'),
        good.replace('phase=after_drain layer=0 first=0 submitted=8','phase=after_drain layer=0 first=0 submitted=16')]
    for text in bad:
        try:original.progress(text)
        except (RuntimeError,ValueError,KeyError):continue
        raise RuntimeError('extended corrupt extent accepted')
    require(legacy.ROWS==32768,'shared parser context changed')
    return dict(passed=True,legacy=old,long_geometry=geometry,
        extended_positive=1,extended_negative=len(bad),GPU_executed=False)


def context():
    ctx=entry.context('zero')
    ctx.ROOT,ctx.KEY,ctx.READ,ctx.ROWS,ctx.RUNS=ROOT,KEY,READ,ROWS,RUNS
    ctx.pinned_sources=pinned_sources
    legacy=ctx.original
    spec=importlib.util.spec_from_file_location('strata_current117k_progress_context',legacy.__file__)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    original.ROWS,original.READ=ROWS,READ
    common=original.progress
    original.progress=lambda text:long_progress(text,common,ctx.comparison)
    original.selfcheck=lambda:parser_controls(original,legacy,ctx.comparison)
    ctx.original=original
    ctx.print=lambda *args,**kw:builtins.print(*(v.replace('32K request','117K request')if isinstance(v,str)else v for v in args),**kw)
    return ctx


def replay(ctx):
    evidence=entry.base.committed(RESULT)
    require(PARSER_KEY not in evidence,'fresh parser replay only')
    failed=json.loads((FAILED_ROOT/'record.json').read_text())
    require(entry.template.compact(failed)==evidence[FAILED_KEY] and failed['stage']=='failed' and
        not failed['passed'] and not failed['active'] and failed['ownership_closed'] and
        failed['failure_diagnosis']['observed_native_exit_code']==0,
        'preserve the closed failed V2 scope')
    fixture=failed['failure_diagnosis']['fixture'];raw=Path(fixture['path']).read_bytes()
    require(len(raw)==fixture['bytes']<=fixture['maximum_bytes'] and
        hashlib.sha256(raw).hexdigest()==fixture['sha256'],'original captured fixture changed')
    recovered={n:hashlib.sha256(b).hexdigest()for n,b in ctx.shared.gen.blobs(P,failed['source_commit'],failed['sources'])}
    revision=ctx.shared.gen.git('rev-parse','HEAD').decode().strip();current=pinned_sources(revision)
    require(recovered==failed['sources'] and set(current)==set(recovered) and
        {n for n in current if current[n]!=recovered[n]}=={SOURCE},'only the declared parser adapter may change')
    controls=ctx.route_controls();text=raw.decode()
    starts=list(re.finditer(r'^strata prefill policy: [^\n]*\btokens='+str(ROWS)+r'\b[^\n]*$',text,re.M))
    ends=list(re.finditer(r'^strata serve: prompt '+str(READ)+r' tokens = 0 reused \+ '+str(READ)+r' read[^\n]*$',text,re.M))
    require(len(starts)==len(ends)==2 and all(s.start()<e.start() for s,e in zip(starts,ends)) and
        ends[0].end()<starts[1].start(),'two bounded original request windows required')
    routes=[ctx.profile_route(text[s.start():e.end()])for s,e in zip(starts,ends)]
    row=dict(source_commit=revision,source_sha256=digest(__file__),passed=True,complete=True,
        ownership_closed=True,GPU_executed=False,requests=len(routes),preserved_failed_scope=FAILED_KEY,
        fixture=dict(path=fixture['path'],bytes=len(raw),sha256=fixture['sha256']),controls=controls,
        routes=routes,original_inputs_recovered=len(recovered),
        limits='Offline parser replay only. Original V2 remains failed;no new model,clean-speed or numerical qualification. Fresh V3 must pass its own complete model/asset/fault/owner checks.')
    evidence[PARSER_KEY]=row;entry.base.write(RESULT,evidence)
    print(json.dumps(row),flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in ('selfcheck','replay','prepare','host','run'),'selfcheck|replay|prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry.base.guards.no_model_owners([])
        ctx=context()
        if sys.argv[1]=='selfcheck':
            revision=ctx.shared.gen.git('rev-parse','HEAD').decode().strip();pinned_sources(revision)
            print(json.dumps(ctx.route_controls()),flush=True)
        elif sys.argv[1]=='replay':replay(ctx)
        elif sys.argv[1]=='prepare':
            revision=ctx.shared.gen.git('rev-parse','HEAD').decode().strip();prior=basis(revision)
            entry.prepare(ctx,'zero');row=json.loads((ROOT/'record.json').read_text())
            require(row['binary']==prior['binary'] and row['native_source_commit']==prior['native_source_commit'] and
                row['environment']['STRATA_SYCL_ADAPT_SYNC']=='1' and
                row['configuration']['args'][row['configuration']['args'].index('--adapt-swaps')+1]=='0',
                'unchanged corrected engine and zero-swap baseline required')
            row['plan_scope']='Fresh V3 corrected-engine117489-read/117488-row check after committed split-aware parser controls and real V2-window replay. Two same-length requests in one process,thinking on,24output cap,zero reuse,throughput strategy,MTP16/GDN16. Require real active-split bounds,guard/drain,bank/checkpoint/owner/route evidence and final full model hashes. First use and subsequent request separate;no clean repeated-speed or sustained-TG claim. V1/V2 stay failed.'
            row['criteria']=row['criteria'].replace('at the pinned read_tokens','at117489actual read tokens')
            row['criteria']=row['criteria'].replace('Two fresh32K requests','Two fresh117K requests')
            row['qualification_basis']['controller_delta']=dict(new_extent_adapter=SOURCE,
                unchanged_existing_generation_adapter=PRIOR_ADAPTER,
                existing_adapter_original_commit=PRIOR_ADAPTER_COMMIT,
                original_entry_inputs_unchanged=True,
                limits='The existing generation adapter is conservatively pinned by benchmark input discovery;no failed generation cohort or V1 admission is qualified or resumed.')
            row['qualification_basis']['parser_replay_result_key']=PARSER_KEY
            row['retention']='Keep compact individual samples,route/extent,source/native identity and fault/closure outcome. Retire successful raw output after publication and source recovery. Both existing model generations remain only while the imminent unchanged entry proof still consumes the fallback.'
            ctx.save(row)
        elif sys.argv[1]=='host':ctx.host()
        else:ctx.run()
