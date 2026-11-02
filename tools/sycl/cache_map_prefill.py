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
import sys

sys.dont_write_bytecode=True
import cache_map_model_entry as entry

P,B,RESULT=entry.P,entry.B,entry.RESULT
SOURCE='tools/sycl/cache_map_prefill.py'
ROOT=entry.model.ROOT/'prefill-read117489-v2'
KEY='cache_map_prefill_read117489_v2'
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
    return row


def parser_controls(original,legacy):
    old=legacy.selfcheck()
    lines=['strata prefill dense: phase=retired residency=Full',
        'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000',
        'strata prefill heads: draft detached bytes=1024 retained-map-bytes=0',
        'strata prefill dense: phase=bank residency=Bank',
        f'strata prefill: layer-major candidate, tokens={ROWS} residual_gpu_rows={ROWS} chunk=8192 residual_dtype=bf16',
        'strata MoE window configuration: requested_mode=10 effective_mode=10',
        'strata MoE split admission: active=0']
    for layer in range(48):
        for phase in ('before_drain','after_drain'):
            lines.append(f'strata MoE progress: phase={phase} layer={layer} first=0 submitted=8 total=8 interval=8 queue=0x100')
    lines+=['strata MoE accumulator window: windows=48 jobs=384 source_H2D_bytes=1024',
        'strata prefill dense: phase=restored residency=Full activations=48']
    good='\n'.join(lines);v=original.progress(good)
    require(v['rows']==v['GPU_residual_rows']==ROWS and v['checkpoint_pairs']==48,'extended extent parser')
    bad=[good.replace(f'tokens={ROWS}', 'tokens=32768'),good.replace(f'residual_gpu_rows={ROWS}',f'residual_gpu_rows={ROWS-1}'),
        good.replace('jobs=384','jobs=385'),good+'\n'+good]
    for text in bad:
        try:original.progress(text)
        except (RuntimeError,ValueError,KeyError):continue
        raise RuntimeError('extended corrupt extent accepted')
    require(legacy.ROWS==32768,'shared parser context changed')
    return dict(passed=True,legacy=old,extended_positive=1,extended_negative=len(bad),GPU_executed=False)


def context():
    ctx=entry.context('zero')
    ctx.ROOT,ctx.KEY,ctx.READ,ctx.ROWS,ctx.RUNS=ROOT,KEY,READ,ROWS,RUNS
    ctx.pinned_sources=pinned_sources
    legacy=ctx.original
    spec=importlib.util.spec_from_file_location('strata_current117k_progress_context',legacy.__file__)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    original.ROWS,original.READ=ROWS,READ
    original.selfcheck=lambda:parser_controls(original,legacy)
    ctx.original=original
    ctx.print=lambda *args,**kw:builtins.print(*(v.replace('32K request','117K request')if isinstance(v,str)else v for v in args),**kw)
    return ctx


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in ('selfcheck','prepare','host','run'),'selfcheck|prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry.base.guards.no_model_owners([])
        ctx=context()
        if sys.argv[1]=='selfcheck':
            revision=ctx.shared.gen.git('rev-parse','HEAD').decode().strip();pinned_sources(revision)
            print(json.dumps(ctx.route_controls()),flush=True)
        elif sys.argv[1]=='prepare':
            revision=ctx.shared.gen.git('rev-parse','HEAD').decode().strip();prior=basis(revision)
            entry.prepare(ctx,'zero');row=json.loads((ROOT/'record.json').read_text())
            require(row['binary']==prior['binary'] and row['native_source_commit']==prior['native_source_commit'] and
                row['environment']['STRATA_SYCL_ADAPT_SYNC']=='1' and
                row['configuration']['args'][row['configuration']['args'].index('--adapt-swaps')+1]=='0',
                'unchanged corrected engine and zero-swap baseline required')
            row['plan_scope']='First corrected-engine117489-read/117488-row check:two same-length requests in one process,thinking on,24output cap,zero reuse,throughput strategy,MTP16/GDN16. Actual route/owner/extent and final full model hashes required. First use and subsequent request reported separately;no clean repeated-speed or sustained-TG claim.'
            row['criteria']=row['criteria'].replace('at the pinned read_tokens','at117489actual read tokens')
            row['criteria']=row['criteria'].replace('Two fresh32K requests','Two fresh117K requests')
            row['qualification_basis']['controller_delta']=dict(new_extent_adapter=SOURCE,
                unchanged_existing_generation_adapter=PRIOR_ADAPTER,
                existing_adapter_original_commit=PRIOR_ADAPTER_COMMIT,
                original_entry_inputs_unchanged=True,
                limits='The existing generation adapter is conservatively pinned by benchmark input discovery;no failed generation cohort or V1 admission is qualified or resumed.')
            row['retention']='Keep compact individual samples,route/extent,source/native identity and fault/closure outcome. Retire successful raw output after publication and source recovery. Both existing model generations remain only while the imminent unchanged entry proof still consumes the fallback.'
            ctx.save(row)
        elif sys.argv[1]=='host':ctx.host()
        else:ctx.run()
