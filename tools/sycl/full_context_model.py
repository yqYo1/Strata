#!/usr/bin/env python3
"""Owned diagnostic-engine model admission/full fill; Main, repository LICENSE.

Synthetic IDs test capacity and checked completion, not task quality or speed.
The selected production engine and its decode placement remain unchanged.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import subprocess

sys.dont_write_bytecode = True
import check_full_context as gate

P, B, RESULT = gate.P, gate.B, gate.RESULT
ROOT = B / 'full-context-model-v3'
KEY = 'model_capacity_v3'
DECODE = P / 'bench/results/2026-10-15-pr6-recovery-audit/canonical-restore-decode-current.json'
BUILD = P / 'bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json'
DECODE_ARCHIVE = dict(commit='a8af3d7fa96a7713f2e516d2da3d7410956be015',
    path=str(DECODE.relative_to(P)),sha256='7dda2bcefa6f7e54d568e6aa3d5226d5e0a7383f0b41efd7584093854d9547ca')
INPUTS = sorted(set([*gate.INPUTS, 'tools/sycl/full_context_model.py', 'data/expert-profile.bin']))
require = gate.native.guards.require
digest = gate.native.build.digest
write = gate.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def inputs(revision):
    pins = {}
    for name,data in gate.native.build.blobs(P,revision,INPUTS):
        require(data == (P/name).read_bytes(),'commit model controller input: '+name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def setup():
    closed = gate.validation.committed(DECODE)
    require(closed['schema']=='strata-canonical-decode-closed-v1' and closed['stage']=='closed' and
        closed['full_model_hashes_matched_after_cohort']==15,'closed canonical model reference required')
    require(all(closed['archive'][k]==DECODE_ARCHIVE[k] for k in ('commit','path')),
        'canonical decode archive differs')
    archive = gate.native.build.archived_model_observation(DECODE_ARCHIVE)
    selected = archive['plans'][0]
    require(selected['arm']==0 and selected['cohort']==0 and len(archive['processes'])==6 and
        all(r['passed'] and r['complete'] and r['ownership_closed'] and r['fresh_faults']==[] and
            r['boot_unchanged'] for r in archive['processes']),'closed original model cohort required')
    build = gate.validation.committed(BUILD)
    model_ref = build['retention_model_reference']
    recovery = gate.native.build.archived_model_observation(model_ref)
    require(build['source_commit']==recovery['plan']['native_source']==
        '6e40afc1beddd059ee518617f6c1efce1da14aca' and len(recovery['models'])==15,
        'original model source/assets differ')
    profile = str(Path(selected['root'])/'expert-profile.bin')
    require(digest(P/'data/expert-profile.bin')==selected['inputs'][profile], 'original profile differs')
    return dict(config=closed['configuration'],environment=selected['environment'],models=recovery['models'],
        reference=dict(decode_archive=DECODE_ARCHIVE,decode_compact_sha256=digest(DECODE),
            model=model_ref,build_compact_sha256=digest(BUILD)))


def model_pins(config, expected, full=False):
    args = config['args']
    native = Path(args[args.index('--native')+1])
    paths = set(native.parent.glob('*.gguf'))
    for flag in ('--pack','--mtp'):
        paths.update(q for q in Path(args[args.index(flag)+1]).rglob('*') if q.is_file())
    require(set(map(str,paths))==set(expected) and len(paths)==15,'model inventory differs')
    deadline = time.monotonic()+300
    for q in sorted(paths):
        def identity():
            s=q.stat()
            return dict(realpath=str(q.resolve()),dev=s.st_dev,ino=s.st_ino,bytes=s.st_size,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
        before = identity()
        require(before==expected[str(q)]['identity'],'model stat differs: '+str(q))
        if full:
            h=hashlib.sha256()
            with q.open('rb',buffering=0) as stream:
                while block := stream.read(8<<20):
                    require(time.monotonic()<deadline,'whole-model digest300s budget')
                    h.update(block)
            require(h.hexdigest()==expected[str(q)]['sha256'] and identity()==before,
                'model bytes changed: '+str(q))
            print('Verified model '+q.name,flush=True)


def compact(record):
    # Effective environment reconstructs from the pinned archive and overrides;
    # retain its hash rather than another copy of the complete environment.
    return {k:v for k,v in record.items() if k not in ('environment','commands')}


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh model diagnostic only')
    smoke=container['device_smoke_v2']
    require(smoke['passed'] and smoke['complete'] and smoke['ownership_closed'] and not smoke['active'] and
        smoke['GPU_work_submitted'] and smoke['fresh_faults']==[],'closed traced device gate required')
    engine=gate.engine(); _,original=gate.native.source.baseline(); base=setup()
    # v1 preparation hashed all15 assets before its preflight-only refusal.
    # Reuse those byte hashes only with the same stat/ctime and inventory guards.
    revision_before='102b83623e93ed39318eda634245ed81b652f997'
    raw=gate.native.build.git(P,'show',revision_before+':'+str(RESULT.relative_to(P)))
    prior=json.loads(raw)['model_capacity_v1']
    require(prior['source_commit']=='9ff5527505665296af1db23f65f5db8148354069' and
        prior['stage']=='prepared' and prior['model_assets']==base['models'], 'original whole-model digest proof differs')
    model_pins(base['config'],base['models'])
    revision=gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    pins=inputs(revision)
    args=list(base['config']['args'])
    pack=Path(args[args.index('--pack')+1])
    vocab=json.loads((pack/'tokenizer/vocab.json').read_text())
    ids=set(vocab.values())
    require(set(range(4096))<=ids and min(ids)==0,'synthetic input IDs outside actual vocabulary')
    ROOT.mkdir(mode=0o700); (ROOT/'tmp').mkdir(); (ROOT/'cache').mkdir()
    oldroot=base['environment']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(oldroot,str(ROOT)) for k,v in base['environment'].items()}
    probe=gate.validation.preload_qualification()
    require(not env.get('LD_PRELOAD'),'unexpected preload')
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),CUDA_MODULE_LOADING='EAGER',
        LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1',STRATA_SYCL_CACHE_SCRATCH_LEASE='0')
    env.pop('UR_LOG_TRACING',None)
    for name in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
        env[name]='level:warning;flush:warning;output:stderr'
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                 'STRATA_SYCL_PREFILL_SAME_QUEUE','SYCL_CACHE_PERSISTENT'):
        require(env[name]==engine['environment'][name],'qualified safety setting changed: '+name)
    args[args.index('--max-context')+1]='262144'
    args[args.index('--expert-profile')+1]=str(P/'data/expert-profile.bin')
    require(args[args.index('--expert-cache')+1]=='1256' and
        args[args.index('--kv-resident')+1]=='131072','decode placement differs')
    cases={}
    for name,n,outputs in (('short',16385,4),('full',262144,0)):
        path=(ROOT/'tokens16385.txt') if name=='short' else gate.ROOT/'tokens262144.txt'
        data=gate.token_bytes(n)
        if name=='short': path.write_bytes(data)
        require(path.read_bytes()==data,'frozen token fixture differs')
        command=[engine['binary']['path'],*args,'--tokens-file',str(path),'--max-new',str(outputs)]
        if name=='full': command.append('--consume-prompt')
        cases[name]=dict(command=command,input_tokens=n,output_tokens=outputs,
            fixture=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),
            wall_s=900 if name=='short' else 2400)
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,binary=engine['binary'],
        selected_original=original,setup_reference=base['reference'],model_assets=base['models'],preload=probe,
        initial_model_digest_reference=dict(commit=revision_before,path=str(RESULT.relative_to(P)),
            sha256=hashlib.sha256(raw).hexdigest(),scope='v1 preparation full15-asset digests;unchanged inode/size/mtime/ctime and inventory required before/after every case;full hashes again after full fill.'),
        environment=env,environment_sha256=fingerprint(env),vocab_size=max(ids)+1,cases=cases,outcomes={},
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver_before=gate.native.guards.driver_guard(),
        active=False,ownership_closed=True,passed=False,complete=False,production_selected=False,
        model_adoption_qualified=False,physical262144_qualified=False,ordinary_runtime_teardown_qualified=False,
        budgets=dict(cpu_s=4800,rss_bytes=104<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=512<<20,
            reason='Two cases have at most64MiB combined raw stdout/stderr;512MiB total additionally bounds private compiler JIT/cache/receipts,all retired after their consumer closes.'),
        question='Can the independently built diagnostic execute the unchanged model route,then allocate/process all262144 physical input slots with fixed1256 decode slots and131072 resident INT8 KV? Host KV tail is allowed and must not be called all-VRAM residency.',
        gate='Run short once with16384 batched rows(two8192 chunks)+final prompt row and4 output IDs. The earlier one-chunk case exited0 but failed its declared layer-major route gate;it does not advance. Only after the new committed normal/validated/fault-free short closure run full once:262143 batched rows+final T1,zero outputs,exact checked final-commit marker. Require BF16 residual allocation conservation,configured MoE10 with effective2 for no accumulation or10 for admitted accumulation,checked prompt release,unchanged sources/models/boot/driver and owned normal exit. Server-only scratch lease is0;the existing standalone lend/refill path remains enabled. This is a capacity check,not preservation of the serve allocation route. Stop first failure;no retry or speed decision.',
        retention='Keep exact plans,model hashes,individual completion/activation/fault/allocation results. Retire successful traces/JIT after verification. Diagnostic engine and full IDs remain only through imminent near-cap/RESTORE/owner checks;original selected13 assets stay untouched.',
        next_gate='Near-cap/RESTORE/fault-owner,model/task and repeated decode-preservation comparisons. Full synthetic fill alone is not general physical-context qualification or throughput evidence.')
    write(ROOT/'record.json',record);container[KEY]=compact(record);write(RESULT,container)
    print('Prepared short/full model diagnostic;commit exact plan before either run.',flush=True)


def admission(record,container):
    require(RESULT.read_bytes()==gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        container[KEY]==compact(record) and inputs(record['source_commit'])==record['input_sha256'],
        'commit exact model plan/inputs first')
    require(fingerprint(record['environment'])==record['environment_sha256'],'effective environment changed')
    require(gate.engine()['binary']==record['binary'] and gate.native.source.baseline()[1]==record['selected_original'],
        'native source/assets changed')
    base=setup()
    require(base['reference']==record['setup_reference'] and base['models']==record['model_assets'] and
        gate.validation.preload_qualification()==record['preload'],'reference/runtime changed')
    model_pins(base['config'],record['model_assets'])
    for row in record['cases'].values():
        require(digest(row['fixture']['path'])==row['fixture']['sha256'],'input fixture changed')


def evidence(stdout,stderr,record,case,pid):
    proof=gate.validation.preload_evidence(stderr,pid)
    boundary=re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
    require(boundary is not None and proof['loader']==record['preload']['loader']['path'],'activation differs')
    body=stderr[boundary.end():]
    require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',body),
        'runtime failure after validation canary')
    n=case['input_tokens']
    pattern=(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(\S+) '
        r'residual_element_bytes=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) '
        r'residual_gpu=(\d+) residual_host=(\d+) residual_staging=(\d+) layer_cache=(\d+);')
    matches=list(re.finditer(pattern,body,re.M));require(len(matches)==1,'one allocation ledger required')
    fields=('tokens','chunk','dtype','element_bytes','total','gpu_rows','gpu_bytes','host_bytes','staging_bytes','layer_cache')
    ledger={k:(v if k=='dtype' else int(v)) for k,v in zip(fields,matches[0].groups())}
    require(ledger['tokens']==n-1 and ledger['chunk']==8192 and ledger['dtype']=='bf16' and
        ledger['element_bytes']==2 and ledger['total']==(n-1)*10240*2 and
        ledger['gpu_bytes']+ledger['host_bytes']==ledger['total'] and
        ledger['gpu_bytes']==ledger['gpu_rows']*10240*2,'actual residual allocation conservation failed')
    window=re.search(r'^strata MoE window configuration: requested_mode=10 effective_mode=(2|10) maximum_rows=(\d+) mixed_fp16_bytes=(\d+)$',body,re.M)
    require(window is not None and
        'MiB free after releasing the standalone prompt buffers' in body,'model route/release evidence missing')
    ledger['effective_moe_mode']=int(window[1])
    bank=re.search(r'^strata MoE expert bank: .*allocation_limit_bytes=(\d+)',body,re.M)
    require(bank is not None,'actual allocation limit missing')
    ledger['device_max_allocation_bytes']=int(bank[1])
    free=[dict(mib=int(m[1]),after=m[2]) for m in re.finditer(r'^strata trace: (\d+) MiB free after (.+)$',body,re.M)]
    prefill=re.search(r'^prefill\s+(\d+) tokens in ([0-9.]+) ms',stdout,re.M)
    require(prefill is not None and int(prefill[1])==n-1,'actual batched extent differs')
    output=re.search(r'^output\s*:(.*)$',stdout,re.M);require(output is not None,'output framing missing')
    tokens=[int(v) for v in output[1].split()]
    require(len(tokens)==case['output_tokens'] and all(0<=t<record['vocab_size'] for t in tokens),'output extent/IDs invalid')
    marker='strata FULL_CONTEXT consumed_position=262144 prompt_tokens=262144 output_tokens=0 window_width=1 commit_count=1 commit_wait=PASS refill_drain=PASS'
    require((marker in stdout)==(n==262144),'full final-slot completion marker differs')
    return dict(validation=proof,batched_rows=n-1,output_tokens=len(tokens),final_slot_checked=n==262144,
        allocation=ledger,free_memory_mib=free,prefill_ms_observed=float(prefill[2]),speed_eligible=False)


def run(container,mode):
    record=json.loads((ROOT/'record.json').read_text());admission(record,container)
    require(mode not in record['outcomes'] and not record['active'] and record['stage'] in ('prepared','short_closed'),
        'no retry/duplicate model case')
    require(mode=='short' or record['outcomes'].get('short',{}).get('passed'),'closed short model gate required')
    require(record['driver_before']==gate.native.guards.driver_guard() and
        record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'driver/boot changed')
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in text.splitlines() if s.startswith('-- cursor: '))
    original=json.loads((ROOT/'record.json').read_text());owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,commands=owner.commands if owner else []))
    owner=gate.native.supervision.Owner(ROOT/('owned-'+mode),persist)
    row=dict(passed=False,complete=False,ownership_closed=False);record['outcomes'][mode]=row
    record['stage']=mode+'_running';persist()
    try:
        c=record['cases'][mode];b=record['budgets']
        entry,out,err=owner.run(mode,c['command'],record['environment'],ROOT,wall=c['wall_s'],cpu=b['cpu_s'],
            rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        row.update(exit_code=entry.get('exit_code'),stdout_sha256=digest(out),stderr_sha256=digest(err))
        require(gate.native.supervision.closed(entry) and entry['exit_code']==0,'native model exit/ownership failed')
        row.update(evidence(out.read_text(errors='replace'),err.read_text(errors='replace'),record,c,entry['pid']))
        admission(original,container)
        if mode=='full':model_pins(setup()['config'],record['model_assets'],full=True)
        row['passed']=True
    except BaseException as error:row['error']=repr(error)
    finally:
        row['ownership_closed']=owner.active is None and all(gate.native.supervision.closed(c) for c in owner.commands)
        try:
            text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row['fresh_faults']=[s for s in text.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            row['driver_after']=gate.native.guards.driver_guard()
            row['boot_unchanged']=record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except BaseException as error:row['final_probe_error']=repr(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row.get('fresh_faults')==[] and
            row.get('boot_unchanged') and row.get('driver_after')==record['driver_before'] and 'final_probe_error' not in row)
        row['complete']=row['passed'];record['ownership_closed']=row['ownership_closed']
        record['stage']=mode+'_closed' if row['passed'] else 'failed'
        record['passed']=record['complete']=bool(mode=='full' and row['passed'])
        persist();container[KEY]=compact(record);write(RESULT,container)
    print(json.dumps({k:row.get(k) for k in ('passed','complete','exit_code','ownership_closed','fresh_faults','error','final_probe_error')}),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['short'],['full']),'prepare|short|full')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container,sys.argv[1]))
