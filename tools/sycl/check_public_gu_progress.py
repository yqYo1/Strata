#!/usr/bin/env python3
"""Main-owned finite public-GU stage smoke; repository LICENSE. No model run."""
import copy
import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys
sys.dont_write_bytecode = True
import qualify_native16_control as q

r,P,B,RESULT=q.r,q.P,q.B,q.RESULT
ROOT=B/'public-gu-stage-smoke-v1'
NATIVE=B/'native16-stage-smoke-v1'
BUILD_KEY='public_gu_stage_smoke_build_v1'
KEY='public_gu_stage_smoke_v1'
require,digest,write=q.require,q.digest,q.write

def compact(row):
    return {k:v for k,v in row.items() if k not in
        ('controller_inputs','runtime_files','environment','native_build')}

def save(row):
    raw=RESULT.read_text();desired=json.loads(raw);q.build.scope(desired)[KEY]=compact(row)
    literal=json.dumps(KEY)+':';value=json.dumps(compact(row),indent=2).replace('\n','\n      ')
    if KEY in q.build.scope(json.loads(raw)):
        require(raw.count(literal)==1,'ambiguous stage-smoke key')
        start=raw.index(literal)+len(literal)
        while raw[start].isspace():start+=1
        _,size=json.JSONDecoder().raw_decode(raw[start:]);raw=raw[:start]+value+raw[start+size:]
    else:
        anchor='"fused_command_profile": {';require(raw.count(anchor)==1,'ambiguous result container')
        raw=raw.replace(anchor,anchor+'\n      '+literal+' '+value+',',1)
    require(json.loads(raw)==desired,'stage update changed unrelated evidence')
    write(ROOT/'record.json',row);RESULT.write_text(raw)

def identities():
    closed=json.loads((q.ROOT/'record.json').read_text());q.current(closed)
    old,_,_=q.native();new,_,proof=q.native(NATIVE,BUILD_KEY)
    require(all(new['fixtures'][k]['sha256']==old['fixtures'][k]['sha256']
        for k in new['fixtures'] if k!='sycl_moe_gu_swiglu_bench'),'other fixture bytes changed')
    def inputs(row):
        root=str(Path(row['compiled_inputs']['path']).parent)
        return {p.replace(root,'<native-root>'):sha for p,sha in
            json.loads(Path(row['compiled_inputs']['path']).read_text()).items()}
    a,b=inputs(old),inputs(new)
    changed={p for p in set(a)|set(b) if a.get(p)!=b.get(p)}
    require(changed=={str(P/'bench/sycl/moe_gu_swiglu.cpp')},'compile inputs changed outside GU fixture')
    helper=str(P/'src/prefill/sycl/vllm_xe2/xe_gemm_bsd.hpp')
    require(b[helper]=='84fb1251c2f9f93bd7774fb2a5a6c928df9ac8e86d1c7f6cf1a6cec058403fa6',
        'corrected TLA helper differs')
    require(new['sdk']==old['sdk'] and
        [v.replace(str(NATIVE),'<native-root>') for v in new['configure_argv']]==
        [v.replace(str(q.build.ROOT),'<native-root>') for v in old['configure_argv']],
        'component recipe/provider changed')
    return closed,new,proof

def prepare():
    require(not ROOT.exists() and KEY not in q.build.scope(r.validation.committed(RESULT)),'single-use diagnostic')
    previous=q.build.scope(r.validation.committed(RESULT))['public_gu_warmed_attribution_v2']
    require(previous['stage']=='failed' and not previous['passed'] and
        previous['failure_analysis']['ownership_reconciliation']['all_sessions_empty'] and
        not (B/'native16-warmed-attribution-v2').exists(),'preserve failed cohort and retire its generation')
    closed,new,proof=identities();revision=r.validation.git('rev-parse','HEAD').decode().strip()
    rt=r.validation.committed(r.RESULT)
    overrides=dict(STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(ROOT/'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'),
        TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    row=dict(stage='prepared',source_commit=revision,controller_inputs=q.gen.benchmark_pins(revision),
        native_build=proof,native_reference=dict(build_key=BUILD_KEY,source_commit=new['source_commit'],
            record_sha256=proof['record']['sha256'],compiled_inputs=proof['compiled_inputs'],
            executable=proof['fixtures']['sycl_moe_gu_swiglu_bench']),
        runtime_files=closed['runtime_files'],runtime_manifest=closed['runtime_manifest'],
        runtime_sha256=digest(r.RESULT),boot=rt['boot'],driver=rt['driver'],
        environment=dict(closed['environment'],**overrides),environment_overrides=overrides,
        environment_basis='native16_compatibility_smoke closed original environment and84-entry runtime catalogue',
        args=['--progress-public','1'],active=False,ownership_closed=True,complete=False,passed=False,
        model_executed=False,inference_adopted=False,performance_qualified=False,
        hardware_saturation_qualified=False,failed2048call_boundary_qualified=False,
        plan='One fresh traced process only. M160 public try-gu refusal,f16_out16 FP16input/BF16output '
            'through vllm/TLA then native swiglu. No warmup/adaptive repeats. Four groups of16calls '
            '(64public calls/128ordinary kernels),checked marker drain/query after each. At most32ordinary '
            'kernels plus checkpoint marker per group. Separate topology/reference/setup/readback operations '
            'are checked and drained. Emit immediate nested stage entry/return/throw records. A returned '
            'stage over3s fails after return;this cannot cancel an API. Only30s owner deadline can terminate '
            'the process. Stop first source/provider/environment/guard/validation/budget/owner/fault failure;'
            'no retry,reset,model test or speed/adoption claim. Recheck full original source/SDK proof before '
            'and after the one native process.',
        budgets=dict(wall=30,cpu=60,rss_cap=2<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        stage_return_max_ns=3000000000,
        limits='Smaller queued depth and added stage/API logging change the envelope. A pass is only a '
            'smoke,not reproduction or avoidance proof for the failed2048calls,ordinary runtime teardown '
            'without retained-loader preload,full numerics,model/legacy oneMKL ABI,decode or262144context. '
            'Stage elapsed time includes diagnostic output and is not isolated API/GPU service.',
        retention='Keep source/recipe/provider identities,stage-count/latency maxima,guard/dispatch results '
            'and any last unmatched stage/novel failure diagnostic. Retire normal full API/progress/JIT '
            'output and disposable objects after verified closure. Choose official v4fallback;retire extra '
            'diagnostic generation after original-source recovery unless a concrete imminent consumer is selected.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir()
    save(row);print('Prepared;commit exact stage plan before run.',flush=True)

def current(row):
    require(q.build.scope(r.validation.committed(RESULT))[KEY]==compact(row),'commit exact stage plan')
    require(q.gen.benchmark_pins(row['source_commit'])==row['controller_inputs'],'controller changed')
    closed,new,proof=identities()
    require(proof==row['native_build'] and closed['runtime_files']==row['runtime_files'],'native/runtime proof changed')
    require(digest(r.RESULT)==row['runtime_sha256'] and
        digest(row['runtime_manifest']['path'])==row['runtime_manifest']['sha256'],'runtime receipt changed')
    require(r.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot/driver changed')

def analyze(text):
    require('fixture_status=PASS mode=progress-public' in text,'native completion missing')
    values=[json.loads(v) for v in text.splitlines() if v.startswith('{')]
    stages=[v for v in values if v.get('kind')=='gu_stage'];stack=[];entered=0;last=0;summary={}
    for v in stages:
        require(type(v['monotonic_ns']) is int and v['monotonic_ns']>=last,'stage clock order')
        last=v['monotonic_ns'];key=(v['sequence'],v['stage'],v['group'],v['call'])
        if v['phase']=='enter':
            entered+=1;require(v['sequence']==entered,'stage sequence gap')
            stack.append((key,v['monotonic_ns'],v['elapsed_ns']))
        else:
            require(v['phase']=='return' and stack and stack[-1][0]==key,'stage failed/nesting differs')
            _,begin,entry_elapsed=stack.pop()
            require(v['elapsed_ns']-entry_elapsed==v['monotonic_ns']-begin,'stage elapsed does not conserve')
            require(0<=v['elapsed_ns']<=3000000000,'returned stage exceeds3s')
            s=summary.setdefault(v['stage'],dict(count=0,max_elapsed_ns=0));s['count']+=1
            s['max_elapsed_ns']=max(s['max_elapsed_ns'],v['elapsed_ns'])
    expected=dict.fromkeys(('runtime.init','queue.create','topology','case.create','public.init',
        'reference.prepare','public.retire','case.retire','final.drain'),1)
    expected.update({'public.gu.try':64,'public.gu.f16_out16':64,'public.gu.swiglu':64,
        'candidate.reset':4,'checkpoint.enqueue':4,'checkpoint.query':4,'guard.compare':4,
        'buffer.init.H2D':7,'buffer.set.H2D':6,'buffer.get.D2H':25,'event.wait':43,'queue.drain':10})
    require(not stack and entered==308 and
        {k:v['count'] for k,v in summary.items()}==expected,'incomplete stage graph/counts')
    require({v.get('kind') for v in values}=={'gu_stage','topology','public_progress_result'} and
        [v for v in values if v.get('kind')=='topology']==
        [dict(kind='topology',work_items=384,status='PASS')],'topology/check output differs')
    for name in ('public.gu.try','public.gu.f16_out16','public.gu.swiglu'):
        require([(v['group'],v['call']) for v in stages if v['phase']=='enter' and v['stage']==name]==
            [(g,c) for g in range(4) for c in range(16)],'actual64call route/count differs')
    for name in ('candidate.reset','checkpoint.enqueue','checkpoint.query','guard.compare'):
        require(summary[name]['count']==4,'four checkpoint groups required')
    result=[v for v in values if v.get('kind')=='public_progress_result']
    require(len(result)==1 and result[0]==dict(kind='public_progress_result',rows=160,groups=4,
        calls_per_group=16,calls=64,selected=0,fallback=64,max_pending_ordinary_kernels=32,
        checkpoint_markers_per_group=1,max_half_ulp=result[0]['max_half_ulp'],guards=True,
        inputs_immutable=True,status='PASS') and result[0]['max_half_ulp']<=1,'guard/dispatch result differs')
    return dict(stage_entries=entered,stage_returns=entered,stage_summary=summary,result=result[0],
        limits='Logged stage durations are diagnostic-only,not speed or saturation evidence.')

def run(row):
    current(row);require(row['stage']=='prepared' and not (ROOT/'owned').exists(),'single-use native run')
    initial=copy.deepcopy(row);owner=None;outcome=dict(host_probe_accepted=False)
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    def persist():write(ROOT/'execution.json',dict(commands=owner.commands if owner else [],active=bool(owner and owner.active)))
    try:
        owner=r.supervision.Owner(ROOT/'owned',persist);artifact=row['native_reference']['executable']
        allowed=dict(row['runtime_files']);allowed[artifact['path']]=artifact['sha256'];seen=set();env_seen=False
        def monitor(io,entry):
            nonlocal env_seen
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                try:
                    mapped=r.executable_mappings(Path('/proc',str(io.proc.pid),'maps').read_text())
                    actual=dict(v.split(b'=',1) for v in Path('/proc',str(io.proc.pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode())==v.encode() for k,v in row['environment'].items()) and
                        set(actual)-{k.encode() for k in row['environment']}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},
                        'native environment differs')
                    env_seen=True
                except (FileNotFoundError,ProcessLookupError):mapped=set()
                except PermissionError:
                    require(io.exit_watch.poll(20),'live native /proc denied');mapped=set()
                require(mapped<=set(allowed),'unplanned executable map: '+repr(mapped-set(allowed)));seen.update(mapped)
                require(r.supervision.artifact_bytes(ROOT)<=64<<20,'whole diagnostic byte budget')
                if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
            io.exit(1)
        entry,out,err=owner.run('stage-smoke',[artifact['path'],*row['args']],row['environment'],P,
            interaction=monitor,**row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code']==0,'native stage process failed')
        activation=r.validation.preload_evidence(err.read_text(),entry['pid'])
        require(activation['loader']==row['environment']['STRATA_UR_PROBE_EXPECTED_LOADER'] and
            env_seen and activation['loader'] in seen and
            str(q.build.OFFICIAL/'lib/libsycl.so.9.0.0') in seen and
            {p for p in seen if '/libze_intel_gpu.so.' in p}=={row['environment']['ZE_ENABLE_ALT_DRIVERS']},
            'actual runtime provider evidence missing')
        require(not (ROOT/'validation.log').exists() or not (ROOT/'validation.log').stat().st_size,'parameter warning')
        require('urEnqueue' in (ROOT/'ur.trace').read_text(),'full small API trace missing')
        row['details']=analyze(out.read_text());current(initial);outcome['host_probe_accepted']=True
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner:r.validation.finish(outcome,owner,cursor,row['boot'])
        row.update(stage='closed' if outcome.get('passed') else 'failed',complete=bool(outcome.get('passed')),
            passed=bool(outcome.get('passed')),active=False,ownership_closed=outcome.get('ownership_closed',True),
            outcome=outcome,GPU_executed=bool(owner and owner.commands))
        persist();save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','complete','passed','ownership_closed','outcome')}),flush=True)
    return int(not row['passed'])

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare();return 0
        if sys.argv[1:]==['run']:return run(json.loads((ROOT/'record.json').read_text()))
        raise RuntimeError('use prepare or run')

if __name__=='__main__':sys.exit(main())
