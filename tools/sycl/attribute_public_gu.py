#!/usr/bin/env python3
"""Main-owned warmed public-GU attribution; repository LICENSE. No model run."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import qualify_native16_control as q
from command_service import parse_profiles, fields

r, P, B, RESULT = q.r, q.P, q.B, q.RESULT
ROOT = B / 'public-gu-warmed-attribution-v2'
NATIVE = B / 'native16-warmed-attribution-v2'
BUILD_KEY = 'public_gu_warmed_attribution_build_v2'
KEY = 'public_gu_warmed_attribution_v2'
require, digest, write = q.require, q.digest, q.write
COLUMNS = ['rows','round','repeat','selected','fallback','wall_ns','queue_interval_ns',
    'charged_wall_ns','wrapper_cpu_ns','wrapper_wall_ns','total_cpu_ns','total_wall_ns',
    'wait_cpu_ns','wait_wall_ns','wrapper_wait_cpu_ns','wrapper_wait_wall_ns',
    'inner_submit_cpu_ns','inner_submit_wall_ns','inner_submit_count','wait_count','max_half_ulp']
GPU_COLUMNS = ['rows','round','job','kind','service_ns','start_minus_submit_ns']

def compact(row):
    return {k:v for k,v in row.items() if k not in ('controller_inputs','runtime_files')}

def save(row):
    raw=RESULT.read_text(); desired=json.loads(raw); q.build.scope(desired)[KEY]=compact(row)
    value=json.dumps(compact(row),indent=2)
    for match in reversed(list(re.finditer(r'"(?:samples|gpu_samples)": \[',value))):
        first=match.end()-1; table,size=json.JSONDecoder().raw_decode(value[first:])
        if not table: continue
        require(all(isinstance(v,list) for v in table),'sample table type differs')
        indent=' '*(match.start()-value.rfind('\n',0,match.start())-1)
        rendered='[\n'+',\n'.join(indent+'  '+json.dumps(v,separators=(',',':')) for v in table)+'\n'+indent+']'
        value=value[:first]+rendered+value[first+size:]
    value=value.replace('\n','\n      '); literal=json.dumps(KEY)+':'
    if KEY in q.build.scope(json.loads(raw)):
        require(raw.count(literal)==1,'ambiguous result key')
        first=raw.index(literal)+len(literal)
        while raw[first].isspace(): first+=1
        _,count=json.JSONDecoder().raw_decode(raw[first:])
        raw=raw[:first]+value+raw[first+count:]
    else:
        anchor='"fused_command_profile": {'
        require(raw.count(anchor)==1,'ambiguous result container')
        raw=raw.replace(anchor,anchor+'\n      '+literal+' '+value+',',1)
    require(json.loads(raw)==desired,'scoped update changed unrelated evidence')
    write(ROOT/'record.json',row); RESULT.write_text(raw)

def identities():
    closed=json.loads((q.ROOT/'record.json').read_text()); q.current(closed)
    old,_,proof=q.native()
    new,_,new_proof=q.native(NATIVE,BUILD_KEY)
    require(new['baseline_native_build']==proof,'compiled baseline changed')
    return closed,old,new,new_proof

def prepare():
    require(not ROOT.exists() and KEY not in q.build.scope(r.validation.committed(RESULT)),'single-use cohort')
    closed,old,new,proof=identities(); rt=r.validation.committed(r.RESULT)
    revision=r.validation.git('rev-parse','HEAD').decode().strip()
    cases=[dict(name=c['name'],target=c['target'],args=c['args'],arm=c.get('gu_arm',1),
        down_arm=c.get('down_arm',1),mode='qualification') for c in closed['cases']
        if c['name'] in ('host-canary','submission','GU-smoke','GU-public0','GU-public1')]
    unchanged=[name for name in new['fixtures'] if name!='sycl_moe_gu_swiglu_bench']
    require(all(new['fixtures'][name]['sha256']==old['fixtures'][name]['sha256'] for name in unchanged),
        'unchanged fixture reuse requires exact original three executable hashes')
    def compile_map(native):
        root=str(Path(native['compiled_inputs']['path']).parent)
        return {p.replace(root,'<fixture-root>'):sha for p,sha in
            json.loads(Path(native['compiled_inputs']['path']).read_text()).items()}
    old_inputs,new_inputs=compile_map(old),compile_map(new)
    changed=[p for p in set(old_inputs)|set(new_inputs) if old_inputs.get(p)!=new_inputs.get(p)]
    require(changed==[str(P/'bench/sycl/moe_gu_swiglu.cpp')],
        'reused fixture native compile inputs changed outside the GU executable')
    require([v.replace(str(NATIVE),'<fixture-root>') for v in new['configure_argv']]==
        [v.replace(str(q.build.ROOT),'<fixture-root>') for v in old['configure_argv']] and
        new['sdk']==old['sdk'],'component compiler/recipe changed')
    # Fixed balanced startup observer order;new batch envelope in both arms.
    for block,order in enumerate((('off','on'),('on','off'),('off','on'))):
        for mode in order:
            cases.append(dict(name=mode+str(block),target='sycl_moe_gu_swiglu_bench',
                args=['--attribute-public' if mode=='on' else '--bench-public',str(block),'1'],
                arm=1,down_arm=1,mode=mode,block=block))
    row=dict(stage='prepared',source_commit=revision,controller_inputs=q.gen.benchmark_pins(revision),
        old_native=closed['native_build'],new_native=proof,
        reused_unchanged_fixture_qualification=dict(names=unchanged,original_source=old['source_commit'],
            reused_GPU_qualification=True,new_GPU_qualification_claimed=False,
            changed_compile_inputs=changed,
            basis='Same three ELF hashes,original native inputs recoverable,unchanged compiler/provider;'
                'the changed GU executable receives fresh traced qualification.'),
        runtime_files=closed['runtime_files'],runtime_manifest=closed['runtime_manifest'],
        runtime_sha256=digest(r.RESULT),boot=rt['boot'],driver=rt['driver'],
        environment=closed['environment'],canary=closed['canary'],cases=cases,processes=[],
        sample_columns=COLUMNS,gpu_sample_columns=GPU_COLUMNS,
        active=False,ownership_closed=True,complete=False,passed=False,model_executed=False,
        inference_adopted=False,hardware_saturation_qualified=False,
        question='Match warmed same-batch GPU service with whole public-wrapper thread CPU,'
            'with same-binary observer cost;no direct old-short-batch timing comparison.',
        plan='Five ordered host/submission/GU traced compatibility cases;reuse the byte-identical three '
            'original fixture qualifications. Then three balanced fresh-process pairs '
            'off/on;on/off;off/on. All timed processes use startup GUlarge. Fixed2048/512calls '
            'at M160/2333/2334,initial timing batch plus five later batches. Keep every sample. '
            'Sparse deterministic four-job GPU samples per batch;actual eligible event count must '
            'conserve all repeated operations. Both wall/queue batches require100ms;endpoint controls remain1percent. '
            'CPU is partitioned into inner-submit outside explicit waits,'
            'remaining public-wrapper outside explicit waits,all explicit waits,and remaining batch work. '
            'Stop first source,provider,environment,oracle,guard,budget,ownership or fresh fault failure. '
            'Full original-Git/SDK admission brackets the cohort;each exec rechecks immutable source/build '
            'receipts,fixture/imported-link bytes,controllers and actual runtime providers. '
            'Speed gate failure is assessed after all declared samples,without a retry or replacement.',
        gates=dict(batch_wall_and_queue_min_ns=100000000,endpoint_max_fraction=.01,
            timer_resolution_max_fraction=.001,
            charged_on_off_each_max=1.02,charged_on_off_geomean_max=1.01),
        limits='Projection/activation fixture only,not the original model runtime or historical cause. '
            'Longer fixed batches change cache/spacing/power history;do not compare directly with old-short-batch speed. '
            'Sparse GPU service is not total GPU busy or an additive critical path. Whole-call CPU includes '
            'ownership/context/span checks;remaining-wrapper is not isolated pointer-check time. '
            'Library-internal polling/blocking outside explicit Wait wrappers may remain in other categories. '
            'Trace planning,queries and footer output are charged by the outer clock;parameter validation,'
            'retained-loader preload and external ownership/provider observation remain charged in all arms. '
            'Their separate historical cost is not isolated. Initial timing batch follows warmup,not cold start.',
        budgets=dict(wall=300,cpu=1200,rss_cap=4<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        whole_bytes=256<<20,
        retention='Keep useful individual samples,source/runtime identities,CPU/event count conservation and '
            'decisions. Retire full successful traces,JIT,process/build logs and disposable objects after '
            'verification. Select one qualified official component generation after comparison closure;original13 '
            'engine assets remain untouched. No whole-model or decode adoption.')
    ROOT.mkdir(mode=0o700);save(row)
    print('Prepared;commit the exact comparison plan before run.',flush=True)

def current(row,full=False):
    require(q.build.scope(r.validation.committed(RESULT))[KEY]==compact(row),'commit exact cohort plan')
    require(q.gen.benchmark_pins(row['source_commit'])==row['controller_inputs'],'controller changed')
    if full:
        closed,old,new,proof=identities()
        require(closed['native_build']==row['old_native'] and proof==row['new_native'] and
            closed['runtime_files']==row['runtime_files'],'source/runtime qualification changed')
    # Full original-Git/SDK admission brackets the cohort. Before every exec,
    # check exact immutable build/closure receipts,all fixture and imported-link
    # bytes,all current controllers and all actual runtime providers again.
    for native in (row['old_native'],row['new_native']):
        for receipt in (native['record'],native['compiled_inputs']):
            require(digest(receipt['path'])==receipt['sha256'],'native source/recipe receipt changed')
        for artifact in native['fixtures'].values():
            require(digest(artifact['path'])==artifact['sha256'] and
                Path(artifact['path']).stat().st_size==artifact['bytes'],'native fixture changed')
        for path,sha in native['imported_link_files'].items():
            require(digest(path)==sha,'imported link file changed')
    require(digest(r.RESULT)==row['runtime_sha256'] and
        digest(row['runtime_manifest']['path'])==row['runtime_manifest']['sha256'],'runtime proof changed')
    require(all(digest(p)==sha for p,sha in row['runtime_files'].items()),'runtime provider changed')
    require(r.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot/driver changed')

def analyze(case,out,err):
    require('fixture_status=PASS mode='+('attribute-public' if case['mode']=='on' else 'bench-public') in out,
        'timing fixture failed')
    values=[json.loads(v) for v in out.splitlines() if v.startswith('{')]
    require(values and all(v.get('status')=='PASS' for v in values),'numerical/topology check failed')
    samples=[v for v in values if v.get('kind')=='public_sample']
    controls=[v for v in values if v.get('kind')=='public_control']
    require([(v['rows'],v['round']) for v in samples]==[(m,t) for m in (160,2333,2334) for t in (-1,0,1,2,3,4)],
        'batch identities/count differ')
    require([(v['phase'],v['index']) for v in controls]==[(p,i) for p in (0,1) for i in range(8)],'control count differs')
    for v in samples:
        repeat=2048 if v['rows']==160 else 512; selected=repeat if v['rows']>=2334 else 0
        require(v['process']==case['block'] and v['arm']==1 and v['repeat']==repeat and
            v['selected']==selected and v['fallback']==repeat-selected and v['eligible'] and
            min(v['wall_ns'],v['queue_interval_ns'])>=1e8 and v['max_half_ulp']<=1,'batch admission differs')
        require(v['scope']=='public_dispatch_queue_interval_includes_host_gaps_not_GPU_service','queue scope changed')
    require(max(v['wall_ns'] for v in controls)<=min(v['wall_ns'] for v in samples)*.01 and
        max(v['timer_resolution_ns'] for v in controls)<=min(v['queue_interval_ns'] for v in samples)*.001,
        'clock/control eligibility differs')
    gpu=[]; inversions=0
    observed=case['mode']=='on'
    if observed:
        profiles=parse_profiles(err,expected_windows=1)
        cpu=[fields(v,'strata prefill cpu wait: ') for v in err.splitlines() if v.startswith('strata prefill cpu wait: ')]
        host=[fields(v,'strata prefill cpu host: ') for v in err.splitlines() if v.startswith('strata prefill cpu host: ')]
        require(len(profiles)==len(cpu)==len(host)==len(samples)==18,'matched CPU/GPU batch count differs')
        for v,p,c,h in zip(samples,profiles,cpu,host):
            kinds=['gu_swiglu'] if v['rows']>=2334 else ['gu','swiglu']
            require(v['observed']==1 and v['inner_submit_count']==v['repeat'] and v['wait_count']>=1 and
                c['version']=='4' and c['available']==c['complete']=='1' and
                c['clock_or_accounting_error']==c['open_waits']==c['wait_throws']=='0' and
                h['available']=='1' and h['open_regions']=='0','incomplete CPU domain')
            require({g['kind'] for g in p['groups']}==set(kinds) and
                all(int(g['eligible'])==v['repeat'] and int(g['selected'])==4 for g in p['groups']),
                'sampled/eligible event counts differ')
            by_kind={k:[s for s in p['samples'] if s['kind']==k] for k in kinds}
            require(all(len(a)==4 and all(int(s['m'])==v['rows'] for s in a) for a in by_kind.values()),
                'four sampled jobs per operation required')
            require(all(sorted(int(s['row']) for s in a)==sorted(int(s['row']) for s in by_kind[kinds[0]])
                for a in by_kind.values()),'GU/activation sample jobs differ')
            # Independent scope footer brackets the same interval plus its tiny
            # begin/end checks. Preserve the difference rather than clamp it.
            require(abs(float(c['main_cpu_s'])*1e9-v['total_cpu_ns'])<=v['total_cpu_ns']*.01+20000,
                'whole CPU domain/footer mismatch')
            require(abs(float(c['wait_cpu_s'])*1e9-v['wait_cpu_ns'])<=100 and
                int(c['wait_count'])==v['wait_count'] and int(h['gemm_submit_count'])==v['repeat'],
                'wait/submit conservation differs')
            tolerance=max(32,v['total_cpu_ns']*2e-5)
            require(v['wrapper_cpu_ns']-v['wrapper_wait_cpu_ns']-v['inner_submit_cpu_ns']>=-tolerance and
                v['total_cpu_ns']-v['wrapper_cpu_ns']-v['wait_cpu_ns']+v['wrapper_wait_cpu_ns']>=-tolerance,
                'reported CPU partition negative')
            for s in p['samples']:
                gpu.append([v['rows'],v['round'],int(s['row']),s['kind'],s['service_ns'],s['start_minus_submit_ns']])
            inversions+=int(p['summaries'][0]['submit_clock_inversions'])
    else:
        require('strata command ' not in err and 'strata prefill cpu wait:' not in err,'unexpected observer')
    return dict(samples=[[v.get(k) for k in COLUMNS] for v in samples],gpu_samples=gpu,
        controls=16,control_max_wall_ns=max(v['wall_ns'] for v in controls),
        timer_resolution_ns=max(v['timer_resolution_ns'] for v in controls),
        full_event_eligible_sampled_conservation=observed,complete_wrapper_CPU_partition=observed,
        submit_clock_inversions=inversions)

def summarize(processes,gates):
    by={p['name']:p for p in processes}; result=[]
    for m in (160,2333,2334):
        wall={};charged={};cpu_parts={};service={}
        for mode in ('off','on'):
            for i in range(3):
                name=mode+str(i); p=by[name]['details']
                rows=[dict(zip(COLUMNS,v)) for v in p['samples'] if v[0]==m and v[1]>=0]
                wall[name]=statistics.median(v['wall_ns']/v['repeat'] for v in rows)
                charged[name]=statistics.median(v['charged_wall_ns']/v['repeat'] for v in rows)
                if mode=='on':
                    cpu_parts[name]=dict(inner_submit_ns=statistics.median(v['inner_submit_cpu_ns']/v['repeat'] for v in rows),
                        other_wrapper_ns=statistics.median((v['wrapper_cpu_ns']-v['wrapper_wait_cpu_ns']-v['inner_submit_cpu_ns'])/v['repeat'] for v in rows),
                        explicit_wait_ns=statistics.median(v['wait_cpu_ns']/v['repeat'] for v in rows),
                        other_batch_ns=statistics.median((v['total_cpu_ns']-v['wrapper_cpu_ns']-v['wait_cpu_ns']+v['wrapper_wait_cpu_ns'])/v['repeat'] for v in rows),
                        whole_thread_ns=statistics.median(v['total_cpu_ns']/v['repeat'] for v in rows))
                    events=[v for v in p['gpu_samples'] if v[0]==m and v[1]>=0]
                    service[name]={k:dict(median_ns=statistics.median(v[4] for v in events if v[3]==k),
                        range_ns=[min(v[4] for v in events if v[3]==k),max(v[4] for v in events if v[3]==k)],
                        samples=sum(v[3]==k for v in events)) for k in sorted({v[3] for v in events})}
        on_off=[charged['on'+str(i)]/charged['off'+str(i)] for i in range(3)]
        observer_pass=max(on_off)<=gates['charged_on_off_each_max'] and statistics.geometric_mean(on_off)<=gates['charged_on_off_geomean_max']
        result.append(dict(rows=m,whole_call_wall_ns_by_process=wall,charged_wall_ns_by_process=charged,
            charged_on_off_ratios=on_off,charged_on_off_geomean=statistics.geometric_mean(on_off),
            observer_cost_gate_passed=observer_pass,
            CPU_parts_per_call_ns=cpu_parts,warmed_GPU_sample_service=service))
    passed=all(v['observer_cost_gate_passed'] for v in result)
    return dict(rows=result,performance_qualified_for_this_fixture=passed,
        decision='Matched component attribution only when its prospectively fixed same-binary cost gate passes;no model adoption or historical cause.',
        next='Use qualified complete-wrapper and warmed-service evidence to select a bounded source/runtime '
            'counterfactual. If a cost gate fails,retain failure and do not advance this observer to a model.')

def run(row):
    current(row,full=True);require(row['stage']=='prepared' and not (ROOT/'owned').exists(),'single-use cohort')
    initial=copy.deepcopy(row);owner=None;outcome=dict(host_probe_accepted=False)
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist():
        write(ROOT/'execution.json',dict(processes=row['processes'],active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner=r.supervision.Owner(ROOT/'owned',persist)
        for case in row['cases']:
            current(initial)
            root=ROOT/case['name'];root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
            qualification=case['mode']=='qualification'; observed=case['mode']=='on'
            env=dict(row['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                STRATA_SYCL_PREFILL_GU_SWIGLU='large' if case['arm'] else '0',
                STRATA_SYCL_PREFILL_DOWN_SCATTER='large' if case['down_arm'] else '0',
                STRATA_SYCL_PREFILL_COMMAND_PROFILE='1' if observed else '0',
                STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='1' if observed else '0',
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if qualification else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
            native=row['new_native']
            artifact=native['fixtures'][case['target']] if case['target'] else row['canary']
            allowed=dict(row['runtime_files']);allowed[artifact['path']]=artifact['sha256'];seen=set();env_seen=False
            def monitor(io,entry):
                nonlocal env_seen
                next_size=0
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try:
                        mapped=r.executable_mappings(Path('/proc',str(io.proc.pid),'maps').read_text())
                        actual=dict(v.split(b'=',1) for v in Path('/proc',str(io.proc.pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                        require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and
                            set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment changed')
                        env_seen=True
                    except (FileNotFoundError,ProcessLookupError):mapped=set()
                    except PermissionError:
                        require(io.exit_watch.poll(20),'live native /proc access denied');mapped=set()
                    require(mapped<=set(allowed),'unplanned executable map: '+repr(mapped-set(allowed)));seen.update(mapped)
                    if time.monotonic()>=next_size:
                        require(r.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],'whole byte budget')
                        next_size=time.monotonic()+.5
                    if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
                io.exit(1)
            entry,out,err=owner.run(case['name'],[artifact['path'],*case['args']],env,P,interaction=monitor,**row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code']==0,'native process failed: '+case['name'])
            activation=r.validation.preload_evidence(err.read_text(),entry['pid'])
            require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'wrong pre-main loader')
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'parameter warning')
            if case['target']:
                require(env_seen and env['STRATA_UR_PROBE_EXPECTED_LOADER'] in seen and
                    str(q.build.OFFICIAL/'lib/libsycl.so.9.0.0') in seen and
                    {p for p in seen if '/libze_intel_gpu.so.' in p}=={env['ZE_ENABLE_ALT_DRIVERS']},'provider evidence missing')
                if qualification:require('urEnqueue' in (root/'ur.trace').read_text(),'small API trace missing')
            if qualification:
                detail=q.analyze(dict(case,gu_arm=case['arm']),out.read_text(),entry['pid'])
            else:detail=analyze(case,out.read_text(),err.read_text())
            check=dict(host_probe_accepted=True);r.validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after']==row['driver'],'owner/fault gate failed')
            row['processes'].append(dict(case,passed=True,ownership_closed=True,details=detail));persist()
            print('Passed '+case['name'],flush=True)
        current(initial,full=True);row['analysis']=summarize(row['processes'],row['gates']);outcome['host_probe_accepted']=True
    except BaseException as error:
        outcome['error']=repr(error)
        if owner and owner.commands:outcome['last_case']=owner.commands[-1]['label']
    finally:
        if owner:r.validation.finish(outcome,owner,cursor,row['boot'])
        passed=bool(outcome.get('passed') and len(row['processes'])==len(row['cases']))
        row.update(stage='closed' if passed else 'failed',passed=passed,complete=passed,active=False,
            ownership_closed=outcome.get('ownership_closed',True),outcome=outcome,GPU_executed=bool(owner and owner.commands))
        persist();save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','outcome')}),flush=True)
    return int(not row['passed'])

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare();return 0
        if sys.argv[1:]==['run']:return run(json.loads((ROOT/'record.json').read_text()))
        raise RuntimeError('use prepare or run')

if __name__=='__main__':sys.exit(main())
