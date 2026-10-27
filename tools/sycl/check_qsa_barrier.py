#!/usr/bin/env python3
"""Main-owned source-bound QSA barrier qualification; repository LICENSE."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_qsa_barrier as build
import measurement_guards as guards
import measurement_owner as supervision
import profile_aggregate as tables
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,ROOT,RESULT=build.P,build.B,build.ROOT,build.RESULT
SEQUENCE=('oracle-smoke','oracle-component','smoke','resources','perf')


def inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen: continue
        seen.add(path);name=str(path.relative_to(P));raw=build.gen.git('show',revision+':'+name)
        guards.require(raw==path.read_bytes(),'commit controller input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file(): pending.append(local)
    return pins


def assets():
    data=validation.committed(build.gen.A/'prefill-resources-current.json',('event_reconciliation',))
    host=data['selected_qsa_environment_host_v1']
    guards.require(host['passed'] and host['ownership_closed'] and host['session_removed'] and
        host['observation']['passed'] and host['observation']['different_keys']==[], 'qualified launcher environment required')
    value={key:data['native_observation']['assets'][key] for key in ('launcher','collector')}
    for a in value.values(): guards.require(build.digest(a['path'])==a['sha256'],'collector changed')
    return value


def runtime(plan):
    _,proof=build.qualified_baseline()
    guards.require(proof==plan['runtime_qualification'],'qualified runtime changed')
    rt=validation.committed(build.qualified_runtime.RESULT)
    loader=rt['preload']['loader']
    guards.require(type(loader) is str,'qualified loader schema changed')
    preload=dict(library=rt['preload']['library'],loader=dict(path=loader,sha256=rt['runtime_files'][loader]))
    guards.require(preload==plan['preload'],'private preload/provider changed')
    return rt


def contract():
    row=validation.committed(build.gen.A/'research-index.json',('qsa_iteration_barrier_fusion_contract_20261027',))
    guards.require(row['status']=='reviewed_accept_conditional' and row['worker_closed'] and
        row['source_revision']=='17bc703d94c792a4d23f45f53359e601ad46348c' and
        row['reviewer']=='native Codex gpt-6-luna max','closed independent source-contract review required')
    report=P/row['report']
    guards.require(report.stat().st_size==row['report_bytes'] and build.digest(report)==row['report_sha256'],
        'reviewed contract reference changed')
    return dict(source_revision=row['source_revision'],report_sha256=row['report_sha256'],
        status=row['status'],worker_closed=True)


def compiled_barriers(record):
    result={}
    for arm,asset in record['device_IR'].items():
        guards.require(build.digest(asset['path'])==asset['sha256'],'compiled IR changed')
        text=Path(asset['path']).read_text()
        bodies=re.findall(r'^define [^\n]*prefix_batch_implILi33ELb0ELb0ELb1ELb1ELb1ELb1ELb0EEE[^\n]*\{\n(.*?)^}',text,re.M|re.S)
        bodies=[b for b in bodies if 'ControlBarrier' in b]
        guards.require(len(bodies)==1,'one selected ordered-exp attention IR required')
        blocks={};name='entry';blocks[name]=[]
        for line in bodies[0].splitlines():
            label=re.match(r'^([-A-Za-z0-9_.$]+):',line)
            if label:name=label[1];blocks[name]=[]
            blocks[name].append(line)
        edges={name:set(re.findall(r'label %([-A-Za-z0-9_.$]+)','\n'.join(lines))) for name,lines in blocks.items()}
        guards.require(all(v<=set(blocks) for v in edges.values()),'unresolved compiled CFG edge')
        cyclic=set()
        for start in blocks:
            pending=list(edges[start]);seen=set()
            while pending:
                node=pending.pop()
                if node==start:cyclic.add(start);break
                if node not in seen:seen.add(node);pending.extend(edges[node])
        counts={name:sum('call ' in line and 'ControlBarrier' in line for line in lines) for name,lines in blocks.items()}
        result[arm]=dict(static=sum(counts.values()),inside_cycles=sum(counts[n] for n in cyclic))
    guards.require(result=={'original':dict(static=7,inside_cycles=5),
        'control':dict(static=7,inside_cycles=5),'merged':dict(static=7,inside_cycles=4)},
        'compiled barrier placement differs from proposal')
    return result


def observe(pid,entry,record,plan,job,rt,stage):
    ident=supervision.identity(pid)
    guards.require(ident and ident['sid']==entry['pid'] and
        str(Path('/proc',str(pid),'exe').resolve())==str(Path(record['binaries'][job['binary']]['path']).resolve()), 'native ownership/executable differs')
    actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
    expected=job['environment']
    guards.require(all(actual.get(k.encode())==v.encode() for k,v in expected.items() if k!='LD_PRELOAD'),'native environment changed')
    preloads=[str(Path(p).resolve()) for p in actual.get(b'LD_PRELOAD',b'').decode().split(':') if p]
    wanted=[str(Path(plan['preload']['library']['path']).resolve())]
    allowed=set(rt['runtime_files'])|{str(Path(record['binaries'][job['binary']]['path']).resolve())}
    if job['session']:
        wanted.insert(0,str(Path(plan['collector_assets']['collector']['path']).resolve()));allowed.add(wanted[0])
        for key in ('ZET_ENABLE_METRICS','UNITRACE_KernelMetrics','UNITRACE_MetricQuery','UNITRACE_MetricSampling','UNITRACE_StallSampling','XPTI_TRACE_ENABLE'):
            guards.require(actual.get(key.encode(),b'0') in (b'0',b''),'unplanned collector mode')
    else:
        guards.require(set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'unexpected startup environment')
    guards.require(preloads==wanted,'native private preload identity/order differs')
    mapped=build.qualified_runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
    guards.require(mapped<=allowed,'unexpected native executable mapping: '+str(sorted(mapped-allowed)))
    if stage=='runtime-ready':
        guards.require({expected['ZE_ENABLE_ALT_DRIVERS'],plan['preload']['loader']['path'],*wanted}<=mapped,'qualified runtime/provider missing')



def admission(record,plan):
    build.admission(record,True)
    guards.require(record['stage']=='built' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'],'closed build required')
    guards.require(inputs(plan['source_commit'])==plan['controller_inputs'],'controller changed')
    guards.require(contract()==plan['contract_qualification'] and compiled_barriers(record)==plan['compiled_barriers'],
        'contract or compiled source qualification changed')
    rt=runtime(plan)
    if plan['mode']=='resources':guards.require(assets()==plan['collector_assets'],'collector changed')
    return rt


def prepare(mode):
    record=build.gen.committed(RESULT);checks=record.get('checks',[])
    position=SEQUENCE.index(mode)
    guards.require(len(checks)==position and all(c['mode']==SEQUENCE[i] and c['passed'] and
        c['complete'] and c['ownership_closed'] and c['fresh_faults']==[] for i,c in enumerate(checks)),
        'ordered independent oracle/smoke/resource/performance gates;no retry')
    root=ROOT/('check-'+mode);guards.require(not root.exists(),'fresh check root only')
    _,proof=build.qualified_baseline();rt=validation.committed(build.qualified_runtime.RESULT)
    loader=rt['preload']['loader'];guards.require(type(loader) is str,'explicit loader path required')
    revision=build.gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(stage='prepared',mode=mode,source_commit=revision,controller_inputs=inputs(revision),
        contract_qualification=contract(),compiled_barriers=compiled_barriers(record),
        runtime_qualification=proof,preload=dict(library=rt['preload']['library'],loader=dict(path=loader,sha256=rt['runtime_files'][loader])),
        gates=record['plan']['qualification'],
        budgets=dict(wall=600 if mode=='perf' else 300,cpu=3600,rss_cap=8<<30,text_cap=16<<20,
            file_cap=64<<20,total_cap=192<<20),jobs=[])
    if mode=='resources':plan.update(collector_assets=assets(),gates=record['plan']['resources'])
    elif mode=='perf':plan['gates']=record['plan']['performance']
    admission(record,plan);root.mkdir(mode=0o700)
    base=build.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    for index in range(1 if mode=='resources' else 3):
        folder=root/str(index);folder.mkdir()
        for name in ('tmp','cache'):(folder/name).mkdir()
        env=dict(base,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),ZES_ENABLE_SYSMAN='0',
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            LD_PRELOAD=plan['preload']['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=loader,
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
        if mode in ('oracle-smoke','smoke'):env['UR_LOG_TRACING']='level:info;flush:info;output:file,'+str(folder/'api.trace')
        else:env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
        binary='oracle' if mode.startswith('oracle-') else 'fixture'
        if binary=='oracle':
            env['STRATA_TEST_MAIN_GATES']='1'
            command=[record['binaries'][binary]['path'],'--smoke' if mode=='oracle-smoke' else '--component',str(index)]
        else:
            command=[record['binaries'][binary]['path'],'--bench' if mode=='perf' else '--smoke',record['selected_fixture']['path']]
            if mode=='perf':command.append(str(index))
        session=None
        if mode=='resources':
            session='StrataQsaBarrier'+secrets.token_hex(12);(folder/'trace').mkdir()
            env.pop('LD_PRELOAD');env.update(STRATA_UNITRACE_TARGET_PRELOAD=plan['preload']['library']['path'],
                UNITRACE_FollowChildProcess='1',UNITRACE_SystemTime='1',UNITRACE_TraceOutputDir=str(folder/'trace'))
            command=[plan['collector_assets']['launcher']['path'],'--session',session,'--device-timing','--verbose',*command]
        plan['jobs'].append(dict(index=index,binary=binary,environment=env,command=command,session=session))
    record['check_plan']=plan;build.write(root/'plan.json',plan);build.write(RESULT,record)
    print('Prepared '+mode+';commit exact plan before execution.',flush=True)


def resources(text):
    table=tables.device_summary(text);counts={};properties={}
    def arm(name):
        if '::barrier_control::' in name:return 1
        if '::barrier_merged::' in name:return 2
        return 0
    def selected(name):return 'prefix_batch_impl<33,' in name and '[SIMD16 ' in name and '{128; 1; 1}]' in name
    for row in table['rows']:
        if selected(row['name']):
            a=arm(row['name']);guards.require(a not in counts,'duplicate attention family');counts[a]=row['calls']
    for line in text.split('=== Kernel Properties ===')[1].splitlines():
        parts=[p.strip() for p in line.rsplit(',',7)]
        if len(parts)!=8 or not selected(parts[0]):continue
        a=arm(parts[0]);guards.require(a not in properties,'duplicate attention descriptor')
        simd,nargs,slm,private,spill,grf=map(int,parts[2:])
        guards.require(parts[1]=='JIT' and simd==16 and nargs==18 and grf==128,'attention geometry changed')
        properties[a]=dict(SIMD=simd,arguments=nargs,raw_SLM=slm,private_memory_descriptor=private,spill_memory_descriptor=spill,GRF=grf)
    guards.require(counts=={0:2,1:2,2:2} and set(properties)=={0,1,2} and properties[0]==properties[1],
        'original/control resources or coverage differs')
    guards.require(properties[0]['spill_memory_descriptor']==12480 and
        properties[0]['raw_SLM']==properties[2]['raw_SLM'],'original spill or merged raw SLM differs')
    return dict(calls=counts,properties=properties,conserved_device_ns=table['total_device_ns'],table_rows=len(table['rows']),
        limits='Spill/private are compiler descriptors,not measured traffic,occupancy or pure barrier instruction cost.')


def oracle_cases(text,mode,arm):
    found=[]
    for line in text.splitlines():
        if not line.startswith('case cells='):continue
        fields=dict(token.split('=',1) for token in line.split()[1:] if '=' in token)
        for name in ('packed_logical_prefix','packed_guards','guards','inputs','status','exact_barrier_arms'):
            guards.require(fields.get(name)=='PASS','oracle coverage failed: '+name)
        guards.require(fields['oracle']=='BF16_operands_FP32_online' and int(fields['tested_arm'])==arm and
            int(fields['packed_guard_elements'])==17 and int(fields['packed_k_mod8'])%8==2 and
            int(fields['compared'])==int(fields['queries'])*24*256 and
            0<=float(fields['max_budget_ratio'])<=1,'oracle numerical/alignment coverage differs')
        found.append(tuple(int(fields[k]) for k in ('cells','live_cells','cap','queries','masked')))
    expected=[(97,49,33,1,0),(97,71,33,3,1)]
    if mode=='oracle-component':
        expected += [(2113,2113,w,3,1) for w in (1,17,31,32,63,64,65,2051)]
        expected += [(97,71,65,3,0),(2113,2113,2051,3,0),(262144,262144,2051,1,0),(262144,13,33,1,0)]
    guards.require(found==expected,'independent oracle case grid incomplete')
    extended=3 if mode=='oracle-component' else 1
    for marker in ('case explicit_duplicate_ids=PASS','case alternating_rows=PASS unique_KV_sentinels=PASS'):
        guards.require(text.count(marker)==extended,'adversarial marker coverage differs')
    guards.require(text.count('case mixed_widths=PASS')==(2 if mode=='oracle-component' else 0),
        'mixed per-query widths missing')
    guards.require(text.count('fixture_status=PASS ')==1,'final oracle status missing')
    return dict(cases=found,extended_cases=extended,independent_oracle=True,all3arms_exact=True,
        guards=True,inputs_immutable=True,physical262144_component=mode=='oracle-component')


def parse_samples(rows,process):
    samples=[r for r in rows if r['kind']=='sample'];seen=set()
    for r in samples:
        key=(r['queries'],r['round'],r['arm']);guards.require(key not in seen,'duplicate timing sample');seen.add(key)
        guards.require(r['queries'] in (512,6491) and r['process']==process and
            0<=r['round']<6 and 0<=r['arm']<3 and r['repeats']==(32 if r['queries']==512 else 2) and
            r['batch_ns']>=100000000 and r['batch_ns']>=1000*r['timer_floor_ns']>0,
            'timing batch eligibility differs')
        guards.require((r['arm']==0 and r['event_calls']==0 and r['attention_device_ns'] is None) or
            (r['arm'] in (1,2) and r['event_calls']==r['repeats'] and
             0<r['attention_device_ns']<=r['batch_ns']+1000000),'direct event coverage differs')
    expected={(q,round_,arm) for q in (512,6491) for round_ in range(6) for arm in range(3)}
    guards.require(seen==expected,'incomplete108sample timing grid')
    for q in (512,6491):
        group=[r for r in samples if r['queries']==q]
        guards.require([r['arm'] for r in group]==[(process+round_+offset)%3 for round_ in range(6) for offset in range(3)],
            'rotated timing order differs')
    return samples


def performance(observations):
    samples=[s for o in observations for s in o['samples']];guards.require(len(samples)==108,'all108samples required')
    result={};advance=True
    for q in (512,6491):
        arms=[]
        for arm in range(3):
            group=[s for s in samples if s['queries']==q and s['arm']==arm];metrics={}
            for name,field in (('wall','batch_ns'),('attention_service','attention_device_ns')):
                if arm==0 and name=='attention_service':continue
                values=[s[field]/s['repeats']/1e6 for s in group]
                metrics[name]=dict(median_ms=statistics.median(values),sample_sd_ms=statistics.stdev(values),
                    process_medians_ms=[statistics.median(s[field]/s['repeats']/1e6 for s in group if s['process']==p) for p in range(3)])
            arms.append(dict(arm=arm,metrics=metrics))
        control=[arms[1]['metrics']['wall']['process_medians_ms'][p]/arms[0]['metrics']['wall']['process_medians_ms'][p] for p in range(3)]
        gain={name:[arms[2]['metrics'][name]['process_medians_ms'][p]/arms[1]['metrics'][name]['process_medians_ms'][p] for p in range(3)] for name in ('wall','attention_service')}
        eligible=all(0.99<=v<=1.01 for v in control) and all(v<=0.99 for values in gain.values() for v in values)
        advance=advance and eligible
        result[str(q)]=dict(arms=arms,control_over_original_wall=control,merged_over_control=gain,performance_gate_passed=eligible)
    return dict(shapes=result,samples=samples,advance_to_model=advance,hardware_saturation_qualified=False,
        limits='Direct-event component comparison only;model/decode/quality/full-context and clean model throughput require separate qualification.')


def run(mode):
    record=build.gen.committed(RESULT);plan=record['check_plan'];root=ROOT/('check-'+mode)
    guards.require(plan['mode']==mode and plan['stage']=='prepared' and plan==json.loads((root/'plan.json').read_text()) and
        not (root/'record.json').exists(),'matching fresh committed check;no retry')
    rt=admission(record,plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor=next(s.split(': ',1)[1] for s in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if s.startswith('-- cursor: '))
    result=dict(mode=mode,source_commit=plan['source_commit'],observations=[],passed=False,complete=False)
    owner=None
    def persist():build.write(root/'record.json',dict(result,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    try:
        owner=supervision.Owner(root/'owned',persist)
        for job in plan['jobs']:
            admission(record,plan);guards.driver_guard();guards.no_model_owners([])
            folder=root/str(job['index']);initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
            def interact(io,entry):
                prefix='QSA_BF16_READY ' if job['binary']=='oracle' else 'QSA_BARRIER_GATE '
                for stage in ('before-runtime','runtime-ready','after-cleanup'):
                    while True:
                        line=io.line(500)
                        if line.startswith(prefix):break
                    guards.require(line==prefix+stage,'fixture gate differs')
                    delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
                    guards.require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'suspend/clock discontinuity')
                    observe(io.proc.pid,entry,record,plan,job,rt,stage);io.send(b'X')
                final=io.line(20)
                guards.require(final.startswith('fixture_status=PASS ') if job['binary']=='oracle' else json.loads(final)==dict(kind='PASS'),
                    'final native result missing')
                io.proc.stdin.close();io.exit(20)
            session=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+job['session']) if job['session'] else None
            guards.require(session is None or not session.exists(),'preexisting collector session')
            print('Running '+mode+' process'+str(job['index']),flush=True)
            entry,out,err=owner.run(str(job['index']),job['command'],job['environment'],P,interaction=interact,**plan['budgets'])
            if session is not None and owner.active is None and entry.get('direct_child_reaped') and entry.get('session_empty') and session.exists():
                stop,_,_=owner.run('stop',[plan['collector_assets']['launcher']['path'],'--stop',job['session']],job['environment'],P,
                    wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=192<<20)
                guards.require(supervision.closed(stop) and stop['exit_code']==0,'collector stop failed')
            guards.require(supervision.closed(entry) and entry['exit_code']==0 and (session is None or not session.exists()),
                'fixture/collector did not close: '+str(entry.get('error')))
            stdout,stderr=out.read_text(),err.read_text();proof=validation.preload_evidence(stderr,entry['pid'])
            guards.require(proof['loader']==plan['preload']['loader']['path'],'loader mismatch')
            body=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body) and
                (not (folder/'validation.log').exists() or not (folder/'validation.log').stat().st_size),'native runtime/validation error')
            item=dict(process=job['index'],pid=entry['pid'],exit_code=0,ownership_closed=True,validation=proof,
                session_removed=session is None or not session.exists())
            if job['binary']=='oracle':item['oracle']=oracle_cases(stdout,mode,job['index'])
            else:
                rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                checked=[r for r in rows if r['kind']=='checked']
                expected=[512,6491] if mode=='perf' else [3]
                guards.require(rows[-1]==dict(kind='PASS') and [r['queries'] for r in checked]==expected and
                    all(r['cells']==39259 and r['guards'] and r['inputs_immutable'] and r['exact_original'] for r in checked),
                    'actual-mask fixture coverage incomplete')
                item['checks']=checked
                if mode=='perf':item['samples']=parse_samples(rows,job['index'])
                if mode=='resources':item['resources']=resources(stderr)
            result['observations'].append(item);persist()
        if mode=='perf':result['performance']=performance(result['observations'])
        admission(record,plan);result.update(passed=True,complete=True)
    except BaseException as error:result.update(passed=False,complete=False,error=repr(error))
    finally:
        result['ownership_closed']=bool(owner and owner.active is None and all(supervision.closed(c) for c in owner.commands))
        postchecks(result,boot,cursor);persist();plan['stage']='closed' if result['passed'] else 'failed'
        record.setdefault('checks',[]).append(result);record['GPU_executed']=True;build.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return int(not result['passed'])


def main():
    guards.require(len(sys.argv)==3 and sys.argv[1] in ('prepare','run') and sys.argv[2] in SEQUENCE,
        'prepare|run oracle-smoke|oracle-component|smoke|resources|perf')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare(sys.argv[2]) if sys.argv[1]=='prepare' else run(sys.argv[2])


if __name__=='__main__':raise SystemExit(main())
