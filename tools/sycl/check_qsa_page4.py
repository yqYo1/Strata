#!/usr/bin/env python3
"""Bounded, first-use QSA Page4-address validation; does not launch the model engine."""
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import build_qsa_page4 as build
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as qualified
import ur_validation as validation
import model_aggregate as aggregate
import profile_aggregate as tables
from measure_down_scatter import postchecks

P,B,ROOT,RESULT=build.P,build.B,build.ROOT,build.RESULT


def inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        raw=build.gen.git('show',revision+':'+str(path.relative_to(P)))
        guards.require(raw==path.read_bytes(),'commit exact check input: '+str(path))
        pins[str(path.relative_to(P))]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return pins


def runtime(record):
    # Qualify this fixture's actual mode without posing as a model/query run.
    identity=lambda delta:{n:{k:v[k] for k in ('qualified_sha256','candidate_sha256')} for n,v in delta.items()}
    expected={n:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for n,v in qualified.DELTA.items()}
    component=build.gen.committed(qualified.COMPONENT)
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and component['fresh_faults']==[] and
        identity(record['runtime_asset_delta'])==identity(component['runtime_asset_delta'])==expected,
        'Page4-address runtime identities differ from the qualified native component')
    settings=build.gen.committed(P/'tools/sycl/measurement_environment.json')
    guards.require(set(expected)<=set(settings['runtime_elfs']),'unknown qualified runtime asset')
    for path,item in settings['runtime_elfs'].items():
        proposed=expected.get(path)
        if proposed:guards.require(proposed['qualified_sha256']==item['sha256'],'original runtime baseline changed')
        guards.require(build.digest(path)==(proposed['candidate_sha256'] if proposed else item['sha256']),
            'Page4-address runtime asset changed: '+path)
    guards.require(validation.preload_qualification()==record['validation_preload'],'validation provider changed')


def admission(record,plan):
    build.admission(record,True)
    guards.require(inputs(plan['source_commit'])==plan['controller_inputs'],'check controller changed')
    runtime(plan)
    if plan['mode']=='resources':
        assets=aggregate.profile_assets(build.gen.committed(aggregate.RESULT))
        guards.require(plan['collector_assets']=={k:assets[k] for k in ('collector','launcher','preload')},
            'qualified collector/launcher/preload changed')
    guards.require(record['stage']=='built' and record['passed'] and record['complete']
        and record['ownership_closed'] and not record['active'],'successful closed fresh v2 build required')
    for name,item in record['device_IR'].items():
        guards.require(build.digest(item['path'])==item['sha256']
            and build.division_evidence(item['path'])==item['division_analysis'],'compiled semantic IR proof changed')


def prepare(mode):
    record=build.gen.committed(RESULT)
    guards.require('check_plan' not in record or record['check_plan']['stage']=='closed','previous check not closed')
    checks=record.get('checks',[])
    sequence=('smoke','component','resources','perf');position=sequence.index(mode)
    guards.require(len(checks)==position and all(c['mode']==sequence[i] and c['passed'] and
        c['complete'] and c['ownership_closed'] and c['fresh_faults']==[] for i,c in enumerate(checks)),
        'require smoke,component,resources,perf in order; no retries')
    root=ROOT/('check-'+mode);guards.require(not root.exists(),'inspect existing check owner; no replacement')
    _,_,setup=build.resource.qualification()
    revision=build.gen.git('rev-parse','HEAD').decode().strip()
    plan=dict(stage='prepared',mode=mode,source_commit=revision,controller_inputs=inputs(revision),
        runtime_asset_delta=setup['runtime_asset_delta'],validation_preload=validation.preload_qualification(),
        gates='Smoke3arms*pages4,7*2cases;component3arms*pages1,4,7,16*11cases. Require exact ordered bits,independent tensororacle,INT_MIN/MAX/negative/out-of-range/duplicate/pageboundary IDs,guards,immutability,refusals,Page4physical262144lastslot/tail,andactualfast/genericrouteproof. SamePIDvalidatedfirstUR,normalclosureandnoFault. Notmodelcapacity/decodequalification.',
        budgets=dict(wall_s=180,cpu_s=180,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=32<<20,total_bytes=192<<20),
        jobs=[])
    if mode=='resources':
        assets=aggregate.profile_assets(build.gen.committed(aggregate.RESULT))
        plan['collector_assets']={k:assets[k] for k in ('collector','launcher','preload')}
        plan['gates']='Page4 timing fixture verifies actualfast route on every invocation;3arms/22calls(default8,control/Page4 each7). Require positive conserved table andexact3compute-family properties(SIMD16,WG128);controlargs18,candidate17or18afterconstantshaperemoval. RecordactualdefaultGRF/private/spill/rawSLM,notoccupancy/hottraffic. SamePIDvalidation/exactbits/guards/immutability/normalclosure/sessionremoved/noFault;notcleanspeed/adoption.'
    elif mode=='perf':
        plan['gates']=record['plan']['performance']
    admission(record,plan);root.mkdir(mode=0o700)
    base=build.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    arms=range(3) if mode=='perf' else ('all',) if mode=='resources' else tuple(
        name+'@'+str(page) for page in ((4,7) if mode=='smoke' else (1,4,7,16))
        for name in ('default','control','page4'))
    for arm in arms:
        arm=str(arm)
        folder=root/arm;folder.mkdir()
        for name in ('tmp','cache'):(folder/name).mkdir()
        env=dict(base,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
            LD_PRELOAD=plan['validation_preload']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=plan['validation_preload']['loader']['path'])
        if mode=='component':
            env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
        command=[record['binaries']['numerical']['path'],'--'+mode,*arm.split('@')];session=None
        if mode=='resources':
            session='StrataQsaPage4'+secrets.token_hex(12)
            (folder/'trace').mkdir()
            env.pop('LD_PRELOAD')
            env.update(STRATA_UNITRACE_TARGET_PRELOAD=plan['validation_preload']['library']['path'],
                UNITRACE_FollowChildProcess='1',UNITRACE_SystemTime='1',UNITRACE_TraceOutputDir=str(folder/'trace'))
            command=[plan['collector_assets']['launcher']['path'],'--session',session,'--device-timing','--verbose',
                record['binaries']['timing']['path'],'--smoke','0']
        elif mode=='perf':
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ENABLE_LAYERS','STRATA_TRACE'):
                    del env[key]
            command=[record['binaries']['timing']['path'],'--component',arm]
        plan['jobs'].append(dict(arm=arm,environment=env,command=command,session=session))
    record['check_plan']=plan
    build.write(root/'plan.json',plan);build.write(RESULT,record)
    print('Prepared '+mode+'; commit exact controller and plan before execution.',flush=True)


def timing_evidence(text,process,small):
    lines=text.splitlines();samples=[]
    floors=[int(x.split('=',1)[1]) for x in lines if x.startswith('timer_floor_ns=')]
    guards.require(len(floors)==1 and floors[0]>0,'missing timer calibration')
    shapes=[(97,3,33,'gather')] if small else [(32768,8192,2051,'gather'),(117488,8192,2051,'gather'),(117488,8192,2051,'clustered')]
    expected={(cells,queries,cap,pattern,round_,order) for cells,queries,cap,pattern in shapes for round_ in range(6) for order in range(3)}
    seen=set()
    for line in lines:
        if not line.startswith('sample '):continue
        row=dict(token.split('=',1) for token in line.split()[1:])
        key=tuple(int(row[k]) for k in ('cells','queries','cap'))+(row['pattern'],int(row['round']),int(row['order']))
        arm=int(row['variant']);ms=float(row['milliseconds'])
        guards.require(key in expected and key not in seen and int(row['process'])==process and
            arm==(process+key[4]+key[5])%3 and row['implementation']==('default','control','page4')[arm] and
            row['page_size']=='4' and row['route']=='PASS' and
            all(row[k]=='PASS' for k in ('exact_bits','guards','status')) and int(row['compared'])==key[1]*24*256 and
            math.isfinite(ms) and ms>0 and (small or ms>=.1 and ms*1e6>=1000*floors[0]),'ineligible/duplicate timing sample')
        seen.add(key);samples.append([process,*key,arm,ms])
    guards.require(seen==expected and sum(x.startswith('case cells=') and
        'samples=18 full_bits=PASS guards=PASS inputs=PASS status=PASS' in x for x in lines)==len(shapes)
        and sum(x.startswith('fixture_status=PASS ') for x in lines)==1,'incomplete timing fixture')
    routes=[dict(token.split('=',1) for token in line.split()[1:])
        for line in lines if line.startswith('page4_timing_routes ')]
    guards.require(len(routes)==1 and int(routes[0]['fast'])==7*len(shapes)
        and int(routes[0]['generic'])==0,'timing route total differs from actual timed/warm invocations')
    return dict(clock_floor_ns=floors[0],samples=samples)


def numerical_evidence(lines,arm,small):
    variant,page_text=arm.split('@');page=int(page_text)
    expected=[(97,33,1,0,49),(97,33,3,1,71)]
    if not small:
        expected += [(2113,width,3,1,2113) for width in (1,17,31,32,63,65,2051)]
        expected += [(262144,2051,1,0,262144),(262144,33,1,0,13)]
    cases=[dict(token.split('=',1) for token in line.split()[1:])
        for line in lines if line.startswith('case cells=')]
    exact=[dict(token.split('=',1) for token in line.split()[1:])
        for line in lines if line.startswith('case exact_prefix33_bits=')]
    keys=('cells','cap','queries','masked','live_cells')
    guards.require([tuple(int(row[k]) for k in keys) for row in cases]==expected
        and len(exact)==len(expected),'missing/duplicate/reordered numerical case identity')
    for row,bits,shape in zip(cases,exact,expected):
        guards.require(int(row['page_size'])==page and int(row['compared'])==shape[2]*24*256
            and int(row['active_rows'])==((shape[4]+page-1)//page)*2*page
            and all(row[k]=='PASS' for k in ('packed_logical_prefix','packed_guards','guards','inputs','status'))
            and bits['exact_prefix33_bits']=='PASS'
            and tuple(int(bits[k]) for k in ('cells','cap','queries'))==shape[:3],
            'numerical output extent, packed rows, guards or exact-bit identity mismatch')
    guards.require('fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED' in lines
        and 'page4_control variant='+variant+' arithmetic=unchanged' in lines
        and lines.count('case explicit_duplicate_ids=PASS aliased_physical_pages=PASS first_later_KV_diagnostics=PASS status=PASS')==1,
        'missing fixture/control/extended-diagnostic footer')
    routes=[dict(token.split('=',1) for token in line.split()[1:]) for line in lines if line.startswith('page4_routes ')]
    guards.require(len(routes)==1 and routes[0]['status']=='PASS' and int(routes[0]['page'])==page,
        'missing actual numerical route proof')
    fast,generic=int(routes[0]['fast']),int(routes[0]['generic'])
    calls=34 if small else 52 # 34 for the two initial cases, then 2 per added case.
    expected_routes=((calls-1,1) if page==4 else (0,calls)) if variant=='page4' else (0,0)
    guards.require((fast,generic)==expected_routes,'actual numeric route counts do not match selected arm/cases')
    return dict(case_count=len(cases),case_identities=expected,exact_prefix33_bits=True,page_size=page,
        physical262144_component_addressed=any(s[0]==s[4]==262144 for s in expected),route=routes[0])


def resource_evidence(text):
    table=tables.device_summary(text);properties={};counts={}
    template='prefix_batch_impl<33, false, false, true, true, true, true, false>'
    def selected(name):return template in name and '[SIMD16 ' in name and '{128; 1; 1}]' in name
    def arm(name):return 2 if '::page4::' in name else 1 if '::control::' in name else 0
    for row in table['rows']:
        if selected(row['name']):
            key=arm(row['name']);guards.require(key not in counts,'ambiguous compute family')
            counts[key]=row['calls']
    for line in text.split('=== Kernel Properties ===')[1].splitlines():
        parts=[x.strip() for x in line.rsplit(',',7)]
        if len(parts)!=8 or not selected(parts[0]):continue
        key=arm(parts[0]);guards.require(key not in properties,'duplicate resource descriptor')
        compiled=parts[1];simd,nargs,slm,private,spill,grf=map(int,parts[2:])
        guards.require(compiled=='JIT' and simd==16 and (nargs in (17,18) if key==2 else nargs==18) and grf in (128,256)
            and min(slm,private,spill)>=0,'unexpected compiled QSA resource properties')
        properties[key]=dict(arm=key,compiled=compiled,SIMD=simd,arguments=nargs,raw_SLM=slm,
            private_memory_descriptor=private,spill_memory_descriptor=spill,GRF=grf)
    guards.require(counts=={0:8,1:7,2:7} and set(properties)=={0,1,2},'wrong QSA arm coverage/resources')
    return dict(conserved_device_ns=table['total_device_ns'],table_rows=len(table['rows']),
        compute_calls=counts,properties=list(properties.values()),
        limits='Allocation descriptors and summed service only;rawSLM units,actual spill traffic,occupancy,physical traffic and model/decode speed remain unmeasured.')


def performance(checks):
    samples=[sample for check in checks for sample in check['timing']['samples']]
    guards.require(len(samples)==162,'full162 timing samples required')
    shapes=[]
    for cells,pattern in ((32768,'gather'),(117488,'gather'),(117488,'clustered')):
        arms=[]
        for arm in range(3):
            subset=[s for s in samples if s[1]==cells and s[4]==pattern and s[7]==arm]
            values=[s[8] for s in subset];guards.require(len(values)==18,'missing per-arm shape samples')
            arms.append(dict(arm=arm,median_ms=statistics.median(values),sample_sd_ms=statistics.stdev(values),
                process_medians_ms=[statistics.median(s[8] for s in subset if s[0]==p) for p in range(3)]))
        shapes.append(dict(cells=cells,pattern=pattern,arms=arms,ratios=[arms[2]['median_ms']/arms[a]['median_ms'] for a in (0,1)]))
    ratios=[statistics.geometric_mean(s['ratios'][a] for s in shapes) for a in (0,1)]
    passed=all(r<=.95 for r in ratios) and all(r<=1.05 for s in shapes for r in s['ratios']) and all(
        s['arms'][2]['process_medians_ms'][p]<s['arms'][a]['process_medians_ms'][p]
        for s in shapes if s['cells']==117488 for a in (0,1) for p in range(3))
    return dict(shapes=shapes,candidate_latency_geomeans_vs_default_and_control=ratios,
        advance_to_model=passed,decode_preservation_qualified=False,model_adoption_qualified=False,
        sample_columns=['process','cells','queries','cap','pattern','round','order','arm','milliseconds'],samples=samples)


def run(mode):
    record=build.gen.committed(RESULT);plan=record['check_plan'];root=ROOT/('check-'+mode)
    guards.require(plan['mode']==mode and plan['stage']=='prepared' and
        plan==json.loads((root/'plan.json').read_text()) and not (root/'record.json').exists(),
        'matching fresh committed check required; inspect existing receipt')
    admission(record,plan)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    result=dict(mode=mode,source_commit=plan['source_commit'],numerical_binary=record['binaries']['numerical'],
        passed=False,complete=False,active=False,ownership_closed=False,observations=[])
    owner=None
    def persist():
        result['active']=bool(owner and owner.active is not None)
        build.write(root/'record.json',dict(result,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(root/'owned',persist);persist()
    try:
        for job in plan['jobs']:
            # No controller/environment changes, even between independent arms.
            guards.require(inputs(plan['source_commit'])==plan['controller_inputs'],'controller changed during comparison')
            runtime(plan);guards.driver_guard();guards.no_model_owners([])
            for artifact in record['binaries'].values():guards.require(build.digest(artifact['path'])==artifact['sha256'],'fixture changed')
            print('Validating '+mode+' Page4-address '+job['arm'],flush=True)
            b=plan['budgets']
            entry=None;session=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+job['session']) if job['session'] else None
            guards.require(session is None or not session.exists(),'unique session already exists;inspect owner')
            try:
                entry,out,err=owner.run(job['arm'],job['command'],job['environment'],P,
                    wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
                    file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            finally:
                if session is not None and entry is not None and supervision.closed(entry) and owner.active is None:
                    if session.exists():
                        stop,_,_=owner.run('stop', [plan['collector_assets']['launcher']['path'],'--stop',job['session']],
                            job['environment'],P,wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=192<<20)
                        guards.require(supervision.closed(stop) and stop['exit_code']==0,'session stop failed')
                    guards.require(not session.exists(),'session was not removed')
            stdout,stderr=out.read_text(),err.read_text();lines=stdout.splitlines()
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'numerical fixture did not close normally')
            proof=None
            if mode!='perf':
                proof=validation.preload_evidence(stderr,entry['pid'])
                guards.require(proof['loader']==plan['validation_preload']['loader']['path'],'unexpected loader')
            # The pinned preload deliberately tests a null UR platform before
            # main (expected46); reject errors after that verified diagnostic.
            body_stderr=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1] if proof else stderr
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body_stderr),
                'runtime/validation diagnostic failed')
            if mode in ('resources','perf'):
                item=dict(arm=job['arm'],pid=entry['pid'],exit_code=entry['exit_code'],ownership_closed=True,
                    validation=proof,session_removed=session is None or not session.exists(),
                    timing=timing_evidence(stdout,0 if mode=='resources' else int(job['arm']),mode=='resources'),
                    stdout_sha256=build.digest(out),stderr_sha256=build.digest(err))
                if mode=='resources':
                    guards.require(not list((root/job['arm']/'trace').rglob('*.json')),'unexpected timeline')
                    item['resources']=resource_evidence(stderr)
                result['observations'].append(item);persist();continue
            evidence=numerical_evidence(lines,job['arm'],mode=='smoke')
            result['observations'].append(dict(arm=job['arm'],pid=entry['pid'],exit_code=entry['exit_code'],
                ownership_closed=True,validation=proof,**evidence,
                stdout_sha256=build.digest(out),stderr_sha256=build.digest(err)))
            persist()
        if mode=='perf':result['performance']=performance(result['observations'])
        result.update(passed=True,complete=True)
    except BaseException as error:result['error']=repr(error)
    finally:
        result['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result,boot,cursor);persist()
        plan['stage']='closed' if result['passed'] else 'failed'
        record.setdefault('checks',[]).append(result);record['GPU_executed']=True
        record['model_adoption_qualified']=False;build.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==3 and sys.argv[1] in ('prepare','run') and sys.argv[2] in ('smoke','component','resources','perf'),
        'use prepare|run smoke|component|resources|perf')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare(sys.argv[2]) if sys.argv[1]=='prepare' else run(sys.argv[2]))
