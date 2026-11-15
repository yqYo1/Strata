#!/usr/bin/env python3
"""Owned bench.sh device-only request profile on the coherent fast engine.

Main-authored from this fork's benchmark controls; repository LICENSE.
Whole-request attribution only: no paused collection, phase-drain proof,
hardware counters, physical bandwidth or clean-speed claim from traced runs.
"""
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import current_pti_admission as admission
import request_profile_reader as reader
import compare_official_prefill as runner
import qsa16_stream_bench as core
import compare_owned_prefill as routes

P, B, RESULT = admission.CHECKOUT, admission.pti.B, admission.RESULT
FUSED = admission.FUSED
ROOT, KEY = B/'current-request-profile-v2', 'unitrace_current_request_profile_v2'
THIS, TEST = 'tools/sycl/current_request_profile.py', 'tools/test_current_request_profile.py'
FIXTURE = 'tests/fixtures/sycl-fast-prefill-route-32k.txt'
fast_request_route = routes.context().request_route
ARMS, RUNS = ('clean','trace','trace','clean'), 3
ORDER = [(32769,1)]*len(ARMS)
require, sha, signature = admission.require, admission.digest, runner.fingerprint
HOST_CHECK = dict(command=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3',
    '-B','-m','unittest','tools.test_current_request_profile','tools.test_request_profile_reader'],
    wall_seconds=90,CPU_seconds=60,RSS_bytes=1<<30,text_bytes=128<<10,total_bytes=4<<20,
    GPU_executed=False)


def compact(row):
    out={k:v for k,v in row.items() if k not in ('sources','environment_template','model_assets')}
    for name in ('sources','environment_template','model_assets'):
        out[name+'_sha256']=signature(row[name])
    return out


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(RESULT.read_text());data[KEY]=compact(row)
    RESULT.write_text(json.dumps(data,indent=2)+'\n')


def basis():
    admitted=admission.admit()
    cpu=admission.committed_json(RESULT,P)
    gate=cpu['unitrace_current_runtime_probe_v2']
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active']
            and [r['label'] for r in gate['results']]==['control','trace']
            and all(r['passed'] and r['native_exit_code']==0 and not r['fresh_faults'] for r in gate['results'])
            and gate['results'][1]['raw_clock_mapping_checked'], 'closed current fixture/clock gate required')
    require({k:v for k,v in admitted.items() if k!='controller_commit'}==
            {k:v for k,v in gate['admission'].items() if k!='controller_commit'},
            'same qualified current collector,fast native source and runtime required')
    seed=json.loads((B/'coherent-generations-prefill-v1/record.json').read_text())
    qualified=runner.shared.runtime.validation.committed(FUSED,('coherent_generations_prefill_v1',))
    require(core.compact(seed)==qualified and seed['stage']=='closed' and seed['complete']
            and seed['ownership_closed'] and not seed['active'], 'closed common production frontend/configuration required')
    return admitted,seed,signature(gate)


def source_pins(revision):
    pins=core.pins(revision)
    for name in (THIS,TEST,FIXTURE,'tools/test_request_profile_reader.py','AGENTS.md','docs/ARTIFACT_RETENTION.md'):
        require((P/name).read_bytes()==admission.blob(P,revision,name),'commit exact profile input: '+name)
        pins[name]=sha(P/name)
    require(THIS in pins,'actual controller must be tracked')
    return pins


def environment(row,plan):
    env=runner.environment(row,plan)
    env.update(STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
        STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR='1',STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_DECODE_KV_COALESCED='0',
        STRATA_SYCL_KV_COPY_GROUPS='0')
    require(not any('xestrata' in str(v).lower() for v in env.values()),'old provider paths refused')
    require(env.get('ZES_ENABLE_SYSMAN')=='0' and env.get('CUDA_MODULE_LOADING')=='EAGER'
            and env.get('UR_ENABLE_LAYERS')=='UR_LAYER_PARAMETER_VALIDATION'
            and not any(k.startswith('UNITRACE_') for k in env),'fixed quiet baseline environment required')
    return env


def configuration(row,plan):
    value=copy.deepcopy(row['configuration']);value['log']=str(Path(plan['root'])/'engine.stderr')
    require(value['exe']==str(P/'tools/sycl/engine_once.py'),'preserve production wrapper')
    return value


def profile(row,plan):
    tools=row['admission']['tools']['artifacts'];root=Path(plan['root'])
    return dict(mode='whole_request',launcher=tools['unitrace'],collector=tools['libunitrace_tool.so'],session='',
        environment=dict(UNITRACE_SystemTime='1',UNITRACE_FollowChildProcess='1',
            UNITRACE_TraceOutputDir=str(root/'trace')))


def native_context(row,plan):
    require(type(plan['index']) is int and 0<=plan['index']<len(ARMS) and plan['arm']==ARMS[plan['index']]
            and plan['read']==32769 and plan['heads']==1,'declared profile arm required')
    out={k:row[k] for k in ('binary','native_source_commit','native_validation_probe','boot')}
    if plan['arm']=='trace':out['native_profile']=profile(row,plan)
    return out


def launch_plan(row,plan):
    native=native_context(row,plan)
    return {k:v for k,v in native.items() if k in ('binary','native_validation_probe','native_profile')}


def check_native_environment(observed,expected,plan):
    actual={k.decode():v.decode() for k,v in observed.items()}
    wanted=dict(expected)
    if plan['arm']=='trace':
        prof=json.loads((Path(plan['root'])/'plan.json').read_text())['native_profile']
        shim=wanted['LD_PRELOAD']
        wanted.update(prof['environment'],STRATA_UNITRACE_TARGET_PRELOAD=shim,
            LD_PRELOAD=prof['collector']['path']+':'+shim,
            UNITRACE_DeviceTiming='1',UNITRACE_ChromeDeviceLogging='1',
            UNITRACE_ChromeEventBufferSize='128',UNITRACE_LD_PRELOAD_OLD=shim,
            ZE_ENABLE_TRACING_LAYER='1')
        version=actual.get('UNITRACE_VERSION','')
        require(0<len(version)<=200 and version.isascii(),'pinned launcher version field required')
        wanted['UNITRACE_VERSION']=version
    require(all(actual.get(k)==v for k,v in wanted.items())
            and set(actual)-set(wanted)<={'PWD','OLDPWD','SHLVL','_'},'actual native environment differs')


def native_extra_maps(plan):
    if plan['arm']=='clean':return set()
    return {json.loads((Path(plan['root'])/'plan.json').read_text())['native_profile']['collector']['path']}


def observe_gate(outcome,request,stage,stamp):
    require(type(stamp) is int and stamp>0,'positive RAW gate clock required')
    windows=outcome.setdefault('external_request_raw_windows',[])
    if stage=='begin':
        require(len(windows)==request and (not windows or 'end' in windows[-1]),'ordered request begin required')
        windows.append(dict(begin=stamp))
    else:
        require(stage=='end' and len(windows)==request+1 and 'end' not in windows[-1]
                and stamp>windows[-1]['begin'],'ordered request end required')
        windows[-1]['end']=stamp


def observe_lifetime(outcome,stage,stamp):
    bounds=outcome.setdefault('enclosing_owned_run_raw_ns',{})
    require(stage in ('begin','end') and stage not in bounds
            and (stage=='begin' or stamp>bounds['begin']),'ordered enclosing RAW bounds required')
    bounds[stage]=stamp


def request_route(text,plan):
    require(plan['read']==32769 and plan['heads']==1,'declared32Kroute required')
    return fast_request_route(text,plan)


def check_sample(sample,count,read,minimum_ms=10000):
    runner.check_sample(sample,count,read,minimum_ms)
    require(sample['generated_tokens']==24,'exact declared24output extent required')


def after_execution(outcome,row,plan):
    bounds=outcome['enclosing_owned_run_raw_ns'];windows=outcome['external_request_raw_windows']
    require(len(windows)==RUNS and all(bounds['begin']<=w['begin']<w['end']<=bounds['end'] for w in windows),
            'actual request windows must be inside the closed owned run')
    if plan['arm']=='clean':return
    root=Path(plan['root']);files=list((root/'trace').glob('*.json'))
    require(len(files)==1 and files[0].name.endswith('.'+str(outcome['native_identity']['pid'])+'.json'),
            'one actual target trace required')
    receipt=dict(schema=reader.SCHEMA,clock_domain=reader.CLOCK,pti_revision=reader.PTI_REVISION,
        unitrace_system_time='1',level_zero_only=True,native_exit_code=outcome['native_exit_code'],
        ownership_closed=outcome['ownership_closed'],fresh_faults=outcome['fresh_faults'],
        boot_unchanged=outcome['boot_unchanged'],source=dict(native_source_commit=row['native_source_commit'],
            binary_sha256=row['binary']['sha256']),native_lifetime=dict(begin_raw_ns=bounds['begin'],end_raw_ns=bounds['end']),
        requests=[dict(request_id=str(i),request_sha256=s['request_sha256'],actual_read_tokens=s['actual_read_tokens'],
            expected_read_tokens=32769,actual_reused_tokens=s['actual_reused_tokens'],generated_tokens=s['generated_tokens'],
            thinking=True,sample_scope='first_long' if i==0 else 'warmed',begin_raw_ns=w['begin'],end_raw_ns=w['end'])
            for i,(s,w) in enumerate(zip(outcome['samples'],windows))])
    summary=reader.summarize(reader.load_json(files[0]),receipt)
    summary['enclosure']='external owned benchmark lifetime;request gates,not native prefill drain'
    outcome['request_profile']=summary;outcome['trace_sha256']=sha(files[0])


def decision(processes):
    require(len(processes)==4 and all(p['passed'] and p['complete'] for p in processes),'four closed processes required')
    require(all(len(p['samples'])==RUNS and all(type(s.get(f)) in (int,float)
            and math.isfinite(s[f]) and s[f]>0 for s in p['samples']
            for f in ('prompt_ms','ttft_s','total_s')) for p in processes),
            'positive finite matched request samples required')
    points={}
    for scope in ('first_long','warmed'):
        def value(index,field):
            samples=processes[index]['samples']
            return samples[0][field] if scope=='first_long' else statistics.median(s[field] for s in samples[1:])
        fields={}
        for field in ('prompt_ms','ttft_s','total_s'):
            ratios=[value(t,field)/value(c,field) for c,t in ((0,1),(3,2))]
            fields[field]=dict(trace_over_clean=ratios,within_10_percent=all(.9<=r<=1.1 for r in ratios))
        points[scope]=fields
    return dict(points=points,observer_overhead_qualified=all(v['within_10_percent'] for fs in points.values() for v in fs.values()),
        phase_attribution_qualified=False,native_request_drain_qualified=False,physical_saturation_qualified=False,
        inference_adopted=False,clean_speed_change_claimed=False,
        next='Use service ranking only with declared observer limits;pin actual GEMM shapes before selecting one optimization.')


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens','32769','--think','--runs',str(RUNS),'--profile-request-gates',
        '--workload-id','current_request_profile_20261115','--fail-fast-engine-error','--ready-timeout','600',
        '--out',str(root/'bench')]


def current(row):
    require(admission.committed_json(RESULT,P)[KEY]==compact(row),'commit exact prior profile stage')
    require(all(sha(P/n)==h for n,h in row['sources'].items()),'frozen profile source changed')
    admitted,seed,gate=basis()
    require({k:v for k,v in admitted.items() if k!='controller_commit'}==
            {k:v for k,v in row['admission'].items() if k!='controller_commit'}
            and signature(core.compact(seed))==row['configuration_evidence_sha256']
            and gate==row['small_probe_evidence_sha256'],'frozen build/configuration/probe changed')
    contract,python=runner.entry.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected'],'qualified Python changed')
    for plan in row['plans']:
        root=Path(plan['root']);require(plan['command']==command(plan)
            and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==launch_plan(row,plan)
            and runner.fingerprint(environment(row,plan))==plan['environment_sha256']
            and all(sha(root/n)==h for n,h in plan['generated'].items()),'frozen generated inputs changed')
    require(runner.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],'whole campaign output budget exceeded')
    runner.shared.models.model_pins(row['configuration'],row['model_assets'],full=False)
    return json.loads(Path(row['admission']['runtime_manifest']['path']).read_text()),python


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh profile plan required')
    admitted,seed,gate=basis();revision=admission.pti.git(P,'rev-parse','HEAD').decode().strip()
    contract,python=runner.entry.python_manifest()
    row={k:copy.deepcopy(seed[k]) for k in ('configuration','environment_template','template','boot','driver','model_assets','native_validation_probe')}
    row['model_assets']=runner.shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
    row.update(stage='prepared',source_commit=revision,sources=source_pins(revision),admission=admitted,
        binary=admitted['binary'],native_source_commit=admitted['native_source_commit'],
        configuration_evidence_sha256=signature(core.compact(seed)),small_probe_evidence_sha256=gate,
        python_contract=contract,python_selected=python['selected'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,whole_bytes=1536<<20,
        budgets=dict(wall=1500,cpu=9000,rss_cap=104<<30,text_cap=16<<20,file_cap=256<<20,total_cap=384<<20),
        plan='Four independent processes clean,trace,trace,clean;three fresh thinking-on32769inputs and24outputs each. '
             'Commit each closure. Trace from native exec with current qualified device-only settings;no pause/session,host API or hardware counters. '
             'Separate first_long and median of subsequent2;two matched trace/clean ratios for prompt,clientTTFT,total request duration. '
             'All ratios must be0.9..1.1 for observer-overhead qualification;otherwise retain timing/ranking as perturbed observations only. '
             'Same fast14508ELF/configuration/placement and native validation in all arms;only device collector differs. '
             'Closed ownership,normal0,quiet validation,actual route/count/thinking/wire identity,current maps/source/providers,unchanged boot/driver,no fresh faults required. '
             'Stop first failure without retry or changed pins. Maximum256MiBtrace and1million events;384MiBraw per process and1536MiBcampaign '
             'are finite exceptions for three whole long requests. This is attribution,not speed adoption,phase-drain proof or saturation.',
        retention='No new binary generation. Keep individual clean/traced samples,top20kernel services plus conserved tail,union/crossing counts and identities/limits. '
                  'Retire full traces,logs,caches and private receipts after their verified result boundary.')
    ROOT.mkdir(mode=0o700)
    for i,arm in enumerate(ARMS):
        root=ROOT/str(i);root.mkdir();(root/'trace').mkdir()
        for name in ('tmp','cache'): (root/name).mkdir()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,arm=arm,read=32769,heads=1,root=str(root),port=port)
        (root/'config.json').write_text(json.dumps(configuration(row,plan),indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(launch_plan(row,plan),indent=2)+'\n')
        plan.update(command=command(plan),generated={n:sha(root/n) for n in ('config.json','plan.json')},
            environment_sha256=signature(environment(row,plan)))
        row['plans'].append(plan)
    save(row);print('Prepared current32Krequest profile;commit before host or native execution.',flush=True)


def host():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not row['host_passed'],'one host stage required')
    owner=runner.build.supervision.Owner(ROOT/'host');spec=HOST_CHECK
    entry,out,err=owner.run('unit',spec['command'],dict(PATH='/usr/bin:/bin',LANG='C',PYTHONDONTWRITEBYTECODE='1'),P,
        wall=spec['wall_seconds'],cpu=spec['CPU_seconds'],rss_cap=spec['RSS_bytes'],text_cap=spec['text_bytes'],file_cap=1<<20,total_cap=spec['total_bytes'])
    text=out.read_text()+err.read_text();passed=runner.build.supervision.closed(entry) and entry['exit_code']==0 and re.search(r'Ran 16 tests in [0-9.]+s',text) and text.rstrip().endswith('OK')
    row['host_evidence']=dict(source_commit=row['source_commit'],tests=16,passed=bool(passed),exit_code=entry['exit_code'],ownership_closed=runner.build.supervision.closed(entry),recipe=spec)
    row.update(host_passed=bool(passed),stage='ready' if passed else 'failed');save(row);require(passed,text[-8000:])


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','0','1','2','3'),'prepare|host|0..3')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.pti.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare':prepare()
        else:
            with (ROOT/(action+'-attempt.json')).open('x') as stream:json.dump(dict(action=action),stream)
            if action=='host':host()
            else:runner.run(int(action),sys.modules[__name__])
