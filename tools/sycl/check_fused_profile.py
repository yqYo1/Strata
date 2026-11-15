#!/usr/bin/env python3
"""Main-owned small traced checks of sparse/fused profiling; no model run.

Repository LICENSE. Inputs and the generated plan must be committed before
execution. Original selected engine/decode placement are never changed.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_fused_profile as n
import command_service as service

r, P, B, RESULT = n.r, n.P, n.B, n.RESULT
ROOT = n.ROOT / 'device-v1'
INPUTS = ['tools/sycl/check_fused_profile.py','tools/sycl/build_fused_profile.py',
          'tools/sycl/command_service.py','tools/sycl/runtime_probe.py',
          'tools/sycl/measurement_owner.py','tools/sycl/measurement_guards.py','tools/sycl/ur_validation.py']
require, digest, write = r.require, r.digest, r.write


def pins(revision):
    out = {}
    for name, raw in n.build.blobs(P,revision,INPUTS):
        require(raw == (P/name).read_bytes(),'commit exact check input: '+name)
        out[name] = hashlib.sha256(raw).hexdigest()
    return out


def native():
    container = r.validation.committed(RESULT)
    row = container['event_reconciliation']['fused_command_profile']['build']
    original = json.loads((n.ROOT/'record.json').read_text())
    require(row == n.compact(original) and row['stage'] == 'built-not-executed' and row['passed'] and
            row['ownership_closed'] and not row['active'] and not row['engine_built'], 'closed component build required')
    n.admit(original)
    manifest = row['compiled_inputs']['manifest']
    require(digest(manifest['path']) == manifest['sha256'],'compiler-input manifest differs')
    compiled = json.loads(Path(manifest['path']).read_text())
    require(len(compiled) == row['compiled_inputs']['files'] and
            all(digest(p) == h for p,h in compiled.items()), 'actual compiled input changed')
    for fixture in row['fixtures'].values():
        require(digest(fixture['path']) == fixture['sha256'] and Path(fixture['path']).stat().st_size == fixture['bytes'],
                'component executable changed')
    return row


def prepare():
    require(not ROOT.exists(),'fresh check only;inspect an existing attempt before retry')
    built = native()
    container = r.validation.committed(RESULT)
    scope = container['event_reconciliation']['fused_command_profile']
    require('device' not in scope,'existing check plan;no replacement')
    rt = r.validation.committed(r.RESULT)
    source = r.validation.git('rev-parse','HEAD').decode().strip()
    cases = []
    for name, argv, profile, error, code in (
        ('sparse-off',['--fixture'],0,None,0),('sparse-on',['--fixture'],1,None,0),
        ('sparse-unfused',['--unfused'],1,None,0),('bad-tag',['--bad-fused-tag'],1,'fused_tag_scope',0),
        ('capacity',['--event-capacity'],1,'event_capacity',0),
        ('injected',['--inject-before-drain'],1,'missing_checked_retirement',1)):
        cases.append(dict(name=name,target=n.TARGETS[0],argv=argv,profile=profile,expected_error=error,exit_code=code))
    for target,label in ((n.TARGETS[1],'gu'),(n.TARGETS[2],'down')):
        for profile in (0,1):
            cases.append(dict(name=label+'-'+str(profile),target=target,argv=['--profile-engine-hook'],
                              profile=profile,expected_error=None,exit_code=0))
    env = dict(rt['environment'],STRATA_SYCL_PREFILL_NATIVE16='bf16',
        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
        STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
        STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='1')
    plan = dict(stage='prepared',source_commit=source,inputs=pins(source),native_source_commit=built['source_commit'],
        fixtures=built['fixtures'],compiled_inputs=built['compiled_inputs'],runtime_sha256=digest(r.RESULT),
        boot=rt['boot'],driver=rt['driver'],environment=env,cases=cases,processes=[],active=False,ownership_closed=True,
        passed=False,model_qualified=False,clean_speed_qualified=False,observer_cost_qualified=False,
        criteria='Ten ordered fresh processes;stop first unexpected exit,oracle/guard/dispatch/profile/provider/source/owner/fault failure. '
            'No retry. Preserve intentional failed profiles. Sparse195/203 retained events cover639/655 enqueues;'
            'bad scope/capacity must fail profiling with valid outputs;injection must drain before releasing owners and query no unfinished event. '
            'Real GU2333/2334/8192 and Down those+irregular2334/16384 compare full outputs and inputs off/on;'
            '2 separate or1 fused event only. Pre-main parameter validation and budgeted full API tracing enabled;no complete-device-coverage,speed/adoption/saturation claim.',
        budgets=dict(wall=300,cpu=600,rss_cap=4 << 30,text_cap=16 << 20,file_cap=32 << 20,total_cap=64 << 20),
        whole_check_bytes=512 << 20,
        retention='Keep compact counts,dispatch/numerics/guard/provider/clock/error/drain/fault outcomes and source/recipe. '
            'Retire successful full traces,JIT and normal process logs after replay;keep one useful distinct unresolved failure only.')
    ROOT.mkdir(mode=0o700)
    scope['device'] = plan
    write(RESULT,container)
    print('Prepared ten bounded native cases;commit exact device plan before running.',flush=True)


def current(plan):
    require(pins(plan['source_commit']) == plan['inputs'] and digest(r.RESULT) == plan['runtime_sha256'],
            'check source/runtime changed')
    built = native()
    require(built['fixtures'] == plan['fixtures'] and built['compiled_inputs'] == plan['compiled_inputs'],
            'native build changed')


def analyze(case, out, err):
    profile = case['profile']
    error = case['expected_error']
    if error:
        footers = re.findall(r'^strata command profile: (.*)$',err,re.M)
        require(len(footers) == 1,'one intentional failed profile required')
        fields = service.fields('strata command profile: '+footers[0],'strata command profile: ')
        require(fields['version'] == '3' and fields['service_complete'] == '0' and fields['error'] == error,
                'wrong intentional profile refusal')
        expected_records = {'bad-tag':195,'capacity':224,'injected':0}[case['name']]
        require(fields['invalid'] == fields['invalid_saved'] == '0' and
                fields['windows'] == fields['expected'] == '1' and fields['cap'] == '224' and
                int(fields['records']) == expected_records and
                int(fields['peak_events']) == (224 if case['name']=='capacity' else 195) and
                0 <= int(fields['max_query_wall_ns']) < 1000000000,
                'intentional refusal hides an unexpected clock/population/cap error')
        if case['name'] == 'injected':
            require('injected-before-drain' in err and 'fixture fallback drain begin\nfixture fallback drain end' in err and
                    fields['records'] == '0' and fields['max_query_wall_ns'] == '0' and
                    'strata command sample:' not in err,'injected retirement queried unfinished events or missed checked drain')
        else:
            require('case status=PASS guards=PASS inputs=PASS' in out and 'fixture_status=PASS' in out,
                    'negative observer fixture lost valid output/guards')
        return dict(expected_profile_refusal=error,records=int(fields['records']),peak=int(fields['peak_events']),
                    checked_fallback_drain=case['name']=='injected',speed_qualified=False)
    require('fixture_status=PASS' in out,'native fixture marker missing')
    profiles = service.parse_profiles(err,expected_windows=1) if profile else []
    if not profile:
        require('strata command ' not in err,'disabled observer emitted a profile')
    if case['target'] == n.TARGETS[0]:
        unfused = case['name'] == 'sparse-unfused'
        count, kernels = (203,580) if unfused else (195,564)
        require('case status=PASS guards=PASS inputs=PASS expected_profile_records='+str(count) in out and
                'synthetic_kernels='+str(kernels)+' synthetic_copies=75' in out,'synthetic population/output differs')
        if profile:
            require(len(profiles)==1 and len(profiles[0]['samples'])==count and
                    sum(int(g['eligible']) for g in profiles[0]['groups']) == kernels+75,'synthetic exact population differs')
            require(sum(int(g['selected']) for g in profiles[0]['groups'] if g['kind']=='expert_upload')==4,
                    'ranked expert-transfer coverage differs')
        return dict(kernels=kernels,copies=75,profile_records=count if profile else 0,guards=True,inputs=True,
                    gpu_service_qualified=bool(profile),speed_qualified=False)
    kind = 'gu' if case['target']==n.TARGETS[1] else 'down'
    cases = [json.loads(line) for line in out.splitlines() if line.startswith('{')]
    cases = [v for v in cases if v.get('kind')=='profiled_engine_'+kind]
    rows = [2333,2334,8192]+([2334] if kind=='down' else [])
    require([v['rows'] for v in cases]==rows and all(v['status']=='PASS' and v['guards'] and v['inputs_immutable']
            and bool(v['selected'])==(v['rows']>=2334) for v in cases),'real engine hook output/dispatch differs')
    if profile:
        require(len(profiles)==len(cases),'real hook profile count differs')
        for value, data in zip(cases,profiles):
            expected = [kind+'_swiglu' if kind=='gu' else 'down_scatter'] if value['selected'] else \
                       [kind,'swiglu' if kind=='gu' else 'accumulate']
            require([s['kind'] for s in data['samples']]==expected and
                    len(data['samples'])==value['expected_records'],'real hook missing/duplicate/mislabeled event')
    return dict(operator=kind,oracle_cases=cases,profile_records=[len(v['samples']) for v in profiles],
                gpu_service_qualified=bool(profile),speed_qualified=False)


def run():
    container = r.validation.committed(RESULT)
    plan = container['event_reconciliation']['fused_command_profile']['device']
    require(plan['stage'] in ('prepared','collecting') and len(plan['processes'])<len(plan['cases']) and
            all(v['passed'] for v in plan['processes']),'ordered successful cases only;no duplicate/retry')
    current(plan)
    case = plan['cases'][len(plan['processes'])]
    root = ROOT/case['name'];require(not root.exists(),'existing attempt;inspect owner,do not retry')
    root.mkdir()
    for name in ('tmp','cache'):(root/name).mkdir()
    trace, warnings = root/'ur.log',root/'validation.log'
    env = dict(plan['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
               STRATA_SYCL_PREFILL_COMMAND_PROFILE=str(case['profile']),
               UR_LOG_TRACING='level:info;flush:info;output:file,'+str(trace),
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(warnings))
    rt = r.validation.committed(r.RESULT)
    allowed = set(rt['runtime_files']) | {str(Path(plan['fixtures'][case['target']]['path']).resolve())}
    cursor = next(line.split(': ',1)[1] for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    outcome = dict(name=case['name'],passed=False,host_probe_accepted=False)
    owner = None;seen = set()
    def persist():
        write(root/'record.json',dict(outcome,active=bool(owner and owner.active is not None),
                                     commands=owner.commands if owner else []))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            try:
                maps = Path('/proc',str(io.proc.pid),'maps').read_text()
            except (FileNotFoundError,ProcessLookupError):
                maps = ''
            mapped = r.executable_mappings(maps)
            require(mapped <= allowed,'unplanned mapped native provider: '+str(mapped-allowed))
            seen.update(mapped)
            if time.monotonic() >= next_size:
                require(r.supervision.artifact_bytes(root)<=plan['budgets']['total_cap'] and
                        r.supervision.artifact_bytes(ROOT)<=plan['whole_check_bytes'],'native check size budget')
                next_size = time.monotonic()+.5
            if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner = r.supervision.Owner(root/'owned',persist)
        entry,out,err = owner.run('native',[plan['fixtures'][case['target']]['path'],*case['argv']],env,P,
                                  interaction=monitor,**plan['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code']==case['exit_code'],'unexpected native exit/ownership')
        text = err.read_text(errors='strict')
        outcome['validation'] = r.validation.preload_evidence(text,entry['pid'])
        require(not warnings.exists() or warnings.stat().st_size==0,'parameter validation warning')
        require(any('libze_intel_gpu.so.1.17.39758' in p for p in seen) and
                str(Path(env['UR_ADAPTERS_FORCE_LOAD']).resolve()) in seen and
                str(Path(env['LD_PRELOAD']).resolve()) in seen,'actual native providers not observed')
        require(trace.is_file() and trace.stat().st_size>0 and 'urEnqueue' in trace.read_text(), 'enabled API trace/enqueue evidence missing')
        outcome['analysis'] = analyze(case,out.read_text(),text)
        current(plan)
        outcome.update(host_probe_accepted=True,complete=True,expected_exit_code=case['exit_code'],
                       actual_provider_mappings=True,parameter_validation_warnings=0,
                       trace_bytes=trace.stat().st_size,trace_sha256=digest(trace))
    except BaseException as error:
        outcome.update(error=repr(error),complete=False)
    finally:
        if owner is not None:r.validation.finish(outcome,owner,cursor,plan['boot'])
        persist();plan['processes'].append(outcome);plan['active']=False
        plan['ownership_closed']=bool(outcome.get('ownership_closed'))
        plan['stage']='collecting' if outcome['passed'] and len(plan['processes'])<len(plan['cases']) else 'closed'
        plan['passed']=bool(outcome['passed'] and len(plan['processes'])==len(plan['cases']))
        write(RESULT,container)
    print(json.dumps(outcome),flush=True)
    return int(not outcome['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run']),'check_fused_profile.py prepare|run')
    with (B/'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.guards.no_model_owners([])
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
