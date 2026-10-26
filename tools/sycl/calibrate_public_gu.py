#!/usr/bin/env python3
"""Main-owned public GU calibration; repository LICENSE, no model execution."""
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

sys.dont_write_bytecode = True
import qualify_native16_control as q
from command_service import parse_profiles

r, P, B, RESULT = q.r, q.P, q.B, q.RESULT
ROOT = B / 'public-gu-dispatch-calibration-v1'
KEY = 'public_gu_dispatch_calibration'
require, digest, write = q.require, q.digest, q.write


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'runtime_files', 'native_build')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    q.build.scope(desired)[KEY] = compact(row)
    value = json.dumps(compact(row), indent=2)
    # Preserve useful individual samples without expanding each scalar into a line.
    for match in reversed(list(re.finditer(r'"samples": \[', value))):
        start = match.end()-1; table, size = json.JSONDecoder().raw_decode(value[start:])
        if not table: continue
        require(all(isinstance(v, list) for v in table), 'sample table type differs')
        indent = ' ' * (match.start()-value.rfind('\n', 0, match.start())-1)
        rendered = '[\n' + ',\n'.join(indent+'  '+json.dumps(v, separators=(',', ':')) for v in table)+'\n'+indent+']'
        value = value[:start]+rendered+value[start+size:]
    value = value.replace('\n', '\n      ')
    literal = json.dumps(KEY)+':'
    if KEY in q.build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous calibration key')
        first = raw.index(literal)+len(literal)
        while raw[first].isspace(): first += 1
        _, count = json.JSONDecoder().raw_decode(raw[first:])
        raw = raw[:first]+value+raw[first+count:]
    else:
        anchor = '"fused_command_profile": {'
        require(raw.count(anchor) == 1, 'ambiguous calibration container')
        raw = raw.replace(anchor, anchor+'\n      '+literal+' '+value+',', 1)
    require(json.loads(raw) == desired, 'scoped update changed other evidence')
    write(ROOT/'record.json', row); RESULT.write_text(raw)


def qualification():
    closed = q.build.scope(r.validation.committed(RESULT))[q.KEY]
    require(closed['stage'] == 'qualified' and closed['complete'] and closed['passed'] and
        closed['ownership_closed'] and not closed['active'] and len(closed['processes']) == 10 and
        closed['artifact_retirement']['post_reduction_admission_passed'], 'closed official compatibility required')
    native, _, proof = q.native()
    require(proof == closed['native_build'], 'qualified native build changed')
    original = json.loads((q.ROOT/'record.json').read_text())
    require(q.compact(original) == closed and
        digest(closed['runtime_manifest']['path']) == closed['runtime_manifest']['sha256'], 'compatibility inputs changed')
    return closed, native, original


def prepare():
    require(not ROOT.exists() and KEY not in q.build.scope(r.validation.committed(RESULT)), 'single-use calibration')
    closed, native, original = qualification()
    rt = r.validation.committed(r.RESULT); r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    cases = [dict(name='profile'+str(arm), arm=arm, profile=True, ordinal=0) for arm in (0, 1)]
    counts = {0: 0, 1: 0}
    for index, arm in enumerate((0, 1, 1, 0, 0, 1)):
        cases.append(dict(name='bench'+str(index), arm=arm, profile=False, ordinal=counts[arm])); counts[arm] += 1
    row = dict(stage='prepared', source_commit=revision, controller_inputs=q.gen.benchmark_pins(revision),
        native_build=closed['native_build'], compatibility_sha256=identity(closed),
        native_build_source_commit=native['source_commit'], fixture=native['fixtures']['sycl_moe_gu_swiglu_bench'],
        compiled_inputs=native['compiled_inputs'], runtime_files=original['runtime_files'],
        runtime_manifest=closed['runtime_manifest'], runtime_sha256=digest(r.RESULT),
        boot=rt['boot'], driver=rt['driver'], environment=original['environment'], cases=cases, processes=[],
        sample_columns=['rows','round','repeat','selected','fallback','wall_ns','queue_interval_ns','max_half_ulp'],
        active=False, ownership_closed=True, complete=False, passed=False, GPU_executed=False,
        model_executed=False, inference_adopted=False, hardware_saturation_qualified=False,
        question='Calibrate the actual GU public dispatcher/fallback at M160/2333/2334 with this official SDK;'
            'separate event GPU service from queue/wall intervals,without claiming original-engine regression causality.',
        plan='Two fresh traced profile processes(startup fusion0/large),then six clean component processes '
            '0/1/1/0/0/1. Fixed512calls at M160 and128calls at M2333/2334;one initial timing batch after setup '
            'and five later batches per shape. All batches require >=10ms wall/queue,endpoint controls<=1percent '
            'and timer resolution<=0.1percent of queue interval;actual timed dispatch counts must agree. '
            'Independent process pairs are(bench0,bench1),(bench3,bench2),(bench4,bench5). '
            'Report every sample;no speed-based stop,retry,selection or model advancement. Stop first '
            'source/provider/oracle/guard/owner/budget/fault failure. Validation/preload safety remains charged.',
        limits='Profile service comes from isolated checked-drain calls,not matched clean-batch GPU service. '
            'Queue intervals include host submission gaps. Batch wall includes submission and completion;'
            'thread CPU submission/wait split is not measured by this existing fixture. '
            'Setup/warmup precedes the first timing batch:it is not cold-process or model first-use latency. '
            'Official compiler/component timings do not attribute the historical same-original-binary decline '
            'or establish a whole-model/physical-bus/GDDR/arithmetic ceiling.',
        budgets=dict(wall=300, cpu=1200, rss_cap=4<<30, text_cap=8<<20, file_cap=32<<20, total_cap=64<<20),
        retention='Keep compact individual timing samples,profile service/dispatch/count/conservation/clock limits '
            'and source/runtime identities. Retire normal API/JIT/process logs after closure. Official4fixtures '
            'and SDK proof remain only until the resulting next component/model qualification decision.')
    ROOT.mkdir(mode=0o700); save(row)
    print('Prepared;commit the exact calibration plan before run.', flush=True)


def current(row):
    require(q.build.scope(r.validation.committed(RESULT))[KEY] == compact(row), 'commit exact calibration plan')
    require(q.gen.benchmark_pins(row['source_commit']) == row['controller_inputs'], 'controller source changed')
    closed, _, original = qualification()
    require(identity(closed) == row['compatibility_sha256'] and closed['native_build'] == row['native_build'] and
        original['runtime_files'] == row['runtime_files'] and digest(r.RESULT) == row['runtime_sha256'], 'qualification changed')
    require(all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'runtime provider changed')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'boot/driver changed')


def analyze(case, stdout, stderr):
    require('fixture_status=PASS mode=' in stdout, 'public fixture failed')
    records = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    require(records and all(v.get('status') == 'PASS' for v in records), 'public numerical/topology failure')
    if case['profile']:
        oracle = [v for v in records if v.get('kind') == 'profiled_engine_gu']
        profiles = parse_profiles(stderr, expected_windows=1)
        require(len(profiles) == len(oracle) == 4 and [v['rows'] for v in oracle] == [3,160,2333,2334], 'profile shapes differ')
        result = []
        for item, profile in zip(oracle, profiles):
            selected = bool(case['arm'] and item['rows'] >= 2334)
            wanted = ['gu_swiglu'] if selected else ['gu','swiglu']
            require(item['selected'] == selected and item['guards'] and item['inputs_immutable'] and
                item['max_half_ulp'] <= 1 and [v['kind'] for v in profile['samples']] == wanted and
                len(profile['samples']) == item['expected_records'], 'profile route/count/oracle differs')
            result.append(dict(rows=item['rows'], selected=selected,
                service_ns_by_kind={v['kind']:v['service_ns'] for v in profile['samples']},
                submit_clock_qualified=profile['summaries'][0]['submit_clock_qualified']=='1',
                submit_clock_inversions=int(profile['summaries'][0]['submit_clock_inversions'])))
        return dict(profiles=result, full_event_count_and_service_conservation=True)
    samples = [v for v in records if v.get('kind') == 'public_sample']
    controls = [v for v in records if v.get('kind') == 'public_control']
    require(len(samples) == 18 and len(controls) == 16 and
        [(v['rows'],v['round']) for v in samples] == [(m,t) for m in (160,2333,2334) for t in (-1,0,1,2,3,4)], 'timing sample count/order differs')
    require([(v['phase'],v['index']) for v in controls] == [(p,i) for p in (0,1) for i in range(8)], 'control order differs')
    for v in samples:
        repeat = 512 if v['rows'] == 160 else 128; selected = repeat if case['arm'] and v['rows'] >= 2334 else 0
        require(v['process'] == case['ordinal'] and v['arm'] == case['arm'] and v['repeat'] == repeat and
            v['selected'] == selected and v['fallback'] == repeat-selected and v['eligible'] and
            min(v['wall_ns'],v['queue_interval_ns']) >= 10000000 and v['max_half_ulp'] <= 1 and
            v['scope'] == 'public_dispatch_queue_interval_includes_host_gaps_not_GPU_service', 'timed eligibility/dispatch differs')
    require(max(v['wall_ns'] for v in controls) <= min(v['wall_ns'] for v in samples)*.01 and
        max(v['timer_resolution_ns'] for v in controls) <= min(v['queue_interval_ns'] for v in samples)*.001,
        'control/resolution eligibility differs')
    return dict(samples=[[v[k] for k in ('rows','round','repeat','selected','fallback','wall_ns','queue_interval_ns','max_half_ulp')] for v in samples],
        control_max_wall_ns=max(v['wall_ns'] for v in controls), controls=16,
        timer_resolution_ns=max(v['timer_resolution_ns'] for v in controls), dispatch_and_all_guards_passed=True)


def summarize(processes):
    by_name={v['name']:v for v in processes}; result=[]
    for rows in (160,2333,2334):
        wall={};queue={}
        for name,p in by_name.items():
            if p['profile']:continue
            warm=[v for v in p['details']['samples'] if v[0]==rows and v[1]>=0]
            require(len(warm)==5,'five subsequent batches required')
            wall[name]=statistics.median(v[5]/v[2] for v in warm)
            queue[name]=statistics.median(v[6]/v[2] for v in warm)
        pairs=[wall['bench'+str(b)]/wall['bench'+str(a)] for a,b in ((0,1),(3,2),(4,5))]
        service={}
        for arm in (0,1):
            p=next(v for v in by_name['profile'+str(arm)]['details']['profiles'] if v['rows']==rows)
            service[str(arm)]=sum(p['service_ns_by_kind'].values())
        result.append(dict(rows=rows,projection_n=1280,projection_k=2560,
            projection_FMA_ops=2*rows*1280*2560,whole_public_call_wall_ns_by_process=wall,
            queue_interval_ns_by_process=queue,isolated_profile_service_ns_by_arm=service,
            fusion_on_off_paired_wall_ratios=pairs,paired_ratio_geomean=statistics.geometric_mean(pairs)))
    return dict(rows=result,decision='Descriptive current official-SDK component calibration only;'
        'no engine adoption,original-binary regression cause or resource saturation established.',
        limits='Projection operation count excludes activation/rounding arithmetic. Isolated one-call service '
            'and warmed batches are different envelopes;do not subtract them to invent CPU time or a critical path.')


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT/'owned').exists(), 'single-use cohort')
    initial=copy.deepcopy(row); owner=None; outcome=dict(host_probe_accepted=False)
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(ROOT/'execution.json', dict(processes=row['processes'], active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner=r.supervision.Owner(ROOT/'owned',persist)
        for case in row['cases']:
            current(initial)
            root=ROOT/case['name'];root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
            profile=case['profile'];env=dict(row['environment'], TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                STRATA_SYCL_PREFILL_GU_SWIGLU='large' if case['arm'] else '0',
                STRATA_SYCL_PREFILL_COMMAND_PROFILE='1' if profile else '0',
                STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='1' if profile else '0',
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if profile else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
            args=['--profile-public',str(case['arm'])] if profile else ['--bench-public',str(case['ordinal']),str(case['arm'])]
            allowed=dict(row['runtime_files']);allowed[row['fixture']['path']]=row['fixture']['sha256'];seen=set();env_seen=False
            def monitor(io,entry):
                nonlocal env_seen
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try:
                        mapped=r.executable_mappings(Path('/proc',str(io.proc.pid),'maps').read_text())
                        actual=dict(v.split(b'=',1) for v in Path('/proc',str(io.proc.pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                        require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and
                            set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'actual environment differs')
                        env_seen=True
                    except (FileNotFoundError,ProcessLookupError):mapped=set()
                    except PermissionError:
                        require(io.exit_watch.poll(20),'live native /proc access denied');mapped=set()
                    require(mapped<=set(allowed),'unplanned code map: '+repr(mapped-set(allowed)));seen.update(mapped)
                    if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
                io.exit(1)
            entry,out,err=owner.run(case['name'],[row['fixture']['path'],*args],env,P,interaction=monitor,**row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code']==0,'public case failed: '+case['name'])
            activation=r.validation.preload_evidence(err.read_text(),entry['pid'])
            require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'] and env_seen and
                env['STRATA_UR_PROBE_EXPECTED_LOADER'] in seen and
                str(q.build.OFFICIAL/'lib/libsycl.so.9.0.0') in seen and
                {p for p in seen if '/libze_intel_gpu.so.' in p}=={env['ZE_ENABLE_ALT_DRIVERS']},'native provider evidence missing')
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'parameter warning')
            if profile:require('urEnqueue' in (root/'ur.trace').read_text(),'profile API trace missing')
            detail=analyze(case,out.read_text(),err.read_text())
            checked=dict(host_probe_accepted=True);r.validation.finish(checked,owner,cursor,row['boot'])
            require(checked['passed'] and checked['driver_after']==row['driver'],'fault/owner check failed')
            row['processes'].append(dict(case,passed=True,ownership_closed=True,parameter_validation_silent=True,details=detail));persist()
            print('Passed '+case['name'],flush=True)
        current(initial);row['analysis']=summarize(row['processes']);outcome['host_probe_accepted']=True
    except BaseException as error:
        outcome['error']=repr(error)
        if owner and owner.commands:outcome['last_case']=owner.commands[-1]['label']
    finally:
        if owner:r.validation.finish(outcome,owner,cursor,row['boot'])
        passed=bool(outcome.get('passed') and len(row['processes'])==8)
        row.update(stage='closed' if passed else 'failed',complete=passed,passed=passed,active=False,
            ownership_closed=outcome.get('ownership_closed',True),outcome=outcome,
            GPU_executed=bool(owner and owner.commands))
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
