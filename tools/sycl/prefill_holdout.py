#!/usr/bin/env python3
"""One predeclared unseen-length check of the selected engine's time curve.

Repository LICENSE. Official bench.sh, unchanged selected native package; this
is neither a candidate adoption screen nor a new regression investigation.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_bench as client
import runtime_probe as runtime
import benchmark_runtime as python_runtime
import cpu_capacity as models
import generation_engine as gen
import engine_once
P, B, A = runtime.P, runtime.B, gen.A
RESULT = A/'fused-moe-service-current.json'
TEMPLATE = A/'phase-prefill-speed-current.json'
REFERENCE = A/'prefill-runtime-comparison-current.json'
ROOT = B/'selected-prefill-length-holdout-v1'
KEY = 'selected_length_holdout'
MINIMUM, RUNS = 54613, 4
require, digest, write = runtime.require, runtime.digest, runtime.write


def pins(revision):
    result = gen.benchmark_pins(revision)
    name = 'data/expert-profile.bin'
    raw = gen.git('show', revision+':'+name)
    require(raw == (P/name).read_bytes(), 'committed profile changed')
    result[name] = hashlib.sha256(raw).hexdigest()
    return result


def environment(root, rt):
    template = runtime.validation.committed(TEMPLATE)['environment_template']
    old = template['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old, str(root)) for k,v in template.items()}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPYCACHEPREFIX=str(root/'cache/python'),
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    env.pop('STRATA_TRACE', None)
    require(not env.get('LD_PRELOAD') and all(v == '0' for k,v in env.items()
            if '_PROFILE' in k or '_AUDIT' in k), 'clean observer settings required')
    python_runtime.require_environment(env)
    return env


def native():
    rt = runtime.validation.committed(runtime.RESULT)
    full = runtime.current(rt)
    runtime.guards.no_model_owners([])
    return full, rt


def compact(row):
    return {k:v for k,v in row.items() if k not in ('sources','model_assets','commands_executed')}


def prepare_python():
    record = runtime.validation.committed(RESULT)
    require(KEY not in record and not ROOT.exists(), 'fresh length characterization only')
    _, rt = native()
    revision = gen.git('rev-parse','HEAD').decode().strip(); source = pins(revision)
    previous = runtime.validation.committed(REFERENCE,
        ('historical_regression_audit','benchmark_python_runtime_v4'))
    manifest = python_runtime.read_manifest(previous['plan']['manifest'])
    python_runtime.validate_manifest(manifest)
    folder = ROOT/'python';folder.mkdir(mode=0o700, parents=True)
    for part in ('tmp','cache'):(folder/part).mkdir()
    env = environment(folder, rt)
    require(python_runtime.selection(env) == manifest['selected'], 'Python provider changed')
    config = runtime.validation.committed(TEMPLATE)['configuration']
    template = Path(config['tokenizer'])/'chat_template.jinja'
    row = dict(stage='prepared', source_commit=revision, sources=source, selected=manifest['selected'],
        manifest=previous['plan']['manifest'], template=dict(path=str(template),sha256=digest(template)),
        environment=env, commands=[[role,[manifest['selected']['path'],str(python_runtime.SOURCE),
            'probe',str(P/entry),str(template)]] for role,entry in
            [('benchmark','tools/strata_bench.py'),('server','serve/server.py')]],
        budgets=dict(wall=45,cpu=30,rss_cap=512<<20,text_cap=512<<10,file_cap=2<<20,total_cap=8<<20),
        passed=False, active=False, ownership_closed=True, GPU_executed=False,
        criterion='Existing external Python inventory only;two fresh current-source definition probes and mock loopback server. Check all project imports and external module/mapping identities,normal owned closure. No native engine or GPU inference.')
    write(folder/'record.json',row);record[KEY] = dict(python=compact(row));write(RESULT,record)
    print('Current-source host probes prepared;commit before run.',flush=True)


def host():
    record = runtime.validation.committed(RESULT)
    row = json.loads((ROOT/'python/record.json').read_text())
    require(compact(row) == record[KEY]['python'] and row['stage'] == 'prepared', 'fresh committed host plan')
    require(pins(row['source_commit']) == row['sources'], 'host source changed')
    manifest = python_runtime.read_manifest(row['manifest']);python_runtime.validate_manifest(manifest)
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    owner = None
    try:
        owner = runtime.supervision.Owner(ROOT/'python/owned')
        for role, argv in row['commands']:
            require(python_runtime.selection(row['environment']) == row['selected'], 'Python selection changed')
            entry,out,err = owner.run(role,argv,row['environment'],P,**row['budgets'])
            require(runtime.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                    'current-source host probe failed')
            meta = json.loads(out.read_text())
            require(str(Path(meta['executable']).resolve()) == row['selected']['file']['realpath'], 'host Python differs')
            for value in [*meta['modules'],*meta['code_maps']]:
                path = Path(value).resolve()
                if path.is_relative_to(P):
                    name = str(path.relative_to(P))
                    require(row['sources'].get(name) == digest(path), 'uncommitted project import')
                else:require(str(path) in allowed, 'unfrozen external provider:'+str(path))
            if role == 'server':
                exercised = meta['exercised']
                require(exercised['late_tokenizer_import'] and exercised['production_telemetry_initialized'] and
                        exercised['native_engine'] is False, 'mock late import coverage')
            python_runtime.validate_manifest(manifest)
        require(pins(row['source_commit']) == row['sources'], 'host source changed during probes')
        row.update(stage='closed',passed=True)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None and
            all(runtime.supervision.closed(c) for c in owner.commands)), commands_executed=owner.commands if owner else [])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        write(ROOT/'python/record.json',row);record[KEY]['python'] = compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)
    require(row['passed'], 'current frontend probe failed')


def python_contract():
    public = runtime.validation.committed(RESULT,(KEY,'python'))
    row = json.loads((ROOT/'python/record.json').read_text())
    require(compact(row) == public and row['stage'] == 'closed' and row['passed'] and
            row['ownership_closed'] and not row['active'], 'closed current frontend gate first')
    require(pins(row['source_commit']) == row['sources'], 'frontend/controller source changed')
    manifest = python_runtime.read_manifest(row['manifest']);python_runtime.validate_manifest(manifest)
    require(digest(row['template']['path']) == row['template']['sha256'], 'template changed')
    return row, manifest


def prepare():
    record = runtime.validation.committed(RESULT)
    host_row, manifest = python_contract();full, rt = native()
    require('measurement' not in record[KEY] and not (ROOT/'model').exists(), 'single held-out length plan')
    folder = ROOT/'model';folder.mkdir(mode=0o700)
    for part in ('tmp','cache'):(folder/part).mkdir()
    env = environment(folder,rt)
    config = copy.deepcopy(runtime.validation.committed(TEMPLATE)['configuration'])
    profile_raw = gen.git('show',full['source_commit']+':data/expert-profile.bin')
    require(profile_raw == (P/'data/expert-profile.bin').read_bytes(),
            'selected original expert placement profile changed')
    config['args'][config['args'].index('--expert-profile')+1] = str(P/'data/expert-profile.bin')
    config['log'] = str(folder/'engine.stderr')
    probe = dict(library=rt['preload']['library'],loader=dict(path=rt['preload']['loader'],
        sha256=digest(rt['preload']['loader'])))
    write(folder/'config.json',config)
    write(folder/'plan.json',dict(binary=full['binary'],native_validation_probe=probe))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    command = ['/bin/sh',str(P/'bench.sh'),'--config',str(folder/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens',str(MINIMUM),'--think','--runs',str(RUNS),
        '--profile-request-gates','--workload-id','native_gdn_20261018','--fail-fast-engine-error',
        '--ready-timeout','600','--out',str(folder/'bench')]
    suite = client.Suite('','',131072,RUNS,True,prefill_only=True,prefill_tokens=MINIMUM,
                        workload_id='native_gdn_20261018')
    hashes = [hashlib.sha256(json.dumps(client.chat_body(suite.prefill_prompt(i),24,True)).encode()).hexdigest()
              for i in range(RUNS)]
    assets = models.model_pins(config,full=True)
    row = dict(stage='prepared',source_commit=host_row['source_commit'],sources=host_row['sources'],
        binary=full['binary'],native_source_commit=full['source_commit'],native_validation_probe=probe,
        expert_profile=dict(source_commit=full['source_commit'],path='data/expert-profile.bin',
                            sha256=hashlib.sha256(profile_raw).hexdigest()),
        configuration=config,environment=env,command=command,request_sha256=hashes,model_assets=assets,
        generated={name:digest(folder/name) for name in ('config.json','plan.json')},
        runtime_sha256=digest(runtime.RESULT),boot=rt['boot'],
        budgets=dict(wall=1800,cpu=5400,rss_cap=104<<30,text_cap=64<<20,file_cap=128<<20,total_cap=512<<20),
        approximation=dict(formula='seconds=13.21464799046+0.000903859994578*actual_read_tokens',
            intercept_seconds=13.21464799046,seconds_per_token=.000903859994578,
            fitted_actual_tokens=[19665,39254,78445,117483],
            previous_max_leave_one_length_out_relative_error=.04309,
            interpretation='Freeze existing coarse long-route affine approximation. Evaluate held-out error without refitting;4.309percent is an observed old prediction-error comparator,not a correctness or speed tolerance.'),
        plan='One fresh selected6e40process;officialbench.sh thinkingon. Shortwarmup then4fresh minimum54613inputs:one firstlong separately,three subsequent samples. Preserveactualreadcounts,PP,engine time,clientTTFT,reuse,request hashes. This minimum was not in the6lengthfit;actualtokens maydifferfrom65536. Same131072residentINT8context,8192chunk,BF16vLLMmode10,GPUresidual,1256decode slots,5workers,spec4,cachesoff. Allproviders/pre-mainvalidation/source/model/owner/fault gates mandatory. Stopfirstfailure/budget,no retry/exclusion. No candidate,degradationattribution,decodequality/70tok/s or full256Kclaim.',
        retention='One compact held-out cohort with usefulindividualsamples and fixed-formula residuals;retire successful logs,journals,duplicate result/privateplans after verification. Selectedengine+12fixtures andcurrentnormalizercomparison remain unchanged.',
        passed=False,complete=False,active=False,ownership_closed=True)
    write(folder/'record.json',row);record[KEY]['measurement'] = compact(row);write(RESULT,record)
    print('Held-out length model plan prepared;commit before execution.',flush=True)


def run():
    record = runtime.validation.committed(RESULT)
    row = json.loads((ROOT/'model/record.json').read_text());folder = ROOT/'model'
    require(row['stage'] == 'prepared' and compact(row) == record[KEY]['measurement'] and
            not (folder/'owned').exists(), 'fresh committed held-out plan')
    _,manifest = python_contract();full,rt = native()
    require(full['binary'] == row['binary'] and digest(runtime.RESULT) == row['runtime_sha256'], 'selected package/runtime differs')
    for name,h in row['generated'].items():require(digest(folder/name) == h, 'generated input changed')
    models.model_pins(row['configuration'],row['model_assets'])
    cursor = re.findall(r'^-- cursor: (.+)$',subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15),re.M)
    require(len(cursor) == 1, 'fault cursor')
    outcome = dict(complete=False,passed=False,host_probe_accepted=False);owner = None
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def persist():
        write(folder/'execution.json',dict(outcome,active=bool(owner and owner.active),
                                         commands=owner.commands if owner else []))
    def interact(io,entry):
        identity = None
        for run_index in range(RUNS):
            for stage in ('begin','end'):
                while not (line := io.line(900)).startswith('STRATA_PROFILE_GATE '):pass
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage,phase='prefill-only',run=run_index), 'request gate differs')
                delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
                require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                if stage == 'begin':
                    claim = json.loads((folder/'engine-start.json').read_text());actual = claim['native'];pid = actual['pid']
                    live = runtime.supervision.identity(pid)
                    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
                        str(Path('/proc',str(pid),'exe').resolve()) == str(Path(row['binary']['path']).resolve()) and
                        (identity is None or identity == actual), 'owned native identity differs')
                    identity = actual
                    expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']),row['environment'])
                    actual_env = dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual_env.get(k.encode()) == v.encode() for k,v in expected.items()) and
                        set(actual_env)-{k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native startup environment differs')
                    mapped = runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                        {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
                         row['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native provider mapping differs')
                    python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],
                        row['environment'],{v['realpath'] for v in manifest['inventory']['files'].values()},row['command'][2:])
                require(not (folder/'validation.log').exists() or not (folder/'validation.log').stat().st_size, 'parameter-validation warning')
                print(f'Length holdout request{run_index} {stage}',flush=True);io.send(b'continue\n')
        io.proc.stdin.close();deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner = runtime.supervision.Owner(folder/'owned',persist)
        entry,_,_ = owner.run('bench',row['command'],row['environment'],P,interaction=interact,**row['budgets'])
        require(runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'held-out process failed')
        files = list((folder/'bench').glob('*/results.json'));require(len(files)==1,'official result missing/duplicate')
        result = json.loads(files[0].read_text());samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples)==RUNS, 'incomplete official cohort')
        for i,s in enumerate(samples):
            require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and
                s['reasoning_observed'] and s['request_sha256'] == row['request_sha256'][i] and
                s['actual_prompt_tokens'] == s['actual_read_tokens'] >= MINIMUM and s['actual_reused_tokens']==0 and
                s['generated_tokens']==24 and s['prompt_tps_source']=='engine' and s['prompt_ms']>10000, 'workload/clock differs')
        require(client.verify_request_journal(files[0].parent)['valid'], 'official request journal invalid')
        end = gen.native_end(folder,row,dict(config=row['configuration']),row['boot'],entry['pid'])
        log = (folder/'engine.stderr').read_text();proof = runtime.validation.preload_evidence(log,end['native']['pid'])
        require(proof['loader']==row['native_validation_probe']['loader']['path'] and
            'strata MoE window configuration: requested_mode=10 effective_mode=10' in log and
            all('strata prefill: layer-major candidate, tokens='+str(s['actual_read_tokens']-1)+' chunk=8192 residual_dtype=bf16' in log for s in samples) and
            not any(k in log for k in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')), 'actual route/validation differs')
        require(pins(row['source_commit'])==row['sources'],'source changed during measurement')
        runtime.current(rt);models.model_pins(row['configuration'],row['model_assets'],full=True)
        f=row['approximation'];a,b=f['intercept_seconds'],f['seconds_per_token']
        fields=('request_sha256','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens','generated_tokens',
                'prompt_ms','prompt_tps','ttft_s','thinking_requested','reasoning_observed','ok','eligible','stream_complete')
        kept=[{k:s[k] for k in fields} for s in samples]
        for s in kept:
            predicted=a+b*s['actual_read_tokens'];observed=s['prompt_ms']/1000
            s.update(fixed_formula_seconds=predicted,fixed_formula_error_seconds=predicted-observed,
                     fixed_formula_relative_error=(predicted-observed)/observed)
        median=sorted(kept[1:],key=lambda s:s['prompt_ms'])[1]
        outcome.update(complete=True,host_probe_accepted=True,samples=kept,
            first_use_index=0,subsequent_indices=[1,2,3],subsequent_median_ms=median['prompt_ms'],
            subsequent_median_pp=median['prompt_tps'],fixed_formula_median_relative_error=median['fixed_formula_relative_error'],
            previous_prediction_error_comparator_met=abs(median['fixed_formula_relative_error'])<=f['previous_max_leave_one_length_out_relative_error'],
            model_refitted=False,decode_speed_qualified=False,physical262144_qualified=False)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner:runtime.validation.finish(outcome,owner,cursor[0],row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete']);persist()
        row.update(stage='closed' if outcome['passed'] else 'failed',outcome=outcome,
                   passed=outcome['passed'],complete=outcome['complete'],active=False,
                   ownership_closed=outcome.get('ownership_closed',False))
        write(folder/'record.json',row);record[KEY]['measurement']=compact(row);write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','subsequent_median_pp','fixed_formula_median_relative_error','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'held-out length measurement failed')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare-python':prepare_python,'host':host,'prepare':prepare,'run':run}[sys.argv[1]]()
