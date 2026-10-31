#!/usr/bin/env python3
"""Exact-count prefill route qualification and current-source frontend checks.

Repository LICENSE. Official bench.sh, unchanged selected native package; separate
validated selected/candidate routes; this is not performance or adoption evidence.
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
import check_moe_split as split
P, B, A = runtime.P, runtime.B, gen.A
RESULT = A/'fused-moe-service-current.json'
TEMPLATE = A/'phase-prefill-speed-current.json'
REFERENCE = A/'prefill-runtime-comparison-current.json'
ROOT = B/'prefill-exact-route-checks-v1'
KEY = 'exact_prefill_route_checks'
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
    require(KEY not in record and not ROOT.exists(), 'fresh exact-route frontend qualification only')
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


# Actual engine rows are checked from the native route log, not inferred from
# count-only preflight. The read-minus-one relationship is a prospective check.
CASES = {'selected-'+str(n): dict(rows=n,read=n+1,split=False)
         for n in (73728,73729,81920,81921)}
CASES['split-117488'] = dict(rows=117488,read=117489,split=True)


def engine(case):
    full,rt = native()
    if case['split']:
        scope = runtime.validation.committed(RESULT,(split.KEY,))
        checks = scope.get('device_checks',[])
        require([v['mode'] for v in checks] == split.ORDER and
                all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] for v in checks),
                'all four split component gates required before model work')
        full = split.native(scope)
        return full['binaries']['strata'],rt
    return full['binary'],rt


def route(text, case):
    def fields(prefix, optional=False):
        lines = [v for v in text.splitlines() if v.startswith(prefix)]
        if optional and not lines:return None
        require(len(lines)==1,'missing/duplicate native route: '+prefix)
        return dict(re.findall(r'(\w+)=([^\s;]+)',lines[0]))
    main = fields('strata prefill: layer-major candidate, ')
    bank = fields('strata MoE expert bank: ')
    window = fields('strata MoE window configuration: ')
    work = fields('strata MoE accumulator window: ')
    require(int(main['tokens'])==case['rows'] and main['chunk']=='8192' and main['residual_dtype']=='bf16' and
            window['requested_mode']==window['effective_mode']=='10' and
            work['maximum_rows']==window['maximum_rows'],'actual engine extent/window differs')
    require(bank['effective_slots']==('128' if case['split'] else '512' if case['rows']==81921 else '384') and
            bank['staged']==('0' if case['rows']==81921 else '1'),'selected/candidate expert placement differs')
    if case['split']:
        admission = fields('strata MoE split admission: ')
        finish = fields('strata MoE split complete: ')
        require(admission['active']=='1' and admission['rows']==finish['rows']=='117488' and
                admission['primary_rows']==finish['primary_rows']=='65536' and
                admission['extra_allocation_bytes']=='531988736' and admission['prefill_slots']=='128' and
                finish['guard']==finish['final_drain']=='checked' and window['maximum_rows']=='117488' and
                int(main['residual_gpu_rows'])==case['rows'],'split allocation/route/retirement incomplete')
    return dict(engine_rows=int(main['tokens']),chunk=int(main['chunk']),residual_dtype=main['residual_dtype'],
        residual_gpu_rows=int(main['residual_gpu_rows']),residual_host_bytes=int(main['residual_host']),
        slots=int(bank['effective_slots']),staged=int(bank['staged']),maximum_window_rows=int(window['maximum_rows']),
        work={k:int(work[k]) for k in ('windows','dequants','jobs','source_H2D_bytes')},
        requested_expert_upload=fields('strata MoE expert bank complete: ',optional=case['rows']==81921),
        limitation='Requested work/bytes,not measured PCIe or VRAM traffic. Diagnostic STRATA_TRACE timings are not clean speed.')


def prepare_model(name):
    require(name in CASES,'unknown exact prefill route')
    record = runtime.validation.committed(RESULT);host,manifest = python_contract();case = CASES[name]
    binary,rt = engine(case)
    scope = record[KEY].setdefault('models',{})
    folder = ROOT/name
    require(name not in scope and not folder.exists(),'fresh exact-route model check only')
    env = environment(folder,rt)
    env.update(STRATA_TRACE='1',STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR=str(int(case['split'])))
    config = copy.deepcopy(runtime.validation.committed(TEMPLATE)['configuration'])
    profile = P/'data/expert-profile.bin'
    require(gen.git('show','6e40afc1beddd059ee518617f6c1efce1da14aca:data/expert-profile.bin')==profile.read_bytes(),
            'selected original expert profile changed')
    config['args'][config['args'].index('--expert-profile')+1] = str(profile)
    config['log'] = str(folder/'engine.stderr')
    expected = json.loads(gen.git('show','1e676d554:'+str(TEMPLATE.relative_to(P))))['model_assets']
    assets = models.model_pins(config,expected,full=True)
    probe = dict(library=rt['preload']['library'],loader=dict(path=rt['preload']['loader'],sha256=digest(rt['preload']['loader'])))
    revision = gen.git('rev-parse','HEAD').decode().strip()
    folder.mkdir(mode=0o700)
    for part in ('tmp','cache'):(folder/part).mkdir()
    write(folder/'config.json',config);write(folder/'plan.json',dict(binary=binary,native_validation_probe=probe))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    row = dict(stage='prepared',source_commit=revision,sources=pins(revision),case=case,name=name,
        configuration=config,model_assets=assets,environment=env,binary=binary,native_validation_probe=probe,
        profile=dict(path=str(profile),identity=models.file_identity(profile),sha256=digest(profile)),
        generated={n:digest(folder/n) for n in ('config.json','plan.json')},
        boot=rt['boot'],runtime_sha256=digest(runtime.RESULT),python_source=host['source_commit'],
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(folder/'config.json'),'--port',str(port),
            '--prefill-exact-tokens',str(case['read']),'--think','--runs','1','--profile-request-gates',
            '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(folder/'bench')],
        budgets=dict(wall=1500,cpu=3600,rss_cap=104<<30,text_cap=64<<20,file_cap=128<<20,total_cap=512<<20),
        passed=False,active=False,ownership_closed=True,
        criterion='One current-source officialbench thinking-on exact-count24-output model diagnostic. '
            'Count-only preflight must match timed body/read with zero reuse;observe actual native engine rows, '
            'placement/window/jobs/requested transfers,normal1256-slot decode handoff,guard/drain and exit. '
            'Selected boundary81921must retain512-slot short-tail fallback. Split117488must admit128slots '
            'and531988736B allocation with65536primary rows and checked guard. Exact sources/model/profile '
            'runtime/provider/Python/owners,no fresh fault. Diagnostic only;not clean speed,full model quality, '
            'sustained decode or physical262144qualification.',
        retention='Compact exact-count/native-route/individual latency sample and faults only. Retire successful '
            'traces,polling,normal request journals after verification. Exact length sweep follows route qualification.')
    write(folder/'record.json',row);scope[name]=compact(row);write(RESULT,record)
    print('Prepared exact-route '+name+';commit before native execution.',flush=True)


def run_model(name):
    record = runtime.validation.committed(RESULT);host,manifest = python_contract();folder = ROOT/name
    row = json.loads((folder/'record.json').read_text());binary,rt = engine(row['case'])
    require(row['name']==name and row['stage']=='prepared' and row['binary']==binary and
            record[KEY]['models'][name]==compact(row) and not (folder/'execution.json').exists(),
            'fresh committed exact-route model plan required')
    require(pins(row['source_commit'])==row['sources'] and digest(runtime.RESULT)==row['runtime_sha256'],
            'exact-route source/runtime changed')
    models.model_pins(row['configuration'],row['model_assets'])
    require(models.file_identity(row['profile']['path'])==row['profile']['identity'] and
            digest(row['profile']['path'])==row['profile']['sha256'] and
            all(digest(folder/n)==sha for n,sha in row['generated'].items()),'model inputs changed')
    runtime.guards.no_model_owners([])
    cursors=re.findall(r'^-- cursor: (.+)$',subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15),re.M)
    require(len(cursors)==1,'fresh model fault cursor required')
    outcome=dict(complete=False,host_probe_accepted=False,passed=False,model_executed=False)
    owner=None
    def persist():write(folder/'execution.json',dict(outcome,active=bool(owner and owner.active),
                                                   commands=owner.commands if owner else []))
    initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def awake():
        delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
        require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'suspend/clock discontinuity')
    def interact(io,entry):
        for stage in ('begin','end'):
            while True:
                line=io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '):break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=0),
                    'unexpected exact-route request gate')
            awake()
            if stage=='begin':
                claim=json.loads((folder/'engine-start.json').read_text());pid=claim['native']['pid']
                identity=runtime.supervision.identity(pid)
                require(identity and identity['start_ticks']==claim['native']['start_ticks'] and identity['sid']==entry['pid'] and
                        str(Path('/proc',str(pid),'exe').resolve())==str(Path(binary['path']).resolve()),'owned native identity differs')
                expected=engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']),row['environment'])
                actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
                        set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native pre-exec environment differs')
                mapped=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                require(mapped<=set(rt['runtime_files'])|{str(Path(binary['path']).resolve())} and
                        {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
                         row['environment']['ZE_ENABLE_ALT_DRIVERS']}<=mapped,'native provider map differs')
                allowed={v['realpath'] for v in manifest['inventory']['files'].values()}
                python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],row['environment'],allowed,row['command'][2:])
                outcome.update(model_executed=True,native_identity=claim['native'])
            warnings=folder/'validation.log'
            require(not warnings.exists() or warnings.stat().st_size==0,'model parameter validation warning/error')
            print('Exact-route '+name+' request '+stage,flush=True);io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=runtime.supervision.Owner(folder/'owned',persist)
        entry,out,err=owner.run('bench',row['command'],row['environment'],P,interaction=interact,**row['budgets'])
        require(runtime.supervision.closed(entry) and entry['exit_code']==0,'exact-route model failed/ownership open')
        files=list((folder/'bench').glob('*/results.json'));require(len(files)==1,'missing/duplicate benchmark result')
        result=json.loads(files[0].read_text());cohort=result['results']['prefill_only'];samples=cohort['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples)==1,'incomplete exact-route result')
        sample=samples[0];preflight=cohort['count_preflight'];case=row['case']
        require(len(preflight)==1 and preflight[0]['counted_input_tokens']==case['read'] and
                preflight[0]['body_sha256']==sample['request_sha256'] and sample['ok'] and sample['eligible'] and
                sample['stream_complete'] and sample['thinking_requested'] and sample['reasoning_observed'] and
                sample['actual_prompt_tokens']==sample['actual_read_tokens']==case['read'] and sample['actual_reused_tokens']==0 and
                sample['generated_tokens']==24 and sample['prompt_tps_source']=='engine' and sample['prompt_ms']>10000,
                'exact-route workload/count/clock differs')
        require(client.verify_request_journal(files[0].parent)['valid'],'request journal invalid')
        end=gen.native_end(folder,row,dict(config=row['configuration']),row['boot'],entry['pid'])
        text=(folder/'engine.stderr').read_text();proof=runtime.validation.preload_evidence(text,end['native']['pid'])
        require(proof['loader']==row['native_validation_probe']['loader']['path'] and
                not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),
                'native runtime error')
        observed=route(text,case)
        models.model_pins(row['configuration'],row['model_assets'],full=True)
        sample={k:sample[k] for k in ('actual_read_tokens','actual_reused_tokens','prompt_ms','prompt_tps','ttft_s',
            'generated_tokens','request_sha256','timing_frontend_state')}
        outcome.update(complete=True,host_probe_accepted=True,sample=sample,count_preflight=preflight,
            route=observed,validation=proof,native_exit_code=0,clean_speed_qualified=False,
            sustained_decode_qualified=False,physical262144_qualified=False)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner:runtime.validation.finish(outcome,owner,cursors[0],row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete']);persist()
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],outcome=outcome,
                   ownership_closed=outcome.get('ownership_closed',False),active=bool(owner and owner.active))
        write(folder/'record.json',row);record[KEY]['models'][name]=compact(row);write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'exact-route model diagnostic failed')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1] in ('prepare-python','host'):
            {'prepare-python':prepare_python,'host':host}[sys.argv[1]]()
        else:{'prepare-model':prepare_model,'run-model':run_model}[sys.argv[1]](sys.argv[2])
