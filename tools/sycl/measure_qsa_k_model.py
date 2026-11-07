#!/usr/bin/env python3
"""Own bounded production-model qualification before clean direct-K comparison.

Fresh CPU-only frontend probes cover changed benchmark/server source without
relabeling the older Python-source admission failure. Native safety, immutable
pre-exec settings, source/recipe/fixture identity and model assets remain gated.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import strata_bench as client
import relink_qsa_k_model as link
import benchmark_runtime as python_runtime
import cpu_capacity as models
import engine_once

m = link.m
P, B, RESULT = m.P, m.B, m.RESULT
ROOT = B / 'qsa-direct-k-model-validation-v1'
TEMPLATE = m.A / 'phase-prefill-speed-current.json'
REFERENCE = m.A / 'prefill-runtime-comparison-current.json'
require, digest, write = m.require, m.digest, m.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def pins(revision):
    result = m.component.gen.benchmark_pins(revision)
    profile = 'data/expert-profile.bin'
    raw = m.component.gen.git('show',revision + ':' + profile)
    require(raw == (P / profile).read_bytes(),'commit exact expert profile')
    result[profile] = hashlib.sha256(raw).hexdigest()
    return result


def native():
    record = m.component.gen.committed(RESULT); scope = record[m.KEY]
    qualification = scope['link_contract']
    row = json.loads((link.OUT / 'record.json').read_text());commands = row.pop('commands_executed')
    require(link.compact(row) == qualification and row['passed'] and row['ownership_closed'] and
            not row['active'] and row['native_qualification_reusable'] and
            all(m.runtime.supervision.closed(c) for c in commands),'closed prospective build/link qualification required')
    old, _ = link.admit(row)
    actual = row['actual_links']
    require(digest(actual['path']) == actual['sha256'] and actual['uncovered_inputs'] == 0,
            'resolved linker input proof changed')
    for name, asset in row['outputs'].items():
        previous = (old['binaries'] | old['fixtures'])[name]
        require(asset['matches_original'] and asset['sha256'] == previous['sha256'] and
                digest(asset['path']) == digest(previous['path']) == asset['sha256'],
                'all five native executable identities must match qualified source')
    checks = scope['same_build_checks']
    require([v['mode'] for v in checks] == ['properties','id-smoke','id-boundary','oracle-smoke','oracle-component'] and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] for v in checks),
            'all source-bound native gates required')
    return record, old, m.runtime.validation.committed(m.runtime.RESULT)


def base_environment(root):
    template = m.component.gen.committed(TEMPLATE)['environment_template']
    old_root = template['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root,str(root)) for k,v in template.items()}
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(root / 'cache/python'),
               UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require('LD_PRELOAD' not in env and not env.get('PYTHONPATH') and not env.get('PYTHONHOME'),
            'private native preload and qualified Python environment required')
    python_runtime.require_environment(env)
    return env


def python_contract():
    host = json.loads((ROOT / 'python/record.json').read_text())
    scope = m.runtime.validation.committed(RESULT, (m.KEY,'model_validation','python'))
    require(host['stage'] == 'closed' and host['passed'] and host['ownership_closed'] and
            scope == {k:v for k,v in host.items() if k not in ('sources','commands_executed')},
            'fresh source-bound frontend definition qualification required')
    require(pins(host['source_commit']) == host['sources'], 'qualified frontend/controller source changed')
    manifest = python_runtime.read_manifest(host['manifest']); python_runtime.validate_manifest(manifest)
    return host, manifest


def prepare_python():
    require(not ROOT.exists(),'fresh model qualification root required')
    record, _, _ = native()
    source = m.component.gen.git('rev-parse','HEAD').decode().strip()
    source_pins = pins(source)
    previous = m.runtime.validation.committed(REFERENCE,('historical_regression_audit','benchmark_python_runtime_v4'))
    require(previous['passed'] and previous['ownership_closed'],'original external-runtime inventory required')
    manifest = python_runtime.read_manifest(previous['plan']['manifest']);python_runtime.validate_manifest(manifest)
    folder = ROOT / 'python';folder.mkdir(parents=True,mode=0o700)
    (folder / 'cache').mkdir();(folder / 'tmp').mkdir()
    env = base_environment(folder)
    require(python_runtime.selection(env) == manifest['selected'],'frontend interpreter changed')
    reference = m.runtime.validation.committed(REFERENCE,('historical_regression_audit','near_dense_original_reference_v1'))
    template = Path(reference['configuration']['tokenizer']) / 'chat_template.jinja'
    expected = json.loads(m.component.gen.git('show','1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    require(digest(template) == expected[str(template)]['sha256'],'model template changed')
    row = dict(stage='prepared',source_commit=source,sources=source_pins,
        selected=manifest['selected'],manifest=previous['plan']['manifest'],template=dict(path=str(template),sha256=digest(template)),
        environment=env,commands=[[role,[manifest['selected']['path'],str(python_runtime.SOURCE),'probe',str(entry),str(template)]]
            for role,entry in [('benchmark',P / 'tools/strata_bench.py'),('server',P / 'serve/server.py')]],
        budgets=dict(wall=45,cpu=30,rss_cap=512 << 20,text_cap=512 << 10,file_cap=2 << 20,total_cap=8 << 20),
        passed=False,active=False,ownership_closed=True,GPU_executed=False,
        previous_admission='V4source admission fails on changed bench.sh/benchmark/server files;preserved as a refusal,no native launch. Reuse only its physically verified external-runtime inventory as prospective input;two fresh definition probes qualify current source.',
        criteria='Two finite CPU-only definition probes using existing MockEngine/ByteTokenizer protocol. Never construct StrataEngine or initialize SYCL/Level Zero. Exact current source/interpreter/template/environment;every project import is committed,every external module/code map is in the frozen provider inventory;normal owned closure. No model or speed qualification.')
    write(folder / 'record.json',row)
    record[m.KEY].setdefault('model_validation',{})['python'] = {k:v for k,v in row.items() if k != 'sources'}
    write(RESULT,record);print('Prepared two fresh CPU-only frontend probes;commit before execution.',flush=True)


def run_python():
    record = m.component.gen.committed(RESULT);folder = ROOT / 'python'
    row = json.loads((folder / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (folder / 'owned').exists() and
            record[m.KEY]['model_validation']['python'] == {k:v for k,v in row.items() if k != 'sources'},
            'fresh committed frontend plan required')
    require(pins(row['source_commit']) == row['sources'],'frontend source changed')
    manifest = python_runtime.read_manifest(row['manifest']);python_runtime.validate_manifest(manifest)
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(folder / 'record.json',dict(row,commands_executed=owner.commands if owner else []))
    try:
        row['stage'] = 'running';owner = m.runtime.supervision.Owner(folder / 'owned',persist);row['descriptors'] = {}
        for role,args in row['commands']:
            require(pins(row['source_commit']) == row['sources'] and python_runtime.selection(row['environment']) == row['selected'],
                    'frontend source/interpreter changed')
            entry,out,err = owner.run(role,args,row['environment'],P,**row['budgets'])
            require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                    'fresh frontend definition probe failed')
            meta = json.loads(out.read_text())
            require(str(Path(meta['executable']).resolve()) == row['selected']['file']['realpath'],'definition probe interpreter differs')
            for path in [*meta['modules'],*meta['code_maps']]:
                path = Path(path).resolve()
                if path.is_relative_to(P):
                    name = str(path.relative_to(P));require(name in row['sources'] and digest(path) == row['sources'][name],
                        'uncommitted frontend import')
                else:require(str(path) in allowed,'unfrozen frontend provider: ' + str(path))
            if role == 'server':
                require(meta['exercised']['late_tokenizer_import'] and meta['exercised']['production_telemetry_initialized'],
                        'required late frontend imports were not exercised')
                require(meta['exercised']['native_engine'] is False,'definition probe constructed native engine')
            row['descriptors'][role] = meta
            python_runtime.validate_manifest(manifest)
        row.update(stage='closed',passed=True)
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(m.runtime.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed']);persist()
        record[m.KEY]['model_validation']['python'] = {k:v for k,v in row.items() if k != 'sources'}
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)
    require(row['passed'],'fresh frontend definition qualification failed')


def prepare_model():
    record, old, rt = native();host,_ = python_contract()
    require(record[m.KEY].get('model_launch_eligible') is True,'source review/build contract must permit model launch')
    scope = record[m.KEY]['model_validation'];require('model' not in scope,'fresh model qualification only')
    root = ROOT / 'model';root.mkdir(mode=0o700)
    for name in ('tmp','cache'):(root / name).mkdir()
    reference = m.runtime.validation.committed(REFERENCE,('historical_regression_audit','near_dense_original_reference_v1'))
    config = copy.deepcopy(reference['configuration']);config['log'] = str(root / 'engine.stderr')
    profile = Path(config['args'][config['args'].index('--expert-profile') + 1])
    source = m.component.gen.git('rev-parse','HEAD').decode().strip();source_pins = pins(source)
    expected = json.loads(m.component.gen.git('show','1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    assets = models.model_pins(config,expected,full=True)
    env = base_environment(root);env.update(STRATA_TRACE='1',ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p))
    probe = dict(library=rt['preload']['library'],loader=dict(path=rt['preload']['loader'],sha256=rt['runtime_files'][rt['preload']['loader']]))
    plan = dict(binary=old['binaries']['strata'],native_validation_probe=probe)
    write(root / 'plan.json',plan);write(root / 'config.json',config)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    suite = client.Suite('', '',131072,1,True,prefill_only=True,prefill_tokens=32768,workload_id='native_gdn_20261018')
    request = client.chat_body(suite.prefill_prompt(0),24,True)
    request_hash = hashlib.sha256(json.dumps(request).encode()).hexdigest()
    row = dict(stage='prepared',source_commit=source,sources=source_pins,model_assets=assets,
        configuration=config,environment=env,binary=plan['binary'],native_validation_probe=probe,
        profile=dict(path=str(profile),identity=models.file_identity(profile),sha256=digest(profile)),
        generated={name:digest(root / name) for name in ('plan.json','config.json')},
        request_sha256=request_hash,boot=rt['boot'],runtime_sha256=digest(m.runtime.RESULT),python_source=host['source_commit'],
        command=['/bin/sh',str(P / 'bench.sh'),'--config',str(root / 'config.json'),'--port',str(port),
            '--prefill-only','--prefill-tokens','32768','--think','--runs','1','--profile-request-gates',
            '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root / 'bench')],
        budgets=dict(wall=1500,cpu=3600,rss_cap=104 << 30,text_cap=64 << 20,file_cap=128 << 20,total_cap=512 << 20),
        passed=False,active=False,ownership_closed=True,
        criteria='First model diagnostic only. Shared131072context/INT8KV,1256decode slots,8192prefill and all original placement/safety settings. Officialbench short warmup then one thinking-on >=32768fresh/zero-reuse long request/24output. Private pre-main parameter validation and STRATA_TRACE1 stage output;no full API trace. Exact source/models/profile/native/runtime/Python/provider/ownership,normal closure,no new fault. Not clean performance,decode/agent quality or physical262144qualification.')
    write(root / 'record.json',row);scope['model'] = {k:v for k,v in row.items() if k not in ('sources','model_assets')}
    write(RESULT,record);print('Prepared first traced production model diagnostic;commit before GPU launch.',flush=True)


def run_model():
    record,_,rt = native();python_host,manifest = python_contract();root = ROOT / 'model'
    row = json.loads((root / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (root / 'execution.json').exists() and
            record[m.KEY]['model_validation']['model'] == {k:v for k,v in row.items() if k not in ('sources','model_assets')},
            'fresh committed first model plan required')
    require(pins(row['source_commit']) == row['sources'] and digest(m.runtime.RESULT) == row['runtime_sha256'],
            'model controller/runtime source changed')
    models.model_pins(row['configuration'],row['model_assets'])
    require(models.file_identity(row['profile']['path']) == row['profile']['identity'] and
            digest(row['profile']['path']) == row['profile']['sha256'],'expert placement profile changed')
    require(all(digest(root / n) == sha for n,sha in row['generated'].items()),'generated model inputs changed')
    m.runtime.guards.no_model_owners([])
    cursor = re.findall(r'^-- cursor: (.+)$',subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15),re.M)
    require(len(cursor) == 1,'fresh model fault cursor required')
    outcome = dict(complete=False,host_probe_accepted=False,passed=False,model_executed=False)
    owner = None
    def persist():write(root / 'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def awake():
        delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000,'suspend/clock discontinuity')
    def interact(io,entry):
        for stage in ('begin','end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '):break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage,phase='prefill-only',run=0),
                    'unexpected first-model request gate')
            awake()
            if stage == 'begin':
                claim = json.loads((root / 'engine-start.json').read_text());pid = claim['native']['pid']
                ident = m.runtime.supervision.identity(pid)
                require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == entry['pid'] and
                        str(Path('/proc',str(pid),'exe').resolve()) == str(Path(row['binary']['path']).resolve()),'owned native identity differs')
                expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']),row['environment'])
                actual = dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native pre-exec environment differs')
                mapped = m.runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                require(mapped <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                        {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],row['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                        'native provider mapping differs')
                allowed_python = {v['realpath'] for v in manifest['inventory']['files'].values()}
                python_runtime.check_running(entry['pid'],P / 'tools/strata_bench.py',manifest['selected'],row['environment'],allowed_python,row['command'][2:])
                outcome.update(model_executed=True,native_identity=claim['native'])
            require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,'model parameter validation warning/error')
            print('First production model request '+stage,flush=True);io.send(b'continue\n')
        io.proc.stdin.close();deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic() < deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner = m.runtime.supervision.Owner(root / 'owned',persist)
        entry,out,err = owner.run('bench',row['command'],row['environment'],P,interaction=interact,**row['budgets'])
        require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0,'first model process failed')
        files = list((root / 'bench').glob('*/results.json'));require(len(files) == 1,'missing/duplicate official benchmark result')
        result = json.loads(files[0].read_text());samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == 1,'incomplete first model diagnostic')
        sample = samples[0]
        require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and sample['thinking_requested'] and
                sample['reasoning_observed'] and sample['request_sha256'] == row['request_sha256'] and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] >= 32768 and sample['actual_reused_tokens'] == 0 and
                sample['generated_tokens'] == 24 and sample['prompt_tps_source'] == 'engine' and sample['prompt_ms'] > 10000,
                'first model workload/clock differs')
        require(client.verify_request_journal(files[0].parent)['valid'],'official request journal invalid')
        end = m.component.gen.native_end(root,row,dict(config=row['configuration']),row['boot'],entry['pid'])
        text = (root / 'engine.stderr').read_text();proof = m.runtime.validation.preload_evidence(text,end['native']['pid'])
        require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                'strata prefill: layer-major candidate, tokens=' + str(sample['actual_read_tokens'] - 1) + ' chunk=8192 residual_dtype=bf16' in text and
                'strata MoE window configuration: requested_mode=10 effective_mode=10' in text and
                not any(t in text for t in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),
                'production route/runtime differs')
        models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,sample=sample,validation=proof,native_exit_code=0,
                       clean_speed_qualified=False,decode_qualified=False,physical262144_qualified=False)
    except BaseException as error:outcome['error'] = repr(error)
    finally:
        if owner:m.runtime.validation.finish(outcome,owner,cursor[0],row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']);persist()
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],outcome=outcome,
                   ownership_closed=outcome.get('ownership_closed',False),active=bool(owner and owner.active))
        write(root / 'record.json',row);record[m.KEY]['model_validation']['model'] = {k:v for k,v in row.items() if k not in ('sources','model_assets')}
        write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'first production model diagnostic failed')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare-python':prepare_python,'run-python':run_python,'prepare-model':prepare_model,'run-model':run_model}[sys.argv[1]]()
