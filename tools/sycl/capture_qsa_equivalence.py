#!/usr/bin/env python3
"""Finite exact selected-ID census on unchanged selected6e40; no speed claim.

Repository LICENSE. Reuses project-owned admission functions from96bd6c987;
new collection/analysis does not change inference or borrow XeStrata code.
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
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'tools/sycl'))
import benchmark_runtime as python_runtime
import cpu_capacity as models
import engine_once
import generation_engine as gen
import measurement_owner as owner_module
import runtime_probe as runtime
import supervision_model as shared
import qsa_selection_capture as capture
from measure_down_scatter import postchecks
B,A=gen.B,gen.A
ROOT=B/'qsa-exact-lists-v1'
RESULT=A/'fused-moe-service-current.json'
REFERENCE=A/'prefill-runtime-comparison-current.json'
TEMPLATE=A/'phase-prefill-speed-current.json'
KEY='qsa_exact_lists'
SOURCE='6e40afc1beddd059ee518617f6c1efce1da14aca'
WARM=capture.headers(73)
LONG=capture.headers(6491,32768)
EXPECTED=WARM+LONG
require,digest=runtime.require,runtime.digest

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(row):
    container=json.loads(RESULT.read_text())
    container[KEY]={k:v for k,v in row.items() if k not in ('inputs','environment','model_assets','python_contract')}
    write(RESULT,container);write(ROOT/'record.json',row)

def physically_closed(entry):
    return entry.get('direct_child_reaped') is True and entry.get('session_empty') is True and \
        entry.get('survivors') == [] and entry.get('ownership_status') == 'observed' and type(entry.get('exit_code')) is int

def finish_benchmark(io):
    """Drain the official CLI's human summary after its final request gate."""
    io.proc.stdin.close()
    deadline = time.monotonic() + 90
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        require(time.monotonic() < deadline, 'benchmark teardown deadline')
        if select.select([io.proc.stdout], [], [], .05)[0]:
            io.read()  # Interactive retains its existing finite pending/output caps.
    require(b'STRATA_PROFILE_GATE' not in io.pending, 'request gate after final boundary')
    io.pending.clear()
    io.exit(1)

def python_contract():
    public = runtime.validation.committed(REFERENCE,
        ('historical_regression_audit', python_runtime.SCOPE))
    original = json.loads((python_runtime.HOST / 'record.json').read_text())
    require(public == python_runtime.compact(original) and public['passed'] and
        public['complete'] and public['ownership_closed'] and not public['active'] and
        all(owner_module.closed(c) for c in original['commands']), 'closed Python qualification required')
    # Reuse provider qualification only for unchanged child scripts. The new
    # Main-only reader is not imported by bench/server/engine_once children.
    # Standalone tools/test_*.py are not imported by the benchmark/server. A
    # new unrelated unit test must not silently enlarge the qualified runtime.
    def child_source(n):
        return n in ('bench.sh', 'tools/sycl/engine_once.py') or n.startswith('serve/') or (
            n.startswith('tools/') and n.count('/') == 1 and not n.startswith('tools/test_'))
    names = [n for n in original['benchmark_inputs'] if child_source(n)]
    current_names = [n for n in gen.benchmark_pins(gen.git('rev-parse', 'HEAD').decode().strip()) if child_source(n)]
    require(set(names) == set(current_names), 'child script inventory changed')
    for name, raw in gen.blobs(P, public['source_commit'], names):
        require(hashlib.sha256(raw).hexdigest() == original['benchmark_inputs'][name] == digest(P / name),
            'qualified child source changed: ' + name)
    manifest = python_runtime.read_manifest(public['plan']['manifest'])
    python_runtime.validate_manifest(manifest)
    return dict(qualification_source=public['source_commit'], manifest=public['plan']['manifest'],
        unchanged_child_sources={n:original['benchmark_inputs'][n] for n in names}), manifest

def native():
    row = runtime.validation.committed(runtime.RESULT)
    full = runtime.current(row)
    require(full.get('binary_origin_source_commit', full['source_commit']) == SOURCE,
        'selected original native source required')
    return full, row

def request_hash(minimum):
    sys.path.insert(0, str(P / 'tools'))
    import strata_bench as client
    suite = client.Suite('', '', 131072, 1, True, log=lambda *a: None,
        prefill_only=True, prefill_tokens=minimum, workload_id='native_gdn_20261018')
    return hashlib.sha256(json.dumps(client.chat_body(suite.prefill_prompt(0), 24, True)).encode()).hexdigest()

def prepared_environment(root, neo):
    template = gen.committed(TEMPLATE)['environment_template']
    old = template['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old, str(root)) for k,v in template.items()}
    require(not env.get('LD_PRELOAD') and env['CUDA_MODULE_LOADING'] == 'EAGER', 'private preload required')
    env.update(STRATA_TRACE='1', STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0', STRATA_PREFILL_CPU_WAIT_PROFILE='0',
        STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', ZE_ENABLE_ALT_DRIVERS=neo,
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
        PYTHONNOUSERSITE='1', PYTHONPYCACHEPREFIX=str(root / 'cache/python'))
    require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
        env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified safety settings required')
    python_runtime.require_environment(env)
    return env


def environment(neo):
    env=prepared_environment(ROOT,neo)
    require(not any(k in env for k in ('STRATA_QSA_DUMP','STRATA_QSA_DUMP_LAST','STRATA_SEL_OVERLAP','STRATA_IDX_FP16_CHECK')),'unexpected diagnostic')
    env.update(STRATA_QSA_DUMP=str(ROOT/'selected-ids.bin'),STRATA_QSA_DUMP_LAST='1')
    return env


def admission(row,full_models=False):
    full,rt=native()
    require(full['binary']==row['binary'] and digest(runtime.RESULT)==row['runtime_sha256'] and
        digest(TEMPLATE)==row['template_sha256'] and rt['boot']==row['boot'],'native/runtime/template changed')
    require(gen.benchmark_pins(row['source_commit'])==row['inputs'],'measurement input changed')
    contract,manifest=python_contract();require(contract==row['python_contract'],'benchmark providers changed')
    shared.profile_pin(row['configuration'],row['source_commit'],row['profile'])
    models.model_pins(row['configuration'],row['model_assets'],full=full_models)
    require(row['environment']==environment(row['NEO']) and python_runtime.selection(row['environment'])==manifest['selected'] and
        request_hash(32768)==row['plan']['request_sha256'],'environment/workload changed')
    for name,pin in row['plan']['generated_sha256'].items():require(digest(ROOT/name)==pin,'generated input changed')
    for name,pin in row['plan']['native_sources'].items():
        require(hashlib.sha256(gen.git('show',SOURCE+':'+name)).hexdigest()==pin,'native source unrecoverable')
    return rt,manifest


def prepare():
    container=gen.committed(RESULT)
    require(KEY not in container and not ROOT.exists(),'fresh single-use capture required')
    require(container['complete'] and container['passed'] and container['selected_qsa_service']['analysis_passed'] and
        container['qsa_exp2_lowering']['complete'] and not container['qsa_exp2_lowering']['model_adopted'],'close prior source/service/exp2 decision first')
    full,rt=native();source=gen.git('rev-parse','HEAD').decode().strip();inputs=gen.benchmark_pins(source)
    require(str(Path(__file__).resolve().relative_to(P)) in inputs,'commit controller first')
    ref=runtime.validation.committed(REFERENCE,('historical_regression_audit','supervision_model_control_v6'))
    require(ref['complete'] and ref['passed'] and ref['ownership_closed'] and ref['binary']==full['binary'],'same-native control required')
    cfg=copy.deepcopy(ref['configuration']);cfg['log']=str(ROOT/'engine.stderr')
    contract,manifest=python_contract()
    original=json.loads(gen.git('show','1e676d554:'+str(TEMPLATE.relative_to(P))))['model_assets']
    print('Hashing all15 model assets before finite capture.',flush=True)
    assets=models.model_pins(cfg,original,full=True)
    loader=rt['preload']['loader'];probe=dict(library=rt['preload']['library'],loader=dict(path=loader,sha256=rt['runtime_files'][loader]))
    ROOT.mkdir(mode=0o700)
    for folder in ('tmp','cache'):(ROOT/folder).mkdir()
    generated={'config.json':cfg,'plan.json':dict(binary=full['binary'],native_validation_probe=probe)}
    for name,value in generated.items():write(ROOT/name,value)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens','32768','--think','--runs','1','--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')]
    row=dict(stage='prepared',source_commit=source,native_source=SOURCE,binary=full['binary'],
        runtime_sha256=digest(runtime.RESULT),template_sha256=digest(TEMPLATE),boot=rt['boot'],
        NEO=next(n for n in rt['runtime_files'] if 'libze_intel_gpu' in n),probe=probe,
        configuration=cfg,profile=shared.profile_pin(cfg,source),python_contract=contract,
        inputs=inputs,model_assets=assets,environment=environment(next(n for n in rt['runtime_files'] if 'libze_intel_gpu' in n)),
        plan=dict(command=command,request_sha256=request_hash(32768),generated_content=generated,
            generated_sha256={name:digest(ROOT/name) for name in generated},expected_headers=EXPECTED,
            source_input_count=len(inputs),source_input_sha256=canonical(inputs),model_assets_sha256=canonical(assets),
            benchmark_python_qualification=contract['qualification_source'],
            native_sources={name:hashlib.sha256(gen.git('show',SOURCE+':'+name)).hexdigest() for name in
                ('src/prefill/prefill.cpp','src/prefill/layer_major.inc','src/sycl/compat.cpp','src/kernels/sycl/qsa_select.cpp',
                 'src/kernels/sycl/qsa_bf16_prefix.cpp','include/strata/kernels/qsa.hpp')},
            environment_delta=dict(STRATA_QSA_DUMP=str(ROOT/'selected-ids.bin'),STRATA_QSA_DUMP_LAST='1'),
            work='Official thinking-on zero-reuse39260input/24output after74-token warmup;selected6e40 unchanged,1256decode slots,131072resident INT8 context,8192chunks,Mode10,BF16 residual,FP16 expert weights/hidden. No profiler or new native build.',
            source_audit='Existing LAST=1 hook copies whole final chunk for all12QSA layers on the same qualified in-order queue,waits before reading the live host vector,and appends a regular private file. Reuse of prior qualified hook;no new native bounds/lifetime/collective/graph behavior. Copy/write errors require exact closed extent,headers,causal ordered initialized IDs,provider/exit/fault gates;exit0 alone insufficient.',
            acceptance='One fresh process,no retry. Validate12warmup records at begin gate before allowing long request. Require24exact headers,complete actual thinking-on zero-reuse official request and journal,selected route,pre-main validation,exact providers,unchanged15model hashes,native/owned normalexit0 and no fresh faults. Compare exact initialized full lists separately per layer/chunk and4/16/64/256query windows. Never infer whole-list equality from overlap or across KV heads/layers.',
            interpretation='Diagnostic prerequisites only. A negative complete-list census rejects only exact-full-list grouping for this captured32K tail,not shared blocks,union/masking,all model workloads or the dominant8192full-chunk geometry. No physical traffic,saturation,speed,precision/adoption or262144/decode qualification.',
            budgets=dict(wall=1200,cpu=7200,rss_cap=96<<30,text_cap=8<<20,file_cap=768<<20,total_cap=1<<30),
            parse_budgets=dict(wall=300,cpu=240,rss_cap=2<<30,text_cap=1<<20,file_cap=768<<20,total_cap=1<<30),
            retention='Main owns <=646213056B all-layer IDs outsideGit solely for this exact-list census and current grouping decision. Commit compact counts/decisions/source/config/hashes;no permanent full histograms or all-layer payload. Retire raw/logs/cache/one-use controller after decision. Retain a selected record only for an explicitly named qualified next prototype;otherwise no fixture remains.'),
        host_tests=dict(source='b34fdbd27',cases=12,passed=True,command='/usr/bin/python3.12 -I -S -B tools/test_sycl_qsa_selection_capture.py'),
        passed=False,complete=False,active=False,ownership_closed=True,model_adopted=False)
    admission(row);save(row);print('Prepared finite exact-list capture;commit generated plan before run.',flush=True)


def live(row,rt,manifest,leader):
    claim=json.loads((ROOT/'engine-start.json').read_text());pid=claim['native']['pid'];ident=owner_module.identity(pid)
    require(ident and ident['start_ticks']==claim['native']['start_ticks'] and ident['sid']==leader['pid'] and
        str(Path('/proc',str(pid),'exe').resolve())==str(Path(row['binary']['path']).resolve()),'native identity differs')
    expected=engine_once.native_environment(json.loads((ROOT/'plan.json').read_text()),row['environment'])
    actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
        set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
    mapped=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
    require(mapped<=set(rt['runtime_files'])|{str(Path(row['binary']['path']).resolve())} and
        {row['NEO'],row['probe']['loader']['path'],row['probe']['library']['path']}<=mapped,'native providers differ')
    wrapper=owner_module.identity(claim['wrapper']['pid']);require(wrapper and wrapper['sid']==leader['pid'],'wrapper differs')
    server=owner_module.identity(wrapper['ppid']);require(server and server['sid']==leader['pid'],'server differs')
    allowed={v['realpath'] for v in manifest['inventory']['files'].values()};command=row['plan']['command'];port=command[command.index('--port')+1]
    python_runtime.check_running(leader['pid'],P/'tools/strata_bench.py',manifest['selected'],row['environment'],allowed,command[2:])
    python_runtime.check_running(server['pid'],P/'serve/server.py',manifest['selected'],row['environment'],allowed,
        ['--engine','strata','--config',str(ROOT/'config.json'),'--port',port])


def parse():
    result=capture.inspect(ROOT/'selected-ids.bin',EXPECTED,equivalence_enabled=True)
    write(ROOT/'analysis.json',dict(bytes=result['bytes'],sha256=result['sha256'],records=[
        {k:r[k] for k in ('header','sha256','valid_entries','unused_cap_entries','exact_lists')} for r in result['records']]))
    print('All24records validated;exact initialized-list classes computed independently.',flush=True)


def run():
    row=json.loads((ROOT/'record.json').read_text());container=gen.committed(RESULT)
    require(container[KEY]['plan']==row['plan'] and row['stage']=='prepared' and not (ROOT/'execution.json').exists(),'fresh committed plan only')
    rt,manifest=admission(row);cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    owner=None;outcome=dict(complete=False,passed=False);initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def persist():write(ROOT/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interaction(io,leader):
        poll=io.poll
        def checked_poll():
            delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
            require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'suspend/clock discontinuity');poll()
        io.poll=checked_poll
        for stage in ('begin','end'):
            while True:
                line=io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '):break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=0),'request gate differs')
            require(not (ROOT/'validation.log').exists() or not (ROOT/'validation.log').stat().st_size,'parameter warning')
            live(row,rt,manifest,leader)
            if stage=='begin':
                warm=capture.inspect(ROOT/'selected-ids.bin',WARM)
                outcome['warmup']=dict(bytes=warm['bytes'],sha256=warm['sha256'],records=len(warm['records']));persist()
                print('Small74-token dump passed;allowing official32K+ request.',flush=True)
            else:
                require((ROOT/'selected-ids.bin').stat().st_size==sum(16+h[2]*h[3]*4 for h in EXPECTED),'capture extent differs')
                print('Long request complete;closing native owner before parsing.',flush=True)
            io.send(b'continue\n')
        finish_benchmark(io)
    try:
        row['stage']='running';owner=owner_module.Owner(ROOT/'owned',persist)
        entry,_,_=owner.run('bench',row['plan']['command'],row['environment'],P,interaction=interaction,**row['plan']['budgets'])
        require(owner_module.closed(entry) and entry['exit_code']==0,'benchmark failed: '+str(entry.get('error')))
        end=gen.native_end(ROOT,dict(binary=row['binary'],native_validation_probe=row['probe']),dict(config=row['configuration']),row['boot'],entry['pid'])
        path,=(ROOT/'bench').glob('*/results.json');official=json.loads(path.read_text());cohort=official['results']['prefill_only']
        require(official['complete'] and not official['results']['failures'] and cohort['complete'] and len(cohort['requests'])==1,'incomplete official request')
        sample=cohort['requests'][0]
        require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and sample['thinking_requested'] and
            not sample['eligibility_errors'] and sample['request_sha256']==row['plan']['request_sha256'] and
            sample['actual_prompt_tokens']==sample['actual_read_tokens']==39260 and sample['actual_reused_tokens']==0 and sample['generated_tokens']==24,'workload differs')
        sys.path.insert(0,str(P/'tools'));from strata_bench import verify_request_journal
        journal=verify_request_journal(path.parent);require(journal['valid'] and journal['complete'],'journal invalid')
        text=(ROOT/'engine.stderr').read_text();preload=runtime.validation.preload_evidence(text,end['native']['pid'])
        footers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*',text,re.M))
        require(len(footers)==2 and tuple(map(int,footers[0].groups()[:3]))==(74,0,74) and tuple(map(int,footers[1].groups()))==(39260,0,39260,24),'native interval differs')
        require(preload['loader']==row['probe']['loader']['path'] and
            text.count('strata prefill: layer-major candidate, tokens=39259 chunk=8192 residual_dtype=bf16')==1 and
            text.count('score_pitch=33 operand_layout=row-major softmax=parallel-exp-ascending-sum')==1 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10')==1 and
            not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|retirement failed',text),'selected route/runtime differs')
        parsed,_,_=owner.run('parse',[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'parse'],row['environment'],P,**row['plan']['parse_budgets'])
        require(owner_module.closed(parsed) and parsed['exit_code']==0,'parser failed: '+str(parsed.get('error')))
        analysis=json.loads((ROOT/'analysis.json').read_text());require(len(analysis['records'])==24,'record count differs')
        print('Capture parsed;checking all15 final model hashes.',flush=True);admission(row,full_models=True)
        outcome.update(passed=True,complete=True,analysis=analysis,native=end['native'],model_hashes_matched=15,
            request={k:sample[k] for k in ('request_sha256','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms')})
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=bool(owner and owner.active is None and all(physically_closed(c) for c in owner.commands))
        postchecks(outcome,row['boot'],cursor);outcome['passed']=bool(outcome['passed'] and outcome['complete'])
        row.update(outcome,active=bool(owner and owner.active));row['stage']='captured' if row['passed'] else 'failed';persist();save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return int(not row['passed'])


if __name__=='__main__':
    if sys.argv[1:]==['parse']:parse();raise SystemExit(0)
    require(sys.argv[1:] in (['prepare'],['run']),'prepare|run|parse')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([]);runtime.guards.driver_guard()
        if sys.argv[1]=='prepare':prepare()
        else:raise SystemExit(run())
