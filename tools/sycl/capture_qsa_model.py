#!/usr/bin/env python3
"""Main-owned production selection capture using the unchanged selected engine.

Repository LICENSE. Reuses this project's qualified ownership/provider checks;
no new native build, profiler, runtime mutation or inference selection.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervision_model as reference
import qsa_selection_capture as capture

P, B, A = reference.P, reference.B, reference.A
ROOT = B / 'selected-qsa-capture-v1'
RESULT = A / 'prefill-resources-current.json'
KEY = 'selected_qsa_capture_v1'
NATIVE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
require, digest = reference.require, reference.digest
WARM = capture.headers(73)
LONG = capture.headers(6491, 32768)
NATIVE_SOURCES = ('src/prefill/prefill.cpp', 'src/prefill/layer_major.inc',
    'src/sycl/compat.cpp', 'src/core/layer.cpp', 'src/program/generate.cpp',
    'include/strata/kernels/qsa.hpp', 'src/kernels/sycl/kv.cpp')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    value = {k:v for k,v in row.items() if k not in ('inputs', 'environment', 'configuration')}
    literal = json.dumps(KEY) + ':'
    text = json.dumps(value, indent=2).replace('\n', '\n    ')
    if KEY in desired['event_reconciliation']:
        require(raw.count(literal) == 1, 'ambiguous capture scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:])
        raw = raw[:start] + text + raw[start+size:]
    else:
        anchor = '"event_reconciliation": {'
        require(raw.count(anchor) == 1, 'ambiguous result scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + text + ',', 1)
    desired['event_reconciliation'][KEY] = value
    require(json.loads(raw) == desired, 'capture changed unrelated evidence')
    write(ROOT/'record.json', row); RESULT.write_text(raw)


def original():
    base = json.loads((reference.ROOT/'record.json').read_text())
    require(base['stage'] == 'closed' and base['passed'] and base['ownership_closed'] and
        not base['active'], 'closed original engine control required')
    reference.current(base)
    return base


def environment(base):
    env = reference.environment(base, ROOT)
    require(not any(k in env for k in ('STRATA_QSA_DUMP', 'STRATA_QSA_DUMP_LAST',
        'STRATA_SEL_OVERLAP', 'STRATA_IDX_FP16_CHECK')), 'unplanned selection diagnostic')
    env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1',
        STRATA_QSA_DUMP=str(ROOT/'selected-ids.bin'), STRATA_QSA_DUMP_LAST='1')
    return env


def current(row):
    base = original()
    require(reference.runtime.validation.committed(RESULT, ('event_reconciliation',KEY,'plan')) == row['plan'],
        'commit exact capture plan before execution')
    source = row['plan']['source_commit']
    require(reference.gen.benchmark_pins(source) == row['inputs'] and
        reference.identity(row['inputs']) == row['plan']['benchmark_input_identity'] and
        row['binary'] == base['binary'] and
        digest(reference.ROOT/'record.json') == row['reference_sha256'], 'source/reference/native changed')
    for name, content in row['plan']['generated_content'].items():
        require((ROOT/name).read_text() == json.dumps(content,indent=2)+'\n' and
            digest(ROOT/name) == row['plan']['generated_sha256'][name], 'generated input changed')
    for name, pin in row['plan']['native_sources'].items():
        require(hashlib.sha256(reference.gen.git('show',NATIVE+':'+name)).hexdigest() == pin,
            'original native source recovery failed')
    require(row['environment'] == environment(base) and
        row['configuration'] == row['plan']['generated_content']['config.json'] and
        row['plan']['expected_headers'] == [list(h) for h in WARM+LONG], 'capture configuration changed')
    reference.models.model_pins(row['configuration'], base['model_assets'])
    return base


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text())['event_reconciliation'],
        'fresh single-use capture only')
    base = original(); source = reference.gen.git('rev-parse','HEAD').decode().strip()
    pins = reference.gen.benchmark_pins(source)
    for name in ('tools/sycl/capture_qsa_model.py','tools/sycl/qsa_selection_capture.py',
                 'tools/test_sycl_qsa_selection_capture.py'):
        require(name in pins, 'commit capture source first')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'): (ROOT/name).mkdir()
    config = copy.deepcopy(base['configuration']); config['log'] = str(ROOT/'engine.stderr')
    native = dict(binary=base['binary'], native_validation_probe=base['probe'])
    generated = {'config.json':config, 'plan.json':native}
    for name, content in generated.items(): write(ROOT/name,content)
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    command = ['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens','32768','--think','--runs','1','--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600',
        '--out',str(ROOT/'bench')]
    row = dict(stage='prepared',binary=base['binary'],reference_sha256=digest(reference.ROOT/'record.json'),
        inputs=pins,environment=environment(base),configuration=config,
        plan=dict(source_commit=source,native_source_commit=NATIVE,command=command,
            benchmark_input_identity=reference.identity(pins), benchmark_input_files=len(pins),
            native_sources={name:hashlib.sha256(reference.gen.git('show',NATIVE+':'+name)).hexdigest()
                for name in NATIVE_SOURCES}, generated_content=generated,
            generated_sha256={name:digest(ROOT/name) for name in generated},
            expected_headers=[list(h) for h in WARM+LONG],
            environment_changes=dict(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1',
                STRATA_QSA_DUMP=str(ROOT/'selected-ids.bin'),STRATA_QSA_DUMP_LAST='1'),
            question='Capture actual selected logical KV IDs for a source-bound QSA locality/working-set control.',
            work='Official thinking-on bench.sh:74-token warmup then39260fresh/zero-reuse input and24outputs. '
                'Unchanged original engine,1256decode slots,131072residentINT8 context,8192chunks,BF16/vLLM/'
                'ordered-exp/mode10,384packed expert bank;no profiler or native rebuild. Existing LAST=1 '
                'dumps entire final chunk,all12QSA layers. Before allowing long input,validate warmup '
                '12records at position0/73rows. After closure validate all24headers and initialized IDs, '
                'preserve final6491-row layer0 record only. No API-wide tracing;qualified private preload '
                'and parameter validation,stage progress and immutable safety settings remain enabled.',
            source_audit='Original hook follows selection on m.cs; q.memcpy and wait_and_throw use that same '
                'qualified in-order queue,host vector lives through drain. It checks neither copy/drain '
                'nor fwrite status:source recovery,parameter/runtime/error/driver/exit checks plus exact '
                'extent/header/order/causal-ID validation are mandatory;exit0 alone is insufficient. '
                'Native step width=min(pos+1,2051);cap suffix is uninitialized and excluded. '
                'Position starts0,body39259,final chunk p032768/T6491. Native layer-major admission '
                'requires resident Mode0;configured KV resident==context yields identity page map. '
                'No native buffer size,graph,queue ownership or lifetime is changed.',
            acceptance='Run committed host corruption controls first;small warmup capture must validate '
                'at benchmark begin gate before continue. Stop first failure,no automatic retry. Require '
                'complete official stream/journal,actual route/provider/environment,all15model hashes, '
                'normal owned/native closure,no validation warnings or fresh fault,exact capture extent '
                'and logical IDs. This is attribution only,no clean-speed or inference adoption.',
            budgets=dict(wall=1800,cpu=43200,rss_cap=104<<30,text_cap=8<<20,
                file_cap=1<<30,total_cap=1<<30),
            parse_budgets=dict(wall=300,cpu=240,rss_cap=2<<30,text_cap=1<<20,
                file_cap=64<<20,total_cap=1<<30),
            retention='Transient all-layer capture<=1GiB;regular fresh private file,no FIFO. Retain '
                'one validated initialized layer0 fixture<=64MiB outsideGit solely for the next '
                'selected-QSA working-set attribution comparison;review at that comparison closure. '
                'Commit compact source/commands/env/config/counts/hashes/locality/outcomes. Retire '
                'all-layer dump and successful logs/cache/transaction receipts after verification.',
            limits='IDs give logical locality,not physical traffic/cache hits/saturation. Source-defined '
                'Mode0 map is not a device page-table capture. No Q/K/V tensor contents are captured; '
                'later calibration must declare synthetic values and a numerical oracle. This diagnostic '
                'does not qualify sustained decode,quality/agent,stateRESTORE or physical262144.'),
        passed=False,complete=False,active=False,ownership_closed=True)
    save(row); print('Prepared bounded original-engine selection capture;commit plan before run.',flush=True)


def live(row, base, rt, manifest, leader):
    claim = json.loads((ROOT/'engine-start.json').read_text()); pid = claim['native']['pid']
    ident = reference.baseline.identity(pid)
    require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
        str(Path('/proc',str(pid),'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
    expected = reference.engine_once.native_environment(json.loads((ROOT/'plan.json').read_text()),row['environment'])
    actual = dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()) and
        set(actual)-{k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native environment differs')
    mapped = reference.runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
    require(mapped <= set(rt['runtime_files'])|{str(Path(row['binary']['path']).resolve())} and
        {base['NEO'],base['probe']['loader']['path'],base['probe']['library']['path']} <= mapped, 'native provider differs')
    wrapper = reference.baseline.identity(claim['wrapper']['pid'])
    require(wrapper and wrapper['sid'] == leader['pid'], 'wrapper ownership differs')
    server = reference.baseline.identity(wrapper['ppid'])
    require(server and server['sid'] == leader['pid'], 'server ownership differs')
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    command = row['plan']['command']; port = command[command.index('--port')+1]
    reference.python_runtime.check_running(leader['pid'],P/'tools/strata_bench.py',manifest['selected'],
        row['environment'],allowed,command[2:])
    reference.python_runtime.check_running(server['pid'],P/'serve/server.py',manifest['selected'],
        row['environment'],allowed,['--engine','strata','--config',str(ROOT/'config.json'),'--port',port])


def parse():
    row = json.loads((ROOT/'record.json').read_text())
    result = capture.inspect(ROOT/'selected-ids.bin',row['plan']['expected_headers'],LONG[0],True)
    fixture = capture.extract(ROOT/'selected-ids.bin',result,ROOT/'layer0-final-ids.bin')
    write(ROOT/'analysis.json',dict(raw_bytes=result['bytes'],raw_sha256=result['sha256'],
        record_count=len(result['records']),valid_entries=sum(r['valid_entries'] for r in result['records']),
        ignored_uninitialized_entries=sum(r['unused_cap_entries'] for r in result['records']),
        fixture=fixture,locality=result['locality']))
    print('Validated all24records;retained exact final-chunk layer0 fixture.',flush=True)


def run(row):
    require(row['stage'] == 'prepared' and not (ROOT/'owned').exists(), 'single-use prepared capture required')
    base = current(row); rt = reference.runtime.validation.committed(reference.runtime.RESULT)
    _, manifest = reference.python_runtime.admitted(reference)
    cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    owner = None; outcome = dict(complete=False,host_probe_accepted=False)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def persist():
        write(ROOT/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interact(io, leader):
        poll = io.poll
        def checked_poll():
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
            require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000, 'suspend/clock discontinuity')
            poll()
        io.poll = checked_poll
        for stage in ('begin','end'):
            while True:
                line = io.line(1200)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage,phase='prefill-only',run=0), 'request gate differs')
            require(not (ROOT/'validation.log').exists() or not (ROOT/'validation.log').stat().st_size,
                'parameter warning')
            live(row,base,rt,manifest,leader)
            if stage == 'begin':
                warm = capture.inspect(ROOT/'selected-ids.bin',WARM)
                outcome['warmup'] = dict(bytes=warm['bytes'],sha256=warm['sha256'],records=len(warm['records']),
                    rows_per_layer=73,causal_ordered_identity_passed=True)
                persist(); print('Small74-token capture passed;allowing official32K+ input.',flush=True)
            else:
                expected_bytes = sum(16+h[2]*h[3]*4 for h in WARM+LONG)
                require((ROOT/'selected-ids.bin').stat().st_size == expected_bytes, 'final dump extent differs')
                print('Long request complete;checking closure and capture.',flush=True)
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic()<deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'running'; owner = reference.baseline.Owner(ROOT/'owned',persist)
        test, out, err = owner.run('host-controls',[sys.executable,'-I','-S','-B',
            str(P/'tools/test_sycl_qsa_selection_capture.py')],row['environment'],P,
            wall=30,cpu=20,rss_cap=256<<20,text_cap=1<<20,file_cap=1<<20,total_cap=1<<30)
        require(reference.baseline.closed(test) and test['exit_code']==0 and
            json.loads(out.read_text()) == dict(cases=9,passed=True), 'host corruption controls failed')
        outcome['host_controls'] = dict(cases=9,passed=True)
        require(not (ROOT/'selected-ids.bin').exists(), 'capture file must be fresh')
        entry,_,_ = owner.run('bench',row['plan']['command'],row['environment'],P,
            interaction=interact,**row['plan']['budgets'])
        require(reference.baseline.closed(entry) and entry['exit_code']==0, 'model owner failed: '+str(entry.get('error')))
        end = reference.gen.native_end(ROOT,dict(binary=base['binary'],native_validation_probe=base['probe']),
            dict(config=row['configuration']),base['boot'],entry['pid'])
        result_file, = (ROOT/'bench').glob('*/results.json'); official = json.loads(result_file.read_text())
        samples = official['results']['prefill_only']['requests']
        require(official['complete'] and not official['results']['failures'] and len(samples)==1, 'benchmark incomplete')
        s = samples[0]
        require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and
            s['reasoning_observed'] and s['actual_read_tokens']==s['actual_prompt_tokens']==39260 and
            s['actual_reused_tokens']==0 and s['generated_tokens']==24, 'workload differs')
        sys.path.insert(0,str(P/'tools')); from strata_bench import verify_request_journal
        verified = verify_request_journal(result_file.parent)
        require(verified['valid'] and verified['complete'], 'request journal invalid')
        text = (ROOT/'engine.stderr').read_text()
        preload = reference.runtime.validation.preload_evidence(text,end['native']['pid'])
        require(preload['loader']==base['probe']['loader']['path'] and
            text.count('strata prefill: layer-major candidate, tokens=39259 chunk=8192 residual_dtype=bf16')==1 and
            text.count('score_pitch=33 operand_layout=row-major softmax=parallel-exp-ascending-sum')==1 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10')==1 and
            not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called',
                'teardown uncertain','strata UR validation failure:')), 'selected route/runtime differs')
        parsed,_,_ = owner.run('parse',[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'parse'],
            row['environment'],P,**row['plan']['parse_budgets'])
        require(reference.baseline.closed(parsed) and parsed['exit_code']==0, 'capture parser failed: '+str(parsed.get('error')))
        analysis = json.loads((ROOT/'analysis.json').read_text())
        require(analysis['record_count']==24 and analysis['fixture']['header']==list(LONG[0]) and
            digest(analysis['fixture']['path'])==analysis['fixture']['sha256'], 'fixture binding differs')
        current(row); print('Capture validated;checking all15model hashes.',flush=True)
        reference.models.model_pins(row['configuration'],base['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,analysis=analysis,native=end['native'],
            model_hashes_matched=15,request={k:s[k] for k in ('request_sha256','actual_read_tokens',
                'actual_reused_tokens','generated_tokens','prompt_ms')},performance_adopted=False,
            hardware_saturation_qualified=False)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner: reference.runtime.validation.finish(outcome,owner,cursor,base['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        row.update(outcome,active=bool(owner and owner.active))
        row['stage'] = 'captured' if row['passed'] else 'failed'
        persist(); save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','complete','error')}),flush=True)
    return int(not row['passed'])


def main():
    if sys.argv[1:] == ['parse']: parse(); return 0
    require(sys.argv[1:] in (['prepare'],['run']), 'use prepare|run|parse')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reference.runtime.guards.no_model_owners([])
        reference.runtime.guards.driver_guard()
        if sys.argv[1] == 'prepare': prepare(); return 0
        return run(json.loads((ROOT/'record.json').read_text()))


if __name__ == '__main__': raise SystemExit(main())
