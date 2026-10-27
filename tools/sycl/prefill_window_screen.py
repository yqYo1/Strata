#!/usr/bin/env python3
"""Unchanged-engine, >=64K actual-input screen; repository LICENSE.

Reuse the closed V6 source/runtime/ownership qualification without changing its
inputs or outcome. This is an input-size screen, not a kernel A/B comparison.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervision_model as prior

P, B, RESULT = prior.P, prior.B, prior.RESULT
ROOT = B / 'prefill-single-window-64k-v2'
KEY = 'single_window_64k_screen_v2'
REVIEW = 'prefill_single_window_screen_source_contract_v2_20261027'
require, digest, identity = prior.require, prior.digest, prior.identity


def reference():
    row = json.loads((prior.ROOT / 'record.json').read_text())
    require(prior.runtime.validation.committed(RESULT, ('historical_regression_audit', prior.KEY)) == prior.compact(row)
        and row['stage'] == 'closed' and row['complete'] and row['passed'] and row['ownership_closed']
        and not row['active'] and len(row['processes']) == 6 and all(p['passed'] for p in row['processes'])
        and row['analysis']['mechanism_speed_screen_passed'] is False, 'closed original V6 reference required')
    runtime = prior.current(row)
    prior.reviewed(row)
    return row, runtime


def request_hashes():
    sys.path.insert(0, str(P / 'tools'))
    import strata_bench as bench
    suite = bench.Suite('', None, 131072, 4, True, workload_id='native_gdn_20261018',
        prefill_only=True, prefill_tokens=65536)
    return [hashlib.sha256(json.dumps(bench.chat_body(suite.prefill_prompt(i), 24, True)).encode()).hexdigest()
        for i in range(4)]


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    public = {k:v for k,v in row.items() if k not in ('sources', 'inputs')}
    literal = json.dumps(KEY) + ':'; value = json.dumps(public, indent=2).replace('\n', '\n    ')
    if KEY in desired['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous screen scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:])
        raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'
        require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    desired['historical_regression_audit'][KEY] = public
    require(json.loads(raw) == desired, 'screen update changed other evidence')
    prior.write(ROOT / 'record.json', row); RESULT.write_text(raw)


def prepare():
    require(not ROOT.exists() and KEY not in prior.gen.committed(RESULT)['historical_regression_audit'], 'single-use screen')
    ref, _ = reference(); revision = prior.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = prior.gen.benchmark_pins(revision)
    ROOT.mkdir(mode=0o700)
    inputs = []
    for index in range(3):
        root = ROOT / str(index); root.mkdir()
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(ref['configuration']); config['log'] = str(root / 'engine.stderr')
        prior.write(root / 'config.json', config)
        prior.write(root / 'plan.json', dict(binary=ref['binary'], native_validation_probe=ref['probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        inputs.append(dict(root=str(root), config=config, environment=prior.environment(ref, root),
            files={name:digest(root / name) for name in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', '65536', '--think', '--runs', '4', '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]))
    row = dict(stage='prepared', sources=sources, inputs=inputs, processes=[], complete=False, passed=False,
        active=False, ownership_closed=True, plan=dict(source_commit=revision, inputs_sha256=identity(inputs),
            reference_record_sha256=digest(prior.ROOT / 'record.json'), binary=ref['binary'], request_hashes=request_hashes(),
            process_inputs=[{k:v for k,v in p.items() if k not in ('config','environment')} for p in inputs],
            question='What clean prefill speed does unchanged6e40 achieve with >=64K input in one MoE window per layer?',
            work='Three fresh serial processes;four78451fresh/zero-reuse/thinking-on/24output requests each. Same131072INT8 context,8192chunks,BF16/vLLM/mode10,384prefill bank,1256decode slots,validation preload and safety settings as V6. Original baseline supervisor,not the rejected candidate.',
            acceptance='All source/provider/request/route/journal/clock/CPU/closure/fault gates pass. First use separate;three subsequent medians and all9samples retained. Throughput screen passes only if every process subsequent median>=1000tok/s. This is not a same-input speedup,mechanism adoption,decode/quality or physical262144 qualification. Stop first failure;no retries or early speed selection.',
            cpu_gate_maximum=.20, timing_floor_ms=10000, expected_prompt_tokens=78451, expected_moe_rows=78450,
            budgets=dict(wall=2400, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20, file_cap=64 << 20, total_cap=128 << 20),
            whole_bytes=384 << 20,
            retention='Keep individual clocks,source/config/runtime/binary pins and decisions. Normal raw output retires after replay. Existing V6 generated inputs,closed ownerV3/PythonV4 qualification and record have this named reader only through this screen;reassess and retire at closure.'))
    save(row)
    print('Prepared three-process unchanged-engine screen;commit plan and exact source review before execution.', flush=True)


def current(row):
    require(prior.runtime.validation.committed(RESULT, ('historical_regression_audit', KEY, 'plan')) == row['plan'], 'commit exact screen plan')
    require(prior.gen.benchmark_pins(row['plan']['source_commit']) == row['sources'] and
        identity(row['inputs']) == row['plan']['inputs_sha256'] and request_hashes() == row['plan']['request_hashes']
        and digest(prior.ROOT / 'record.json') == row['plan']['reference_record_sha256'], 'screen inputs changed')
    ref, runtime = reference()
    require(ref['binary'] == row['plan']['binary'], 'screen binary changed')
    for p in row['inputs']:
        root = Path(p['root']); config = copy.deepcopy(ref['configuration']); config['log'] = str(root / 'engine.stderr')
        require(p['config'] == config and p['environment'] == prior.environment(ref, root)
            and all(digest(root / n) == s for n,s in p['files'].items()), 'generated configuration/environment changed')
    return ref, runtime


def parse_samples(row, samples, text):
    require(len(samples) == 4, 'four eligible requests required')
    for s, pin in zip(samples, row['plan']['request_hashes']):
        require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and s['reasoning_observed']
            and s['request_sha256'] == pin and s['actual_prompt_tokens'] == s['actual_read_tokens'] == 78451
            and s['actual_reused_tokens'] == 0 and s['generated_tokens'] == 24 and s['prompt_tps_source'] == 'engine'
            and row['plan']['timing_floor_ms'] <= s['prompt_ms'] < 1500000, 'workload/clock eligibility differs')
    require(text.count('strata prefill: layer-major candidate, tokens=78450 chunk=8192 residual_dtype=bf16') == 4
        and text.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 4
        and not any(v in text for v in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
            'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'selected route/runtime differs')
    work = [dict(v.split('=',1) for v in line.split(': ',1)[1].split())
        for line in text.splitlines() if line.startswith('strata MoE accumulator window:')]
    def counters(prefix):
        return [dict(v.split('=',1) for v in line.split(': ',1)[1].split())
            for line in text.splitlines() if line.startswith(prefix)]
    gu, down, bank = (counters(s) for s in ('strata MoE GU fusion:', 'strata MoE Down fusion:', 'strata MoE expert bank complete:'))
    require(len(work) == len(gu) == len(down) == len(bank) == 4, 'missing completed route counters')
    for w,g,d,b in zip(work,gu,down,bank):
        require(int(w['windows']) == 48 and int(w['maximum_rows']) == 78450
            and int(w['operator_max']) == 8192 and int(w['GPU_resident_windows_only']) == 1
            and 0 < int(w['dequants']) <= 48*512 and int(w['dequants']) <= int(w['jobs']) <= 48*512*10
            and int(w['jobs']) == int(g['fused_jobs'])+int(g['separate_jobs']) == int(d['separate_jobs'])
            and int(d['fused_jobs']) == int(d['fused_rows']) == 0 and int(g['fused_jobs']) > 0
            and int(g['fused_rows']) >= 2334*int(g['fused_jobs']) and int(g['min_rows']) == int(d['min_rows']) == 2334
            and int(w['source_H2D_bytes']) == 78450*10*4*48 and int(b['uploads']) == int(w['dequants'])
            and int(b['permanent_hits']) == 0 and int(b['total_slots']) == 384 and b['final_drain'] == 'checked',
            'one-window route/work conservation differs')
    clocks = list(map(int, re.findall(r'^strata serve: prompt 78451 tokens = 0 reused \+ 78451 read in (\d+) ms ', text, re.M)))
    require(len(clocks) == 4 and all(abs(c-s['prompt_ms']) <= .51 for c,s in zip(clocks,samples)), 'native/API clocks differ')
    fields = ('run','request_sha256','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms')
    compact_work=[dict(windows=int(w['windows']), maximum_rows=int(w['maximum_rows']), jobs=int(w['jobs']),
        dequants=int(w['dequants']),requested_router_ID_D2H_bytes=int(w['source_H2D_bytes']),
        requested_expert_H2D_bytes=int(b['requested_H2D_bytes'])) for w,b in zip(work,bank)]
    return [dict((k,s[k]) for k in fields) for s in samples], [78451000/s['prompt_ms'] for s in samples], compact_work


def run_one(row):
    index = len(row['processes'])
    require(row['stage'] in ('prepared','running') and index < 3 and all(p['passed'] for p in row['processes']), 'single-use ordered screen;stop on failure')
    try:
        ref, rt = current(row)
        review = prior.runtime.validation.committed(prior.runtime.validation.INDEX, (REVIEW,))
        require(review['status'] == 'reviewed_model_control_eligible' and review['model_trial_allowed'] is True
            and review['source_sha256'] == digest(__file__) and review['plan_identity'] == identity(row['plan']), 'exact source/plan review required')
        p = row['inputs'][index]; root = Path(p['root']); require(not (root/'execution.json').exists(), 'never repeat launch')
        cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
        host, manifest = prior.python_runtime.admitted(prior)
        allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    except BaseException as error:
        row['processes'].append(dict(index=index, complete=False, passed=False, error=repr(error),
            native_launched=False, ownership='not_launched', ownership_closed=True))
        row.update(stage='admission-failed', admission_error=repr(error), native_launched=False, active=False, ownership_closed=True)
        save(row); raise
    owner = None; outcome = dict(index=index, complete=False, host_probe_accepted=False, request_windows=[])
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def awake():
        now = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()); delta = [a-b for a,b in zip(now,initial)]
        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
        return delta[0]
    def persist(): prior.write(root/'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, leader):
        poll = io.poll
        def bounded_poll(): require(awake() < 2400000000000, 'absolute screen deadline'); poll()
        io.poll = bounded_poll
        for request in range(4):
            for stage in ('begin','end'):
                while True:
                    line = io.line(1200)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                if stage == 'begin': boundary = (time.process_time_ns(),awake())
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage,phase='prefill-only',run=request), 'request gate differs')
                require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size, 'parameter warning')
                require(prior.baseline.artifact_bytes(ROOT) <= row['plan']['whole_bytes'], 'whole byte budget')
                if stage == 'begin':
                    claim = json.loads((root/'engine-start.json').read_text()); pid=claim['native']['pid']; ident=prior.baseline.identity(pid)
                    require(ident and ident['start_ticks']==claim['native']['start_ticks'] and ident['sid']==leader['pid']
                        and str(Path('/proc',str(pid),'exe').resolve())==str(Path(ref['binary']['path']).resolve()), 'native identity differs')
                    expected=prior.engine_once.native_environment(dict(native_validation_probe=ref['probe']),p['environment'])
                    actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
                        set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native environment differs')
                    mapped=prior.runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                    require(mapped<=set(rt['runtime_files'])|{str(Path(ref['binary']['path']).resolve())} and
                        {ref['NEO'],ref['probe']['loader']['path'],ref['probe']['library']['path']}<=mapped, 'native provider differs')
                    wrapper=prior.baseline.identity(claim['wrapper']['pid']); require(wrapper and wrapper['sid']==leader['pid'], 'wrapper identity differs')
                    server=prior.baseline.identity(wrapper['ppid']); require(server and server['sid']==leader['pid'], 'server identity differs')
                    port=p['command'][p['command'].index('--port')+1]
                    prior.python_runtime.check_running(leader['pid'],P/'tools/strata_bench.py',manifest['selected'],p['environment'],allowed,p['command'][2:])
                    prior.python_runtime.check_running(server['pid'],P/'serve/server.py',manifest['selected'],p['environment'],allowed,
                        ['--engine','strata','--config',str(root/'config.json'),'--port',port])
                else:
                    cpu,elapsed=time.process_time_ns()-boundary[0],awake()-boundary[1]
                    outcome['request_windows'].append(dict(request=request,controller_cpu_ns=cpu,gate_elapsed_ns=elapsed,fraction=cpu/elapsed))
                    persist(); require(0<=cpu/elapsed<=row['plan']['cpu_gate_maximum'], 'controller CPU cap')
                    print(f'64K process{index}: request{request+1}/4 complete',flush=True)
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic()<deadline,'teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner=prior.baseline.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',p['command'],p['environment'],P,interaction=interact,**row['plan']['budgets'])
        require(prior.baseline.closed(entry) and entry['exit_code']==0, 'model owner failed: '+str(entry.get('error')))
        result_file,=(root/'bench').glob('*/results.json'); result=json.loads(result_file.read_text())
        samples=result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(outcome['request_windows'])==4, 'incomplete benchmark')
        sys.path.insert(0,str(P/'tools')); from strata_bench import verify_request_journal
        journal,=(root/'bench').glob('*/requests.jsonl'); verified=verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'invalid request journal')
        events=[json.loads(v) for v in journal.read_text().splitlines()]
        require([dict(v['sample'],run=v['run']) for v in events if v['kind']=='result' and v['phase']=='prefill-only']==samples, 'journal binding differs')
        end=prior.gen.native_end(root,dict(binary=ref['binary'],native_validation_probe=ref['probe']),dict(config=p['config']),ref['boot'],entry['pid'])
        text=(root/'engine.stderr').read_text(); probe=prior.runtime.validation.preload_evidence(text,end['native']['pid'])
        require(probe['loader']==ref['probe']['loader']['path'], 'validation loader differs')
        clocks,rates,work=parse_samples(row,samples,text)
        current(row); prior.models.model_pins(ref['configuration'],ref['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,requests=clocks,tok_s=rates,first_use_tok_s=rates[0],
            subsequent_median=statistics.median(rates[1:]),window_work=work,native_identity=end['native'],validation=probe)
    except BaseException as error: outcome['error']=repr(error)
    finally:
        if owner: prior.runtime.validation.finish(outcome,owner,cursor,ref['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome)
        row.update(active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',False),stage='running' if outcome['passed'] else 'failed')
        if len(row['processes'])==3 and all(p['passed'] for p in row['processes']):
            medians=[p['subsequent_median'] for p in row['processes']]
            row.update(stage='closed',complete=True,passed=True,analysis=dict(subsequent_process_medians=medians,
                median=statistics.median(medians),throughput_screen_passed=all(v>=1000 for v in medians),
                same_input_gain_established=False,inference_adoption=False))
        save(row)
    print(json.dumps(outcome),flush=True); return int(not outcome['passed'])


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); prior.runtime.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']: prepare(); return 0
        if sys.argv[1:]==['run-one']: return run_one(json.loads((ROOT/'record.json').read_text()))
        raise RuntimeError('use prepare or run-one;commit source,plan and exact review first')


if __name__=='__main__': sys.exit(main())
