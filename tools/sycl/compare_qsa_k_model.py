#!/usr/bin/env python3
"""Own a source-frozen, six-process direct-K prefill comparison.

Only the separately qualified native executable changes. Keep validation and
loader-lifetime safety in both arms; remove the diagnostic's STRATA_TRACE.
This measures long prefill, not decode/task/full-context adoption.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import measure_qsa_k_model as diagnostic
import engine_once

m = diagnostic.m
P, B, RESULT = diagnostic.P, diagnostic.B, diagnostic.RESULT
ROOT = B / 'qsa-direct-k-model-comparison-v1'
require, digest, write = diagnostic.require, diagnostic.digest, diagnostic.write
ORDER = ('original', 'candidate', 'candidate', 'original', 'original', 'candidate')
KEY = 'clean_prefill_comparison'
MINIMUM, RUNS = 98304, 4


def qualification():
    record, built, rt = diagnostic.native()
    host, manifest = diagnostic.python_contract()
    model = record[m.KEY]['model_validation']['model']
    private = json.loads((diagnostic.ROOT / 'model/record.json').read_text())
    require(model == {k:v for k,v in private.items() if k not in ('sources','model_assets')} and
            model['stage'] == 'closed' and model['passed'] and model['ownership_closed'] and
            model['outcome']['complete'] and model['outcome']['fresh_faults'] == [] and
            not model['active'], 'closed first production diagnostic required')
    require(set(record[m.KEY]['link_contract']['outputs']) ==
            {'strata','sycl_submission_test','sycl_graph_envelope_test','fixture','numerical'},
            'exact five-binary contract required')
    return record, built, rt, manifest, private


def environment(root, rt):
    env = diagnostic.base_environment(root)
    env.pop('STRATA_TRACE', None)
    env['ZE_ENABLE_ALT_DRIVERS'] = next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p)
    require(all(v == '0' for k,v in env.items() if '_PROFILE' in k or '_AUDIT' in k),
            'heavy observer enabled in clean comparison')
    return env


def prepare():
    record, built, rt, _, model = qualification()
    require(KEY not in record[m.KEY] and not ROOT.exists(), 'fresh comparison only')
    revision = m.component.gen.git('rev-parse','HEAD').decode().strip()
    pins = diagnostic.pins(revision)
    assets = diagnostic.models.model_pins(model['configuration'], model['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    probe = model['native_validation_probe']
    binaries = {'original':rt['binary'], 'candidate':built['binaries']['strata']}
    plans = []
    for index, arm in enumerate(ORDER):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root / name).mkdir()
        config = copy.deepcopy(model['configuration']); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=binaries[arm], native_validation_probe=probe))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        command = ['/bin/sh',str(P / 'bench.sh'),'--config',str(root / 'config.json'),'--port',str(port),
            '--prefill-only','--prefill-tokens',str(MINIMUM),'--think','--runs',str(RUNS),
            '--profile-request-gates','--workload-id','native_gdn_20261018',
            '--fail-fast-engine-error','--ready-timeout','600','--out',str(root / 'bench')]
        plans.append(dict(index=index,arm=arm,root=str(root),binary=binaries[arm],command=command,
            generated={n:digest(root / n) for n in ('plan.json','config.json')},
            environment_sha256=diagnostic.fingerprint(environment(root,rt))))
    suite = diagnostic.client.Suite('', '',131072,RUNS,True,prefill_only=True,
        prefill_tokens=MINIMUM,workload_id='native_gdn_20261018')
    # Bind the wire JSON,including its actual key order and ASCII encoding.
    hashes = [hashlib.sha256(json.dumps(diagnostic.client.chat_body(suite.prefill_prompt(i),24,True)).encode()).hexdigest()
              for i in range(RUNS)]
    row = dict(stage='prepared',source_commit=revision,sources=pins,model_assets=assets,
        runtime_sha256=digest(m.runtime.RESULT),boot=rt['boot'],profile=model['profile'],
        native_validation_probe=probe,plans=plans,request_sha256=hashes,processes=[],
        budgets=dict(wall=1800,cpu=5400,rss_cap=104 << 30,text_cap=64 << 20,
                     file_cap=128 << 20,total_cap=512 << 20),
        plan=dict(order=list(ORDER),runs_per_process=RUNS,
            question='Does the independently qualified direct-K row-address hoist improve whole long prefill with unchanged decode configuration?',
            mechanism='Actual QSA component wall/service improves about4percent at6491queries;QSA service26.94s is not all critical-path wall. Expect only a small whole-model gain,not4percent or a hardware-saturation claim.',
            constants='131072 INT8 resident context,1256 expert slots,8192 chunks,BF16/vLLM/mode10,5workers,spec4,identical models/profile/requests and safety. Native parameter-validation/retained-loader policy stays enabled in both arms;STRATA_TRACE and profiling remain off.',
            samples='Six independent fresh processes in three adjacent opposite-order pairs. Each has one short warmup and four >=98304fresh/zero-reuse requests. Keep first long use separately;compare median of subsequent three within each process. Keep every sample,PP,TTFT and ranges. Not cold disk;source/model hashing primes file cache.',
            acceptance='All six processes and24requests must qualify. Candidate/original subsequent-time ratios in all three adjacent pairs must be below1;geometric mean must be <=0.995 for this small-gain screen. First-use ratios are descriptive. Otherwise inconclusive/rejected,not noise dismissal or adoption. No throughput-target claim from one point.',
            stopping='Stop first correctness,source,runtime,ownership,validation,fault or budget failure. No performance-based stop/retry/replacement/exclusion. Finish all six before decision;no criteria changes after readings.',
            limits='Prefill mechanism screen only. Output24 is not decode70,task quality,agent-session,long generation or physical262144 qualification. API prefill time includes handoff back to unchanged decode layout. Independent decode preservation remains mandatory before selection.',
            retention='Keep compact individual request samples,source/recipe/environment/qualification identity and decision. Retire raw logs and duplicate outputs after verification;retain fallback and candidate only until comparison/adoption decision.'),
        complete=False,passed=False,active=False,ownership_closed=True)
    write(ROOT / 'record.json',row)
    record[m.KEY][KEY] = compact(row); write(RESULT,record)
    print('Prepared six-process long-prefill comparison;commit before execution.',flush=True)


def compact(row):
    return {k:v for k,v in row.items() if k not in ('sources','model_assets')}


def run(index):
    record, _, rt, manifest, _ = qualification()
    row = json.loads((ROOT / 'record.json').read_text())
    require(record[m.KEY][KEY] == compact(row) and len(row['processes']) == index and
            all(v['passed'] for v in row['processes']) and index < len(ORDER),
            'committed ordered comparison required;no restart or skipped process')
    require(diagnostic.pins(row['source_commit']) == row['sources'] and
            digest(m.runtime.RESULT) == row['runtime_sha256'], 'comparison source/runtime changed')
    diagnostic.models.model_pins(json.loads((ROOT / str(index) / 'config.json').read_text()),row['model_assets'])
    profile = row['profile']
    require(diagnostic.models.file_identity(profile['path']) == profile['identity'] and
            digest(profile['path']) == profile['sha256'], 'expert placement profile changed')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(root,rt)
    require(not (root / 'execution.json').exists() and plan['arm'] == ORDER[index] and
            diagnostic.fingerprint(env) == plan['environment_sha256'] and
            all(digest(root / n) == sha for n,sha in plan['generated'].items()), 'process plan changed')
    require(digest(plan['binary']['path']) == plan['binary']['sha256'], 'actual comparison executable changed')
    m.runtime.guards.no_model_owners([])
    cursor = re.findall(r'^-- cursor: (.+)$',subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15),re.M)
    require(len(cursor) == 1,'fresh fault cursor required')
    outcome = dict(index=index,arm=plan['arm'],complete=False,host_probe_accepted=False,passed=False)
    owner = None
    def persist():write(root / 'execution.json',dict(outcome,active=bool(owner and owner.active),
        commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def awake():
        delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000,'suspend/clock discontinuity')
    def interact(io,entry):
        identity = None
        for run_index in range(RUNS):
            for stage in ('begin','end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage,phase='prefill-only',run=run_index),'unexpected comparison gate')
                awake()
                if stage == 'begin':
                    claim = json.loads((root / 'engine-start.json').read_text()); actual = claim['native']
                    pid = actual['pid']; live = m.runtime.supervision.identity(pid)
                    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
                            str(Path('/proc',str(pid),'exe').resolve()) == str(Path(plan['binary']['path']).resolve()) and
                            (identity is None or identity == actual),'owned native identity changed')
                    identity = actual
                    expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']),env)
                    actual_env = dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual_env.get(k.encode()) == v.encode() for k,v in expected.items()) and
                            set(actual_env) - {k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'},
                            'native pre-exec environment differs')
                    mapped = m.runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())} and
                            {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
                             env['ZE_ENABLE_ALT_DRIVERS']} <= mapped,'native provider mapping differs')
                    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
                    diagnostic.python_runtime.check_running(entry['pid'],P / 'tools/strata_bench.py',
                        manifest['selected'],env,allowed,plan['command'][2:])
                require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                        'parameter-validation warning/error')
                print(f'Process{index} {plan["arm"]} request{run_index} {stage}',flush=True);io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic() < deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner = m.runtime.supervision.Owner(root / 'owned',persist)
        entry,_,_ = owner.run('bench',plan['command'],env,P,interaction=interact,**row['budgets'])
        require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0,'comparison process failed')
        files = list((root / 'bench').glob('*/results.json'));require(len(files) == 1,'missing/duplicate official result')
        result = json.loads(files[0].read_text());samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == RUNS,
                'incomplete official comparison')
        for i,sample in enumerate(samples):
            require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                    sample['thinking_requested'] and sample['reasoning_observed'] and
                    sample['request_sha256'] == row['request_sha256'][i] and
                    sample['actual_prompt_tokens'] == sample['actual_read_tokens'] >= MINIMUM and
                    sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
                    sample['prompt_tps_source'] == 'engine' and sample['prompt_ms'] > 10000,
                    'comparison workload/clock differs')
            if row['processes']:
                previous = row['processes'][0]['samples'][i]
                require(all(sample[k] == previous[k] for k in
                            ('actual_prompt_tokens','actual_read_tokens','actual_reused_tokens')),
                        'actual workload differs between independent processes')
        require(diagnostic.client.verify_request_journal(files[0].parent)['valid'],'request journal invalid')
        require(diagnostic.pins(row['source_commit']) == row['sources'] and
                digest(m.runtime.RESULT) == row['runtime_sha256'],'source/runtime changed during comparison')
        config = json.loads((root / 'config.json').read_text())
        end = m.component.gen.native_end(root,dict(binary=plan['binary'],native_validation_probe=row['native_validation_probe']),
            dict(config=config),row['boot'],entry['pid'])
        log = (root / 'engine.stderr').read_text()
        proof = m.runtime.validation.preload_evidence(log,end['native']['pid'])
        require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                'strata MoE window configuration: requested_mode=10 effective_mode=10' in log and
                all('strata prefill: layer-major candidate, tokens=' + str(s['actual_read_tokens'] - 1) +
                    ' chunk=8192 residual_dtype=bf16' in log for s in samples) and
                not any(t in log for t in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),
                'actual route/runtime differs')
        # Keep every useful request sample,without full generated text/provider arrays.
        fields = ('request_sha256','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens',
            'generated_tokens','prompt_ms','prompt_tps','prompt_tps_source','ttft_s','total_s',
            'thinking_requested','reasoning_observed','ok','eligible','stream_complete')
        outcome.update(complete=True,host_probe_accepted=True,native_exit_code=0,
            samples=[{k:s[k] for k in fields if k in s} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]),validation=proof)
        if index == len(ORDER)-1:diagnostic.models.model_pins(config,row['model_assets'],full=True)
    except BaseException as error:outcome['error'] = repr(error)
    finally:
        if owner:m.runtime.validation.finish(outcome,owner,cursor[0],row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']);persist()
        row['processes'].append(outcome);row.update(stage='running' if outcome['passed'] else 'failed',
            ownership_closed=outcome.get('ownership_closed',False),active=bool(owner and owner.active))
        if outcome['passed'] and len(row['processes']) == len(ORDER):
            pairs = []
            for offset in (0,2,4):
                pair = {p['arm']:p for p in row['processes'][offset:offset+2]}
                pairs.append(pair['candidate']['subsequent_median_ms'] / pair['original']['subsequent_median_ms'])
            ratio = math.exp(statistics.mean(math.log(v) for v in pairs))
            row.update(stage='closed',complete=True,passed=True,decision=dict(pair_time_ratios=pairs,
                geometric_mean_time_ratio=ratio,small_gain_screen_passed=all(v < 1 for v in pairs) and ratio <= .995,
                inference_adopted=False,decode_qualified=False,physical262144_qualified=False))
        write(ROOT / 'record.json',row);record[m.KEY][KEY] = compact(row);write(RESULT,record)
    print(json.dumps({k:outcome.get(k) for k in ('index','arm','passed','complete','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'long-prefill comparison failed')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else run(int(sys.argv[2]))
