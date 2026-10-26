#!/usr/bin/env python3
"""Main-owned matched supervisor-cost model control; repository LICENSE.

Original engine, private validation, safety settings and decode placement stay
fixed. Source review and the separate host performance gate precede execution.
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
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import cpu_capacity as models
import engine_once
import generation_engine as gen
import measurement_owner as baseline
import owner_candidate as candidate
import runtime_probe as runtime

B, A = gen.B, gen.A
ROOT = B / 'prefill-supervision-cost-v1'
HOST = B / 'owner-scan-control-v3'
RESULT = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
KEY = 'supervision_model_control_v1'
REVIEW = 'supervision_model_source_contract_20261027'
ORDER = ('baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate')
require, digest = runtime.require, runtime.digest


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def input_identity(row):
    return identity({k: row[k] for k in ('binary', 'qualification', 'configuration',
        'template_sha256', 'runtime_sha256', 'environment_template', 'probe', 'NEO',
        'boot', 'request_hashes', 'model_assets')})


def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'model_assets', 'environment_template', 'process_plans')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    literal = json.dumps(KEY) + ':'; text = json.dumps(compact(row), indent=2).replace('\n', '\n    ')
    if KEY in desired['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous model supervisor scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + text + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'; require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + text + ',', 1)
    desired['historical_regression_audit'][KEY] = compact(row)
    require(json.loads(raw) == desired, 'model update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def qualification():
    rt = runtime.validation.committed(runtime.RESULT); full = runtime.current(rt)
    host = runtime.validation.committed(RESULT, ('historical_regression_audit', 'owner_scan_control_v3'))
    require(host == json.loads((HOST / 'record.json').read_text()) and host['stage'] == 'closed' and
        host['complete'] and host['passed'] and host['ownership_closed'] and
        host['tests']['passed'] and host['tests']['cases'] == 14 and len(host['processes']) == 6 and
        all(v['passed'] and v['ownership_closed'] for v in host['processes']) and
        host['analysis']['host_gate_passed'], 'closed exact host correctness/performance gate required')
    for name, sha in host['plan']['sources'].items():
        require(digest(P / name) == sha and hashlib.sha256(gen.git('show', host['plan']['source_commit'] + ':' + name)).hexdigest() == sha,
            'host-qualified source changed: ' + name)
    require(digest(HOST / 'providers.json') == host['plan']['providers_sha256'] and
        all(digest(p) == sha for p, sha in json.loads((HOST / 'providers.json').read_text()).items()) and
        str(Path(sys.executable).resolve()) == host['plan']['python'] and
        sys.flags.isolated == sys.flags.no_site == sys.flags.dont_write_bytecode == 1 and sys.flags.optimize == 0,
        'qualified host Python/providers or flags changed')
    return full, rt, dict(host_gate=identity(host), candidate_sha256=digest(candidate.__file__),
        baseline_sha256=digest(baseline.__file__), providers_sha256=host['plan']['providers_sha256'])


def reviewed(row):
    audit = runtime.validation.committed(runtime.validation.INDEX, (REVIEW,))
    require(audit['status'] == 'reviewed_model_control_eligible' and audit['model_trial_allowed'] is True and
        audit['sources'] == {str(Path(p).relative_to(P)): digest(p) for p in
            (Path(__file__).resolve(), Path(candidate.__file__).resolve(), Path(baseline.__file__).resolve())} and
        audit['model_plan_identity'] == identity(row['plan']), 'exact model controller/source review required')


def environment(row, root):
    template = row['environment_template']; old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in template.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_TRACE', 'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY',
        'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE', 'STRATA_CPU_WORK_CENSUS', 'ZE_ENABLE_ALT_DRIVERS')), 'unexpected observer/provider')
    env.update(UR_ENABLE_LAYERS='', ZE_ENABLE_ALT_DRIVERS=row['NEO'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    return env


def current(row):
    committed = runtime.validation.committed(RESULT, ('historical_regression_audit', KEY, 'plan'))
    require(committed == row['plan'], 'commit exact immutable model plan')
    require(input_identity(row) == row['plan']['qualified_input_identity'] and
        row['environment_template'] == gen.committed(TEMPLATE)['environment_template'],
        'qualified model inputs differ from immutable plan')
    full, rt, proof = qualification()
    require(proof == row['qualification'] and full['binary'] == row['binary'] and
        gen.benchmark_pins(row['plan']['source_commit']) == row['controller_inputs'] and
        digest(TEMPLATE) == row['template_sha256'] and digest(runtime.RESULT) == row['runtime_sha256'], 'model source/native/runtime changed')
    require(row['plan']['process_plans'] == [dict(index=p['index'], arm=p['arm'], root=p['root'], inputs=p['inputs'],
        command=p['command']) for p in row['process_plans']], 'immutable process plans changed')
    for p in row['process_plans']:
        root = Path(p['root'])
        require(environment(row, root) == p['environment'] and
            all(digest(root / name) == sha for name, sha in p['inputs'].items()), 'generated input/environment changed')
        require(json.loads((root / 'config.json').read_text()) == p['config'], 'native configuration changed')
        require(json.loads((root / 'plan.json').read_text()) == dict(binary=full['binary'], native_validation_probe=row['probe']), 'native plan changed')
    models.model_pins(row['configuration'], row['model_assets'])
    return rt


def prepare():
    require(not ROOT.exists() and KEY not in gen.committed(RESULT)['historical_regression_audit'], 'single-use model control')
    full, rt, proof = qualification()
    reference = gen.committed(RESULT)['historical_regression_audit']['near_dense_original_reference_v1']
    require(reference['passed'] and reference['ownership_closed'] and reference['binary'] == full['binary'], 'closed same-native reference')
    cfg = copy.deepcopy(reference['configuration'])
    assets = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    pins = models.model_pins(cfg, assets, full=True)
    loader = rt['preload']['loader']; probe = dict(library=rt['preload']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]))
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', binary=full['binary'], qualification=proof, configuration=cfg, model_assets=pins,
        controller_inputs=gen.benchmark_pins(source), template_sha256=digest(TEMPLATE), runtime_sha256=digest(runtime.RESULT),
        environment_template=gen.committed(TEMPLATE)['environment_template'], probe=probe,
        NEO=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p), boot=rt['boot'], request_hashes=reference['request_hashes'],
        process_plans=[], processes=[], complete=False, passed=False, active=False, ownership_closed=True,
        plan=dict(source_commit=source, order=list(ORDER), process_plans=[],
            question='Does lower-cost equivalent host supervision improve original6e40 whole-prefill timing?',
            work='Six fresh processes,baseline/candidate/candidate/baseline/baseline/candidate;four117489fresh/zero-reuse/thinking-on/24output requests each. Original22182f1a engine,1256decode slots,131072INT8 resident context,8192chunks,BF16/vLLM/mode10,384bank,original preload and immutable qualified safety settings. No profiler,clock sampler,inference or placement change.',
            timing='First use separate;each process contributes median of subsequent3. All3 paired candidate/baseline ratios and geometric mean retained. Actual controller CPU/gate elapsed measured once/request,including existing resource,artifact and protocol work.',
            CPU_budget='Host-only candidate gate<=5percent has passed. The full model control separately permits<=8percent candidate and<=20percent legacy controller CPU/gate elapsed to cover model-artifact/protocol/provider handling. Persist operands before checking. This does not repair the original failed5percent clock-observer diagnostic;no clock observer is used here.',
            acceptance='All source/provider/request/stream/journal/route/validation/fault/closure gates pass. Mechanism speed screen:all3pair ratios>1 and geometric mean>1.01. Historical1003and1006 comparisons descriptive only. No speed-based early stop/retries;stop first execution/contract/budget failure. No inference/default adoption from this control.',
            limits='24token outputs do not qualify sustained decode/70tok_s,quality/agent/physical262144/handoff or new engine. Parameter validation and existing launch boundaries remain charged;source review and host gate precede execution. BOOTTIME/MONOTONIC divergence>50ms rejects suspend.',
            budgets=dict(wall=3000, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20, file_cap=64 << 20, total_cap=128 << 20),
            whole_bytes=768 << 20,
            retention='Keep24individual clock/work/CPU samples,source/native/runtime/config/recipe and decisions. Retire normal logs/JIT/closed controls after replay;no new engine build. Retain qualified host record/provider manifest only through this comparison.'))
    ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ORDER):
        root = ROOT / str(index); root.mkdir()
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(cfg); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config); write(root / 'plan.json', dict(binary=full['binary'], native_validation_probe=probe))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        process = dict(index=index, arm=arm, root=str(root), config=config, environment=environment(row, root),
            inputs={name: digest(root / name) for name in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '4', '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')])
        row['process_plans'].append(process)
    row['plan']['process_plans'] = [dict(index=p['index'], arm=p['arm'], root=p['root'], inputs=p['inputs'], command=p['command']) for p in row['process_plans']]
    row['plan']['qualified_input_identity'] = input_identity(row)
    save(row); print('Prepared six-process matched model plan;commit and source-review before GPU execution.', flush=True)


def run_one(row):
    try:
        rt = current(row); reviewed(row)
    except BaseException as error:
        row.update(stage='admission-failed', admission_error=repr(error), native_launched=False)
        save(row); raise
    index = len(row['processes'])
    require(row['stage'] in ('prepared', 'running') and index < 6 and all(p['passed'] for p in row['processes']), 'ordered single-use process;stop on failure')
    process = row['process_plans'][index]; root = Path(process['root']); module = baseline if process['arm'] == 'baseline' else candidate
    require(not (root / 'execution.json').exists(), 'never repeat model launch')
    start = next(v.split(': ', 1)[1] for v in subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    owner = None; outcome = dict(complete=False, host_probe_accepted=False, request_windows=[])
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()); boundary = None
    def awake():
        now = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()); delta = [a-b for a,b in zip(now, initial)]
        require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity'); return delta[0]
    def persist(): write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, leader):
        nonlocal boundary
        poll = io.poll
        def checked_poll(): require(awake() < 3000000000000, 'absolute model deadline'); poll()
        io.poll = checked_poll
        for request in range(4):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
                require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size, 'parameter warning')
                if stage == 'begin':
                    claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']; ident = baseline.identity(pid)
                    require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                        str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
                    expected = engine_once.native_environment(dict(native_validation_probe=row['probe']), process['environment'])
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                    mapped = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                        {row['NEO'], row['probe']['loader']['path'], row['probe']['library']['path']} <= mapped, 'actual provider differs')
                    boundary = (time.process_time_ns(), awake())
                else:
                    cpu, elapsed = time.process_time_ns() - boundary[0], awake() - boundary[1]
                    window = dict(request=request, controller_cpu_ns=cpu, gate_elapsed_ns=elapsed, fraction=cpu / elapsed,
                        maximum=.20 if process['arm'] == 'baseline' else .08)
                    outcome['request_windows'].append(window); persist()
                    require(0 <= window['fraction'] <= window['maximum'], 'request controller CPU budget: ' + json.dumps(window))
                    print(f"Process{index}/{process['arm']}:long request{request + 1}/4 complete", flush=True)
                require(baseline.artifact_bytes(ROOT) <= row['plan']['whole_bytes'], 'whole model byte budget')
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'model teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = module.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', process['command'], process['environment'], P, interaction=interact, **row['plan']['budgets'])
        require(module.closed(entry) and entry['exit_code'] == 0, 'model owner failed: ' + str(entry.get('error')))
        result_file, = (root / 'bench').glob('*/results.json'); result = json.loads(result_file.read_text()); samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(outcome['request_windows']) == 4, 'incomplete official benchmark')
        for s, expected in zip(samples, row['request_hashes']):
            require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and s['reasoning_observed'] and
                s['request_sha256'] == expected and s['actual_prompt_tokens'] == s['actual_read_tokens'] == 117489 and
                s['actual_reused_tokens'] == 0 and s['generated_tokens'] == 24 and s['prompt_tps_source'] == 'engine' and
                10000 <= s['prompt_ms'] < 1500000, 'workload/clock eligibility differs')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (root / 'bench').glob('*/requests.jsonl'); verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'request journal invalid')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        require([dict(v['sample'], run=v['run']) for v in events if v['kind'] == 'result' and v['phase'] == 'prefill-only'] == samples, 'journal/request binding differs')
        end = gen.native_end(root, dict(binary=row['binary'], native_validation_probe=row['probe']), dict(config=process['config']), row['boot'], entry['pid'])
        text = (root / 'engine.stderr').read_text(); probe = runtime.validation.preload_evidence(text, end['native']['pid'])
        require(probe['loader'] == row['probe']['loader']['path'] and
            text.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16') == 4 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 4 and
            not any(v in text for v in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'route/runtime differs')
        clocks = list(map(int, re.findall(r'^strata serve: prompt 117489 tokens = 0 reused \+ 117489 read in (\d+) ms ', text, re.M)))
        require(len(clocks) == 4 and all(abs(clock - s['prompt_ms']) <= .51 for clock,s in zip(clocks,samples)), 'native/API clock binding differs')
        current(row); models.model_pins(row['configuration'], row['model_assets'], full=True)
        fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens', 'generated_tokens', 'prompt_ms')
        rates = [117489000 / s['prompt_ms'] for s in samples]
        outcome.update(complete=True, host_probe_accepted=True, index=index, arm=process['arm'],
            requests=[{k:s[k] for k in fields} for s in samples], tok_s=rates, first_use_tok_s=rates[0],
            subsequent_median=statistics.median(rates[1:]), native_identity=end['native'], validation=probe)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: runtime.validation.finish(outcome, owner, start, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome)
        row.update(active=bool(owner and owner.active), ownership_closed=outcome.get('ownership_closed', False))
        row['stage'] = 'running' if outcome['passed'] else 'failed'
        if len(row['processes']) == 6 and all(p['passed'] for p in row['processes']):
            pairs = []
            for offset in (0, 2, 4):
                group = {p['arm']:p['subsequent_median'] for p in row['processes'][offset:offset+2]}; pairs.append(group['candidate'] / group['baseline'])
            ratio = statistics.geometric_mean(pairs)
            row.update(stage='closed', complete=True, passed=True, analysis=dict(candidate_over_baseline=pairs, geometric_mean=ratio,
                mechanism_speed_screen_passed=all(v > 1 for v in pairs) and ratio > 1.01,
                inference_adoption=False, historical_cause_established=False))
        save(row)
    print(json.dumps(outcome), flush=True); return int(not outcome['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); runtime.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run-one']: return run_one(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run-one;commit plan and exact source review before GPU')


if __name__ == '__main__': sys.exit(main())
