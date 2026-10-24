#!/usr/bin/env python3
"""Main-owned official long-prefill comparison; no engine rebuild or adoption.

Both packages retain the original decode configuration. Decode preservation
requires its own subsequent long-generation/agent comparison, even if this passes.
Repository LICENSE; authored from this fork's own maintained measurement helpers.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import full_context_model as model
import generation_engine as gen

gate = model.gate
guards, owner_lib = gate.native.guards, gate.native.supervision
P, B = model.P, model.B
ROOT = B / 'phase-prefill-speed-v1'
RESULT = gen.A / 'phase-prefill-speed-current.json'
HOST = B / 'phase-prefill-speed-host-v1'
ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))
CONTROL = 'neo_readback_control_v1'
TEST = 'tools/sycl/test_phase_prefill_speed.py'
WORKLOAD = 'native_gdn_20261018'
require, digest, write = guards.require, model.digest, model.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan():
    return dict(question='Does the full-context-corrected package preserve/improve the original fast long-prefill route?',
        order=list(ORDER), pairs=[list(p) for p in PAIRS], processes=6, requests_per_process=4,
        execution='Official bench.sh --prefill-only --prefill-tokens98304 --think --runs4;one short warmup then four fresh117489-token requests per independent process. Original6e40/candidate590f/candidate590f/original6e40/original6e40/candidate590f. First long request remains separate from the three subsequent requests.',
        constants='Original canonical decode configuration:131072 resident INT8 context,1256 slots,8192 prefill,5 workers,pcie0,adapt0,spec4,suffix0,caches0,Q8 MTP,no token graph/no coupled draft. Identical arithmetic/placement/safety flags. Only the native executable differs;no new tuning or environment mutation.',
        observers='Remove NEO allocation logs,UR memory subscriber,stage/state/cache readbacks,census and timers. Keep the qualified native-only UR validation/retained-loader preload and warning logs in BOTH arms. Parameter validation is charged to both;this is not an observer-free speed claim or ordinary-teardown qualification.',
        decision='Use each process subsequent-three median,exact tokens/engine_ms. Preserve every sample. Candidate faster screen requires all three paired ratios>1 and geomean>=1.0005. Prefill no-decline screen requires every ratio>=1. Target screen requires every candidate process median>=1000. A failed screen remains failed;no threshold/order change or noise dismissal. None qualifies decode or adoption.',
        stopping='Stop first source/config/model/route/journal/owner/fault/budget failure. No retry,skipped/replaced process or speed-based early stopping. Commit each closed outcome before the next process. Full15 model hashes at preparation and after process5;identity/stat pins before/after each process.',
        next='Independently compare repeated thinking-on long generation for both observed MTP classes and upstream long/agent-session suite with the original decode implementation/configuration/1256 slots. Any decode impact is a separation defect;do not adopt while it remains. Physical262144/task/session/failure-owner gates remain distinct.',
        interpretation='Package comparison after capacity/state gates,not causal kernel attribution or measured hardware saturation. Existing whole-model limits remain;no new speed gain is claimed before the samples.',
        retention='Keep individual request clocks/counts/hashes,phase decisions,source/config/actual route/owner/fault evidence. Successful raw logs/JIT retire after committed replay;retain only the two existing native generations for this imminent prefill/decode comparison.')


def qualification():
    container = gen.committed(model.RESULT)
    control = container[CONTROL]
    require(control['passed'] and control['complete'] and not control['active'] and
        control['ownership_closed'] and control['full_model_hashes_matched'] == 15 and
        all(s['passed'] and s['complete'] and s['ownership_closed'] and s['fresh_faults'] == []
            for s in control['outcomes'].values()) and len(control['outcomes']) == 4,
        'closed four-process audit control required')
    candidate = gate.engine()
    original, proof = gate.native.source.baseline()
    components, state = container['prefill_layout_components_v4'], container['model_state_v4']
    require(components['passed'] and components['complete'] and components['ownership_closed'] and
        components['binary'] == state['binary'] == candidate['binary'] and
        state['passed'] and state['complete'] and not state['active'] and state['ownership_closed'] and
        set(state['outcomes']) == {'protocol', 'small', 'checkpoint', 'near-cap', 'restore'} and
        all(s['passed'] and s['complete'] and s['ownership_closed'] and s['fresh_faults'] == []
            for s in state['outcomes'].values()), 'same-native model state and component gates required')
    preload = gate.validation.preload_qualification()
    driver = control['driver_library']
    require(digest(driver['path']) == driver['sha256'], 'qualified actual driver changed')
    setup = model.setup()
    return setup, dict(binaries=[original['binary'], candidate['binary']],
        native_sources=[original['source_commit'], candidate['source_commit']], selected=proof,
        candidate_receipt_sha256=digest(gate.native.ENGINE_RECORD),
        components_sha256=fingerprint(components), state_sha256=fingerprint(state),
        control_sha256=fingerprint(control), preload=preload, driver=driver,
        reference=setup['reference'])


def environment(setup, root):
    old = setup['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in setup['environment'].items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'UR_ENABLE_LAYERS', 'STRATA_STATE_HASH',
        'STRATA_TRACE', 'STRATA_SNAPSHOT_VERIFY', 'STRATA_SYCL_CACHE_LOAN_AUDIT')),
        'canonical clean reference unexpectedly contains an observer')
    safety = gen.committed(gen.SETTINGS)['environment']
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
              'STRATA_SYCL_PREFILL_SAME_QUEUE', 'ZES_ENABLE_SYSMAN'):
        require(env[k] == safety[k], 'qualified safety setting differs: ' + k)
    env.update(TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    for k in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
        env[k] = 'level:warning;flush:warning;output:stderr'
    return env


def config(setup, root):
    result = json.loads(json.dumps(setup['config']).replace('${RUN_ROOT}', str(root)))
    require(result['exe'] == str(P/'tools/sycl/engine_once.py') and
        result['args'][result['args'].index('--expert-cache')+1] == '1256' and
        result['args'][result['args'].index('--max-context')+1] == '131072',
        'original decode wrapper/placement/context differs')
    return result


def current(r):
    setup, proof = qualification()
    require(proof == r['qualification'] and plan() == r['plan'] and
        fingerprint(gen.benchmark_pins(r['source_commit'])) == r['benchmark_fingerprint'] and
        digest(P/TEST) == r['host']['test_sha256'], 'phase comparison inputs changed')
    require(r['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
        r['driver'] == guards.driver_guard(), 'boot/device changed')
    for i, row in enumerate(r['plans']):
        root = ROOT/str(i)
        require(row['arm'] == ORDER[i] and row['root'] == str(root) and
            row['environment'] == environment(setup, root) and
            json.loads((root/'config.json').read_text()) == config(setup, root) and
            json.loads((root/'plan.json').read_text()) == dict(
                binary=proof['binaries'][ORDER[i]], native_validation_probe=proof['preload']) and
            all(digest(p) == h for p, h in row['inputs'].items()), 'generated phase inputs changed')
    model.model_pins(setup['config'], r['model_assets'])
    return setup


def request_rates(samples):
    require(len(samples) == 4, 'four long requests required')
    rates = []
    for i, s in enumerate(samples):
        require(s['run'] == i and all(s.get(k) is True for k in
            ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
            s['actual_prompt_tokens'] == s['actual_read_tokens'] == 117489 and
            s['actual_reused_tokens'] == 0 and type(s['generated_tokens']) is int and
            1 <= s['generated_tokens'] <= 24 and s['prompt_tps_source'] == 'engine' and
            type(s['prompt_ms']) in (int, float) and math.isfinite(s['prompt_ms']) and
            s['prompt_ms'] > 0, 'ineligible long-prefill sample')
        rates.append(117489000/s['prompt_ms'])
    return rates


def summarize(rows):
    require(len(rows) == 6 and [s['arm'] for s in rows] == list(ORDER) and
        all(s['passed'] and s['complete'] and s['ownership_closed'] and s['fresh_faults'] == []
            for s in rows), 'six successful independent processes required')
    require(len({(s['native_identity']['pid'], s['native_identity']['start_ticks']) for s in rows}) == 6,
        'native processes were reused')
    first, medians = [], []
    for row in rows:
        rates = request_rates(row['requests'])
        first.append(rates[0]); medians.append(statistics.median(rates[1:]))
    for a, b in PAIRS:
        require([s['request_sha256'] for s in rows[a]['requests']] ==
            [s['request_sha256'] for s in rows[b]['requests']], 'paired request changed')
    ratios = [medians[b]/medians[a] for a, b in PAIRS]
    geo = math.exp(statistics.mean(map(math.log, ratios)))
    return dict(first_use_tok_s=first, subsequent_median_tok_s=medians, pair_ratios=ratios,
        ratio_geomean=geo, prefill_no_decline_screen=all(x >= 1 for x in ratios),
        candidate_faster_screen=all(x > 1 for x in ratios) and geo >= 1.0005,
        candidate1000_each_process=all(medians[i] >= 1000 for i in (1, 2, 5)),
        decode_preservation_qualified=False, adoption_qualified=False)


def prepare():
    previous = gen.committed(RESULT)
    require(not ROOT.exists() and previous['stage'] == 'host-checked' and previous['plan'] == plan(),
        'fresh CPU-qualified single-use comparison only')
    setup, proof = qualification()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    require(digest(P/TEST) == pins[TEST], 'commit exact CPU controls first')
    host = previous['host']
    require(host['passed'] and host['tests'] == 5 and host['exit_code'] == 0 and
        host['controller_sha256'] == pins[str(Path(__file__).relative_to(P))] and
        host['test_sha256'] == pins[TEST], 'five current CPU evidence controls required')
    model.model_pins(setup['config'], setup['models'], full=True)
    ROOT.mkdir(mode=0o700)
    plans, ports = [], set()
    for i, arm in enumerate(ORDER):
        root = ROOT/str(i); root.mkdir()
        for name in ('tmp', 'cache'): (root/name).mkdir()
        (root/'expert-profile.bin').write_bytes((P/'data/expert-profile.bin').read_bytes())
        write(root/'config.json', config(setup, root))
        write(root/'plan.json', dict(binary=proof['binaries'][arm], native_validation_probe=proof['preload']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        require(port not in ports, 'duplicate bound port;no retry'); ports.add(port)
        plans.append(dict(arm=arm, root=str(root), environment=environment(setup, root),
            inputs={str(root/n): digest(root/n) for n in ('config.json', 'plan.json', 'expert-profile.bin')},
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '4',
                '--workload-id', WORKLOAD, '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')]))
    r = dict(stage='prepared', source_commit=revision, host=host, plan=plan(), qualification=proof,
        benchmark_fingerprint=fingerprint(pins), plans=plans, processes=[], model_assets=setup['models'],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        budgets=dict(wall=3000, cpu=72000, rss_cap=104<<30, text_cap=64<<20, file_cap=64<<20, total_cap=128<<20))
    current(r); write(ROOT/'frozen.json', r); write(RESULT, r)
    print('Prepared six-process phase-prefill comparison;commit before execution.', flush=True)


def host():
    require(not HOST.exists() and not RESULT.exists(), 'fresh CPU evidence-control gate only')
    revision = gen.git('rev-parse','HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    require(digest(P/TEST) == pins[TEST], 'commit exact CPU test source')
    HOST.mkdir(mode=0o700)
    row = dict(source_commit=revision, controller_sha256=pins[str(Path(__file__).relative_to(P))],
        test_sha256=pins[TEST], passed=False, tests=5, GPU_executed=False)
    owner = None
    def persist():
        write(HOST/'record.json',dict(row,commands=owner.commands if owner else []))
    owner = owner_lib.Owner(HOST/'owned',persist)
    try:
        entry,_,err = owner.run('host',['/usr/bin/python3','-B',TEST],
            dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'), P,
            wall=30,cpu=30,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=4<<20)
        row['exit_code'] = entry['exit_code']
        require(owner_lib.closed(entry) and row['exit_code'] == 0 and
            'Ran 5 tests' in err.read_text() and err.read_text().rstrip().endswith('OK'),
            'five CPU evidence controls failed')
        row['passed'] = True
    except BaseException as error: row['error'] = repr(error)
    finally:
        row['ownership_closed'] = owner.active is None and all(owner_lib.closed(e) for e in owner.commands)
        row['passed'] = row['passed'] and row['ownership_closed']; persist()
        write(RESULT,dict(stage='host-checked' if row['passed'] else 'host-failed',host=row,plan=plan()))
    print(json.dumps(row),flush=True)
    return 0 if row['passed'] else 1


def run():
    r = gen.committed(RESULT); setup = current(r)
    frozen = json.loads((ROOT/'frozen.json').read_text())
    require({k:v for k,v in r.items() if k not in ('stage','processes','analysis')} ==
        {k:v for k,v in frozen.items() if k not in ('stage','processes','analysis')} and
        r['stage'] in ('prepared','collecting') and len(r['processes']) < 6 and
        all(s['passed'] for s in r['processes']), 'closed/failed/changed cohort;no retry')
    i = len(r['processes']); selected = r['plans'][i]; root = Path(selected['root'])
    require(not (root/'record.json').exists(), 'existing attempt;inspect owner')
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    row = dict(arm=selected['arm'], passed=False, complete=False, requests=[]); owner = None
    def persist():
        write(root/'record.json', dict(row, active=bool(owner and owner.active is not None), commands=owner.commands if owner else []))
    owner = owner_lib.Owner(root/'owned', persist)
    try:
        entry, _, _ = owner.run('bench', selected['command'], selected['environment'], P, **r['budgets'])
        files = list((root/'bench').glob('*/results.json'))
        require(len(files) == 1, 'one official result required')
        result = json.loads(files[0].read_text())
        row['requests'] = result['results']['prefill_only']['requests']
        request_rates(row['requests'])
        require(owner_lib.closed(entry) and entry['exit_code'] == 0 and result['complete'] and
            not result['results']['failures'] and result['params']['mode'] == 'prefill-only' and
            result['params']['runs'] == 4 and result['params']['think'] is True, 'official benchmark incomplete')
        sys.path.insert(0, str(P/'tools'))
        from strata_bench import verify_request_journal
        journal, = (root/'bench').glob('*/requests.jsonl')
        verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'official journal incomplete')
        events = [json.loads(s) for s in journal.read_text().splitlines()]
        samples = [e['sample'] for e in events if e['kind'] == 'result']
        longs = [e for e in events if e['kind'] == 'result' and e['phase'] == 'prefill-only']
        require(len(samples) == 5 and len(longs) == 4 and
            [dict(e['sample'], run=e['run']) for e in longs] == row['requests'], 'official journal/request binding differs')
        native = gen.native_end(root, dict(binary=r['qualification']['binaries'][selected['arm']],
            native_validation_probe=r['qualification']['preload']), dict(config=config(setup,root)), r['boot'], entry['pid'])
        text = (root/'engine.stderr').read_text(errors='replace')
        activation = gate.validation.preload_evidence(text, native['native']['pid'])
        boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n', text, re.M)
        require(boundary and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', text[boundary.end():]), 'runtime error after validation canary')
        headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M))
        require(len(headers) == len(samples) and all(tuple(map(int,h.groups())) ==
            (s['prompt_tokens'],s['reused_tokens'],s['read_tokens'],s['generated_tokens']) for h,s in zip(headers,samples)), 'native/request extent differs')
        routes = []
        for j in range(1, 5):
            frame = text[headers[j-1].end():headers[j].start()]
            route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 ', frame, re.M)
            require(len(route) == 1 and tuple(map(int,route[0])) == (117488,117488*20480,117488,117488*20480) and
                'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0' in frame,
                'actual long GPU/BF16/mode10 route differs')
            routes.append(dict(rows=117488, full_GPU_BF16=True, MoE_mode=10))
        require(not any(s in text for s in ('strata trace:', 'strata loan audit:', 'strata decode timing:',
            'strata UR memory:', 'strata serve: STATE_HASH', 'strata serve: SNAPSHOT_VERIFY')), 'unexpected heavy observer')
        if r['processes']:
            require([s['request_sha256'] for s in row['requests']] ==
                [s['request_sha256'] for s in r['processes'][0]['requests']], 'fresh request bodies changed')
        current(r)
        if i == 5: model.model_pins(setup['config'],r['model_assets'],full=True)
        row.update(native_identity=native['native'], native_exit_code=0, validation=activation,
            route=routes, journal_sha256=digest(journal), host_probe_accepted=True, passed=True, complete=True)
    except BaseException as error:
        row['error'] = repr(error)
        if (root/'engine.stderr').exists(): row['error_excerpt'] = (root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        gate.validation.finish(row, owner, cursor, r['boot'])
        row['passed'] = bool(row['passed'] and row['complete']); persist()
        r['processes'].append(row); r['stage'] = 'collecting' if row['passed'] and i < 5 else 'closed'
        if row['passed'] and i == 5: r['analysis'] = summarize(r['processes'])
        write(RESULT,r)
    print(json.dumps({k:row.get(k) for k in ('arm','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__':
    require(sys.argv[1:] in (['host'], ['prepare'], ['run']), 'host|prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        sys.exit({'host':host,'prepare':prepare,'run':run}[sys.argv[1]]())
