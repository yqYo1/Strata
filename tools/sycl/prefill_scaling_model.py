#!/usr/bin/env python3
"""One owned input-length characterization on the selected original engine.

Repository LICENSE. Main adaptation of our b8c31f1ae reference controller;same owned qualification/runtime checks.
No engine rebuild, inference selection, runtime change or speed-based retry.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as models
import generation_engine as gen
import measurement_owner as owners
import runtime_probe as runtime

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'prefill-length-scaling-v1'
RESULT = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
KEY = 'length_characterization'
require, digest, write = runtime.require, runtime.digest, runtime.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('benchmark_pins', 'models', 'environment_template')}


def save(row):
    container = json.loads(RESULT.read_text())
    container[KEY] = compact(row)
    write(RESULT, container)
    write(ROOT / 'record.json', row)


def environment(row, root, trace=False):
    base = row['environment_template']; old = base['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in base.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_TRACE', 'STRATA_STATE_HASH',
        'STRATA_SNAPSHOT_VERIFY', 'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE',
        'STRATA_CPU_WORK_CENSUS', 'ZE_ENABLE_ALT_DRIVERS')), 'unexpected template observer/provider')
    env.update(ZE_ENABLE_ALT_DRIVERS=row['neo']['path'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    if trace:
        env.update(LD_PRELOAD=row['preload']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'))
    return env


def current(row):
    require(gen.committed(RESULT)[KEY] == compact(row),
        'commit exact reference plan/outcome')
    require(gen.benchmark_pins(row['source_commit']) == row['benchmark_pins'], 'benchmark/controller changed')
    require(source_review() == row['source_contract'], 'reviewed scaling source contract changed')
    for path, sha in row['qualification_inputs'].items():
        require(digest(path) == sha, 'qualification input changed: ' + path)
    rt = gen.committed(runtime.RESULT); full = runtime.current(rt)
    require(full['binary'] == row['binary'] and rt['original_build'] == row['build_proof'] and
        rt['passed'] and rt['ownership_closed'] and not rt['active'], 'selected qualification changed')
    require(row['environment_template'] == gen.committed(TEMPLATE)['environment_template'],
        'reference environment changed')
    for name in ('config.json', 'plan.json'):
        require(digest(ROOT / name) == row['generated_sha256'][name], 'generated native input changed')
    require(json.loads((ROOT / 'plan.json').read_text()) ==
        dict(binary=row['binary'], native_validation_probe=row['preload']), 'native plan differs')
    models.model_pins(row['configuration'], row['models'])
    return rt


def prepare():
    previous = gen.committed(RESULT)
    require(not ROOT.exists() and previous[KEY]['stage'] == 'client-tested_model-execution-pending',
        'fresh single-use scaling root and tested client required')
    prior = previous['historical_regression_audit']['near_dense_original_reference_v1']
    require(prior['stage'] == 'closed' and prior['passed'] and prior['ownership_closed'] and not prior['active'],
        'closed successful selected-engine reference required')
    reviewed = source_review()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    full, proof = runtime.selected.baseline()
    rt = gen.committed(runtime.RESULT); runtime.current(rt)
    cfg = copy.deepcopy(prior['configuration']); cfg['log'] = str(ROOT / 'engine.stderr')
    expected = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    pinned_models = models.model_pins(cfg, expected, full=True)
    preload = rt['preload']; preload = dict(library=preload['library'], loader=dict(
        path=preload['loader'], sha256=rt['runtime_files'][preload['loader']]))
    neo = '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758'
    require(digest(P / 'data/expert-profile.bin') ==
        '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf', 'expert profile changed')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    row = dict(stage='prepared', source_commit=revision, benchmark_pins=gen.benchmark_pins(revision), source_contract=reviewed,
        qualification_inputs={str(p): digest(p) for p in (TEMPLATE, runtime.RESULT, P / 'data/expert-profile.bin')},
        environment_template=gen.committed(TEMPLATE)['environment_template'], configuration=cfg,
        binary=full['binary'], build_proof=proof, preload=preload,
        neo=dict(path=neo, sha256=rt['runtime_files'][neo]), boot=rt['boot'], driver=rt['driver'],
        models=pinned_models,
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-scaling', '--think', '--runs', '3', '--profile-request-gates',
            '--workload-id', 'prefill_scaling_v1', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(ROOT / 'bench')],
        plan=dict(question='How do total prefill time,pp and thinking-inclusive TTFT scale with actual input length on the selected6e40 engine?',
            repetition='One fresh server;one short warmup;one maximum-length first-use request retained separately;'
                'three warmed six-length sweeps in frozen ascending,descending,rotated order. Not three independent processes.',
            minima=[1024,4096,16384,32768,65536,98304],
            constants='Original22182f1a ELF;1256-slot decode placement;131072 resident INT8 KV;8192 chunks;'
                'BF16/vLLM/mode10 on long route;unchanged86-key template plus explicit qualified NEO;'
                'native-only retained-loader validation;24 output cap;thinking on;zero reuse.',
            prerequisites='Fresh host-only canary and bounded traced submission fixture;original13 artifacts '
                'and9853 Git inputs;qualified current driver/runtime;actual native environment/maps at19request boundaries;no fresh fault.',
            decision='No inference/runtime adoption. Keep every individual T,actualread,pp,TTFT,HTTPtotal and output count. '
                'Compare simple affine,quadratic,nlogn hypotheses by residuals and held-out-length errors;no automatic '
                'selection or measured-range extrapolation. Check actual route before interpreting fixed costs;'
                'short/decode-placement versus long/layer-major fits separate. Follow boundary points onlyif evidence requires.',
            stopping='No speed-based stop,retry,replaced or dropped sample. Stop first correctness,ownership,validation,budget,source or fault failure.',
            scope='No profiling,census,resource polling,state readback,rebuild,service/reset/global changes. '
                'Existing prefill API clock includes handoff back to decode. First nonempty reasoning/content defines TTFT. '
                'Output24 is not tg/agent/quality/physical262144 qualification. Same-process repeatability only.'),
        budgets=dict(wall=3000, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20,
            file_cap=64 << 20, total_cap=128 << 20),
        active=False, ownership_closed=True, qualification=[], complete=False, passed=False,
        retention='One compact curve withindividual counts/timings/TTFT,source/config/runtime identity,route andformula checks. '
            'Retire successful logs/journals/JIT and this one-use controller after verified extraction;original13 assets remain.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    write(ROOT / 'config.json', cfg)
    write(ROOT / 'plan.json', dict(binary=row['binary'], native_validation_probe=preload))
    row['generated_sha256'] = {name: digest(ROOT / name) for name in ('config.json', 'plan.json')}
    save(row)
    print('Prepared length characterization;commit the frozen plan before qualification/execution.', flush=True)


def cursor():
    text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    return next(s.split(': ', 1)[1] for s in text.splitlines() if s.startswith('-- cursor: '))


def check_maps(paths, allowed, neo, device=True):
    require(paths <= set(allowed), 'unplanned mapped executable: ' + repr(paths - set(allowed)))
    drivers = {p for p in paths if '/libze_intel_gpu.so.' in p}
    require(drivers == {neo} if device else not drivers or drivers == {neo}, 'NEO provider differs')


def qualify(row):
    rt = current(row); require(row['stage'] == 'prepared', 'single-use qualification')
    initial = copy.deepcopy(row); outcome = dict(host_probe_accepted=False); owner = None; start = cursor()
    root = ROOT / 'qualification'; root.mkdir()
    def persist():
        write(root / 'record.json', dict(outcome, commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    try:
        owner = owners.Owner(root / 'owned', persist)
        for name, artifact, args, marker in (
            ('canary', rt['preload']['canary'], [], None),
            ('submission', json.loads(runtime.selected.REFERENCE.read_text())['fixtures']['sycl_submission_test'],
                ['--submission'], 'fixture_status=PASS mode=submission')):
            case = root / name; case.mkdir()
            for sub in ('tmp', 'cache'): (case / sub).mkdir()
            env = environment(row, case, True); seen = set()
            allowed = dict(rt['runtime_files']); allowed[str(Path(artifact['path']).resolve())] = artifact['sha256']
            def observe(io, entry):
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try: mapped = runtime.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
                    except (FileNotFoundError, ProcessLookupError): mapped = set()
                    check_maps(mapped, allowed, row['neo']['path'], False); seen.update(mapped)
                    if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
                io.exit(1)
            entry, out, err = owner.run(name, [artifact['path'], *args], env, P, interaction=observe,
                wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=128 << 20)
            require(owners.closed(entry) and entry['exit_code'] == 0, 'qualification fixture failed')
            if marker:
                require(marker in out.read_text() and 'urEnqueue' in (case / 'ur.trace').read_text(),
                    'submission marker/API trace missing')
            else:
                require(json.loads(out.read_text()) == dict(pid=entry['pid'], initialized=0, checked=46,
                    expected=46, version=(1 << 32) - 1, closed=0), 'host canary differs')
            check_maps(seen, allowed, row['neo']['path'], marker is not None)
            runtime.validation.preload_evidence(err.read_text(), entry['pid'])
            require(not (case / 'validation.log').exists() or not (case / 'validation.log').stat().st_size,
                'qualification validation warning')
            row['qualification'].append(dict(case=name, passed=True, ownership_closed=True))
        current(initial); outcome['host_probe_accepted'] = True
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: runtime.validation.finish(outcome, owner, start, row['boot'])
        row.update(stage='qualified' if outcome.get('passed') else 'closed',
            ownership_closed=outcome.get('ownership_closed', True), qualification_outcome=outcome)
        persist(); save(row)
    print(json.dumps(outcome), flush=True)
    return int(row['stage'] != 'qualified')


def run(row):
    rt = current(row)
    require(row['stage'] == 'qualified' and len(row['qualification']) == 2 and not (ROOT / 'owned').exists(),
        'fresh qualified process only')
    initial = copy.deepcopy(row); env = environment(row, ROOT); owner = None; start = cursor()
    outcome = dict(passed=False, complete=False, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    def persist():
        write(ROOT / 'process.json', dict(outcome, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    def interaction(io, leader):
        for request, (minimum, round_index) in enumerate(request_order()):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase=f'prefill-scaling/{minimum}', run=round_index), 'request gate differs')
                require(not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size,
                    'native parameter warning')
                if stage == 'begin':
                    claim = json.loads((ROOT / 'engine-start.json').read_text()); pid = claim['native']['pid']
                    ident = owners.identity(pid)
                    require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                        str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    expected = dict(env, LD_PRELOAD=row['preload']['library']['path'],
                        STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                    require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                    paths = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    allowed = dict(rt['runtime_files']); allowed[str(Path(row['binary']['path']).resolve())] = row['binary']['sha256']
                    check_maps(paths, allowed, row['neo']['path'])
                    require(any('/libigdgmm.so.' in p for p in paths), 'actual GMM missing')
                    outcome['mapped_files'] = sorted(set(outcome.get('mapped_files', [])) | paths)
                    outcome['verified_request_boundaries'] = request + 1
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = owners.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('bench', row['command'], env, P, interaction=interaction, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0, 'official benchmark/owner failed')
        result_file, = (ROOT / 'bench').glob('*/results.json'); result = json.loads(result_file.read_text())
        cohort = result['results']['prefill_scaling']
        samples = [cohort['first_long_request'], *cohort['requests']]
        require(result['complete'] and cohort['complete'] and not result['results']['failures'] and
            len(samples) == 19 and outcome.get('verified_request_boundaries') == 19,
            'incomplete characterization')
        for sample, (minimum, round_index) in zip(samples, request_order()):
            require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                sample['thinking_requested'] and sample['reasoning_observed'] and
                sample['requested_min_tokens'] == minimum and sample['run'] == round_index and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] >= minimum and
                sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
                sample['prompt_tps_source'] == 'engine' and type(sample['prompt_ms']) in (int, float) and
                sample['prompt_ms'] > 0 and sample['ttft_s'] > 0 and sample['total_s'] >= sample['ttft_s'],
                'workload/timing eligibility differs')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (ROOT / 'bench').glob('*/requests.jsonl'); verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'journal invalid/incomplete')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        longs = [v for v in events if v['kind'] == 'result' and v['phase'].startswith('prefill-scaling/')]
        require([dict(v['sample'], run=v['run'], requested_min_tokens=int(v['phase'].rsplit('/',1)[1]))
                 for v in longs] == samples, 'journal/request binding differs')
        end = gen.native_end(ROOT, dict(binary=row['binary'], native_validation_probe=row['preload']),
            dict(config=row['configuration']), row['boot'], entry['pid'])
        text = (ROOT / 'engine.stderr').read_text()
        activation = runtime.validation.preload_evidence(text, end['native']['pid'])
        routes = route_evidence(text, samples)
        require(not any(s in text for s in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'route/runtime differs')
        current(initial); models.model_pins(row['configuration'], row['models'], full=True)
        fields = ('run','requested_min_tokens','request_sha256','actual_prompt_tokens','actual_read_tokens',
            'actual_reused_tokens','generated_tokens','prompt_ms','prompt_tps','thinking_requested',
            'reasoning_observed','ttft_s','total_s')
        outcome.update(complete=True, host_probe_accepted=True,
            first_long_request={k:samples[0][k] for k in fields},
            requests=[{k:s[k] for k in fields} for s in samples[1:]],
            routes=routes,linear_fit=cohort['linear_fit'],approximation_models=cohort['approximation_models'],
            adjacent_slopes=cohort['adjacent_slopes'],native_identity=end['native'],
            validation=activation,journal_sha256=digest(journal))
    except BaseException as error:
        outcome['error'] = repr(error)
        if (ROOT / 'engine.stderr').exists(): outcome['error_excerpt'] = (ROOT / 'engine.stderr').read_text()[-4096:]
    finally:
        if owner: runtime.validation.finish(outcome, owner, start, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        outcome['closed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); persist()
        row.update(stage='closed', active=False, ownership_closed=outcome.get('ownership_closed', True),
            complete=outcome['complete'], passed=outcome['passed'], outcome=outcome)
        save(row)
    print(json.dumps(outcome), flush=True)
    return int(not outcome['passed'])


def request_order():
    # Import the committed client implementation;gate protocol shares frozen order.
    sys.path.insert(0,str(P/'tools'))
    from prefill_scaling_bench import MINIMA,order
    return [(max(MINIMA),-1),*((n,r) for r in range(3) for n in order(r))]


def source_review():
    scope=runtime.validation.committed(gen.A/'research-index.json',
        ('prefill_length_scaling_admission_20261028',))
    require(scope['status']=='reviewed_bounded_measurement_only' and scope['worker_closed'] is True and
        scope['review']['status']=='completed' and scope['review']['model']=='gpt-6-luna' and
        scope['review']['reasoning_effort']=='max' and scope['review']['execution_eligible'] is True,
        'closed independent source audit and native curation required')
    for name,sha in scope['source_snapshot'].items():
        require(digest(P/name)==sha and
            hashlib.sha256(gen.git('show',scope['source_commit']+':'+name)).hexdigest()==sha,
            'reviewed length-characterization input changed: '+name)
    return dict(source_commit=scope['source_commit'],review_sha256=scope['review']['sha256'],
                source_snapshot=scope['source_snapshot'])


def route_evidence(text, samples):
    import re
    footers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M))
    require(len(footers)==20,'native footer count differs')
    evidence=[];previous=footers[0].end()
    for footer,sample in zip(footers[1:],samples):
        require(tuple(map(int,footer.groups()))==(sample['actual_prompt_tokens'],0,
            sample['actual_read_tokens'],sample['generated_tokens']),'native/API token count differs')
        part=text[previous:footer.start()];previous=footer.end()
        markers=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(\S+)',part,re.M)
        long_input=sample['actual_read_tokens']-1>8192
        require(markers==[(str(sample['actual_read_tokens']-1),'8192','bf16')] if long_input else not markers,
            'actual long/short route differs from configured chunk boundary')
        effective=re.findall(r'^strata MoE window configuration: requested_mode=(\d+) effective_mode=(\d+)',part,re.M)
        require(effective==[('10','10')] if long_input else not effective,'actual MoE route differs')
        evidence.append(dict(run=sample['run'],requested_min_tokens=sample['requested_min_tokens'],
            actual_read_tokens=sample['actual_read_tokens'],route='layer-major BF16 mode10' if long_input else 'non-layer-major short',
            prefill_rows=sample['actual_read_tokens']-1))
    return evidence


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); runtime.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        row = json.loads((ROOT / 'record.json').read_text())
        if sys.argv[1:] == ['qualify']: return qualify(row)
        if sys.argv[1:] == ['run']: return run(row)
        raise RuntimeError('use prepare, qualify or run')


if __name__ == '__main__': sys.exit(main())
