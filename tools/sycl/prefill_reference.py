#!/usr/bin/env python3
"""One fresh original-engine reference after the independent dense control.

Repository LICENSE. Main-authored from our 0697c74e8 crossover controller.
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
ROOT = B / 'prefill-near-dense-reference-v1'
RESULT = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
KEY = 'near_dense_original_reference_v1'
require, digest, write = runtime.require, runtime.digest, runtime.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('benchmark_pins', 'models', 'environment_template')}


def save(row):
    container = json.loads(RESULT.read_text())
    container['historical_regression_audit'][KEY] = compact(row)
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
    require(gen.committed(RESULT)['historical_regression_audit'][KEY] == compact(row),
        'commit exact reference plan/outcome')
    require(gen.benchmark_pins(row['source_commit']) == row['benchmark_pins'], 'benchmark/controller changed')
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
    previous = gen.committed(RESULT)['historical_regression_audit']
    require(not ROOT.exists() and KEY not in previous, 'single-use root; inspect existing attempt')
    prior = previous['userland_crossover']
    require(prior['stage'] == 'closed' and prior['ownership_closed'] and not prior['active'],
        'prior crossover must be closed')
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
    row = dict(stage='prepared', source_commit=revision, benchmark_pins=gen.benchmark_pins(revision),
        qualification_inputs={str(p): digest(p) for p in (TEMPLATE, runtime.RESULT, P / 'data/expert-profile.bin')},
        environment_template=gen.committed(TEMPLATE)['environment_template'], configuration=cfg,
        binary=full['binary'], build_proof=proof, preload=preload,
        neo=dict(path=neo, sha256=rt['runtime_files'][neo]), boot=rt['boot'], driver=rt['driver'],
        models=pinned_models, request_hashes=prior['request_hashes'],
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '4',
            '--profile-request-gates', '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(ROOT / 'bench')],
        plan=dict(question='Does original6e40 remain slower in a fresh process after the independent9B control?',
            repetition='One independent process,one short warmup,four fresh117489-token requests. '
                'Keep first use separate;report every sample and subsequent-three median/range. '
                'Descriptive current-state check,not matched dense/Strata hardware ceiling or three-process restoration.',
            constants='Original22182f1a ELF;1256-slot decode placement;131072 resident INT8 KV;8192 chunks;'
                'BF16/vLLM/mode10;unchanged86-key template plus explicit current NEO in both previous current '
                'and this process;native-only retained-loader validation;24 output tokens;thinking on;zero reuse.',
            prerequisites='Fresh host-only canary and bounded traced submission fixture;original13 artifacts '
                'and9853 Git inputs;qualified current driver/runtime;actual native environment/maps;no fresh fault.',
            decision='No candidate or runtime adoption. Compare subsequent median descriptively with '
                'historical1002.650659 and latest current-arm911.737134. No speed-based stop,retry or replacement. '
                'Stop first correctness,ownership,validation,budget,source or fault failure.',
            scope='No profiling,census,resource polling,state readback,rebuild,service change,reset or global settings. '
                'Existing prefill API clock includes handoff back to decode. This24-token output is not decode qualification.'),
        budgets=dict(wall=3000, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20,
            file_cap=64 << 20, total_cap=128 << 20),
        active=False, ownership_closed=True, qualification=[], complete=False, passed=False,
        retention='Keep useful individual samples,source/config/environment references and decision. '
            'Retire successful trace/raw/JIT output and this one-use controller after closure;original13 assets remain.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    write(ROOT / 'config.json', cfg)
    write(ROOT / 'plan.json', dict(binary=row['binary'], native_validation_probe=preload))
    row['generated_sha256'] = {name: digest(ROOT / name) for name in ('config.json', 'plan.json')}
    save(row)
    print('Prepared; commit the frozen plan before execution.', flush=True)


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
        for request in range(4):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
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
        samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == 4 and
            outcome.get('verified_request_boundaries') == 4, 'incomplete benchmark')
        for sample, expected in zip(samples, row['request_hashes']):
            require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                sample['thinking_requested'] and sample['reasoning_observed'] and sample['request_sha256'] == expected and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
                sample['prompt_tps_source'] == 'engine' and type(sample['prompt_ms']) in (int, float) and
                10000 <= sample['prompt_ms'] < 1500000, 'workload/timing eligibility differs')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (ROOT / 'bench').glob('*/requests.jsonl'); verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'journal invalid/incomplete')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        longs = [v for v in events if v['kind'] == 'result' and v['phase'] == 'prefill-only']
        require([dict(v['sample'], run=v['run']) for v in longs] == samples, 'journal/request binding differs')
        end = gen.native_end(ROOT, dict(binary=row['binary'], native_validation_probe=row['preload']),
            dict(config=row['configuration']), row['boot'], entry['pid'])
        text = (ROOT / 'engine.stderr').read_text()
        activation = runtime.validation.preload_evidence(text, end['native']['pid'])
        require(text.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16') == 4 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 4 and
            not any(s in text for s in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'route/runtime differs')
        current(initial); models.model_pins(row['configuration'], row['models'], full=True)
        fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
            'generated_tokens', 'prompt_ms', 'thinking_requested')
        rates = [117489000 / s['prompt_ms'] for s in samples]; median = statistics.median(rates[1:])
        outcome.update(complete=True, host_probe_accepted=True, requests=[{k: s[k] for k in fields} for s in samples],
            tok_s=rates, first_use_tok_s=rates[0], subsequent_median=median,
            subsequent_range=[min(rates[1:]), max(rates[1:])],
            ratio_to_historical_6e40=median / 1002.650659,
            ratio_to_previous_current_cohort=median / 911.7371344, native_identity=end['native'],
            validation=activation, journal_sha256=digest(journal))
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


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); runtime.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        row = json.loads((ROOT / 'record.json').read_text())
        if sys.argv[1:] == ['qualify']: return qualify(row)
        if sys.argv[1:] == ['run']: return run(row)
        raise RuntimeError('use prepare, qualify or run')


if __name__ == '__main__': sys.exit(main())
