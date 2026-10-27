#!/usr/bin/env python3
"""Planned MoE work on selected6e40; no API/device coverage or speed claim.

Uses the existing source-emitted census and official bench.sh, without rebuilding
the engine. Independently authored under the repository LICENSE.
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
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import benchmark_runtime as python_runtime
import cpu_capacity as models
import engine_once
import generation_engine as gen
import measurement_owner as owner_module
import runtime_probe as runtime
import supervision_model as shared
from measure_down_scatter import postchecks

B, A = gen.B, gen.A
ROOT = B / 'selected-model-work-census-v3'
RESULT = A / 'model-work-census-current.json'
REFERENCE = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
SOURCE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
CASES = dict(smoke=(32768, 39260, ((0, 39259),)),
             long=(98304, 117489, ((0, 81920), (81920, 35568))))
require, digest = runtime.require, runtime.digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def continue_benchmark(io):
    """Match the actual strata_bench.profile_request_gate line protocol."""
    io.send(b'continue\n')


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


def admission(row, full_models=False):
    full, rt = native()
    require(full['binary'] == row['binary'] and digest(runtime.RESULT) == row['runtime_sha256'] and
        digest(TEMPLATE) == row['template_sha256'] and rt['boot'] == row['boot'], 'native/runtime/template changed')
    require(gen.benchmark_pins(row['source_commit']) == row['controller_inputs'], 'census source changed')
    contract, manifest = python_contract()
    require(contract == row['python_contract'], 'benchmark runtime qualification changed')
    shared.profile_pin(row['configuration'], row['source_commit'], row['profile'])
    models.model_pins(row['configuration'], row['model_assets'], full=full_models)
    for mode, plan in row['plans'].items():
        root = Path(plan['root'])
        require(plan['environment'] == prepared_environment(root, row['NEO']) and
            python_runtime.selection(plan['environment']) == manifest['selected'] and
            request_hash(CASES[mode][0]) == plan['request_sha256'], 'workload/environment changed')
        for name, sha in plan['inputs'].items(): require(digest(root / name) == sha, 'generated input changed')
    return rt, manifest


def prepare():
    require(not ROOT.exists(), 'fresh census only; inspect ownership before retry')
    prior = gen.committed(RESULT)
    require(prior['stage'] == 'failed' and not prior['active'] and prior['ownership_closed'] and
        len(prior['checks']) == 1, 'preserve the closed failed prior census')
    failed = prior['checks'][0]
    require(not failed['passed'] and not failed['complete'] and failed['ownership_closed'] and
        failed['postchecks_complete'] and failed['fresh_faults'] == [] and
        failed['failure_diagnosis']['benchmark_exit_code'] == 0 and
        failed['failure_diagnosis']['native_exit_code'] == 0 and
        failed['failure_diagnosis']['long_requests_completed'] == 1 and
        failed['failure_diagnosis']['actual_input_tokens'] == 39260,
        'closed benchmark-summary/extent failure required')
    previous_failure = dict(commit=gen.git('log', '-1', '--format=%H', '--',
        str(RESULT.relative_to(P))).decode().strip(), path=str(RESULT.relative_to(P)),
        sha256=digest(RESULT), source_commit=prior['source_commit'],
        native_exit_code=0, long_requests_completed=1, passed=False)
    full, rt = native()
    reference = runtime.validation.committed(REFERENCE, ('historical_regression_audit', 'supervision_model_control_v6'))
    require(reference['complete'] and reference['passed'] and reference['ownership_closed'] and
        reference['binary'] == full['binary'], 'closed same-native reference required')
    cfg = copy.deepcopy(reference['configuration'])
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    contract, manifest = python_contract()
    old_assets = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    model_assets = models.model_pins(cfg, old_assets, full=True)
    loader = rt['preload']['loader']
    probe = dict(library=rt['preload']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]))
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=source, native_source=SOURCE, binary=full['binary'],
        runtime_sha256=digest(runtime.RESULT), template_sha256=digest(TEMPLATE), boot=rt['boot'],
        NEO=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p), probe=probe,
        configuration=cfg, profile=shared.profile_pin(cfg, source), python_contract=contract,
        controller_inputs=gen.benchmark_pins(source), model_assets=model_assets, plans={}, checks=[],
        active=False, ownership_closed=True, complete=False, passed=False, model_adopted=False,
        previous_failure=previous_failure,
        question='Obtain source-planned MoE jobs on the actual selected route and verify retired totals before a complete API/device ledger.',
        limits='Diagnostic census only. No complete device coverage,physical bus traffic,hardware saturation,clean speed,decode or physical262144 qualification.',
        retention='Keep compact counts and one exact job-size histogram for the imminent source/API work planner. Retire successful raw logs and private caches after extraction;no new engine build. Review histogram consumer after route-plan selection.')
    for mode, (minimum, actual, pairs) in CASES.items():
        root = ROOT / mode; root.mkdir()
        for folder in ('tmp', 'cache'): (root / folder).mkdir()
        config = copy.deepcopy(cfg); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config); write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=probe))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        row['plans'][mode] = dict(root=str(root), config=config, expected_tokens=actual, expected_window_pairs=pairs,
            request_sha256=request_hash(minimum), environment=prepared_environment(root, row['NEO']),
            inputs={n:digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', str(minimum), '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')],
            budgets=dict(wall=1200, cpu=7200, rss_cap=96 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20))
    require(row['plans']['long']['request_sha256'] == reference['request_hashes'][0], 'official long request changed')
    admission(row)
    write(ROOT / 'record.json', row); write(RESULT, row)
    print('Prepared source census;commit exact plans before the smoke.', flush=True)


def run(mode):
    row = gen.committed(RESULT); plan = row['plans'][mode]; root = Path(plan['root'])
    require(row == json.loads((ROOT / 'record.json').read_text()), 'external census record differs')
    position = ('smoke', 'long').index(mode)
    require(len(row['checks']) == position and all(c['passed'] and c['ownership_closed'] for c in row['checks']) and
        not (root / 'execution.json').exists(), 'ordered single-use census;stop on failure')
    rt, manifest = admission(row)
    cursor = next(x.split(': ', 1)[1] for x in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True).splitlines() if x.startswith('-- cursor: '))
    result = dict(mode=mode, passed=False, complete=False, ownership_closed=False, native_launched=None)
    owner = None
    def persist():
        write(root / 'execution.json', dict(result, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    owner = owner_module.Owner(root / 'owned', persist)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def gate(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'request gate differs')
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']
            result['native_launched'] = True
            current = owner_module.identity(pid)
            require(current and current['start_ticks'] == claim['native']['start_ticks'] and
                current['sid'] == leader['pid'] and str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native ownership differs')
            expected = engine_once.native_environment(dict(native_validation_probe=row['probe']), plan['environment'])
            actual = dict(x.split(b'=', 1) for x in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in x)
            require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()) and
                set(actual)-{k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native environment differs')
            maps = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
            require(maps <= set(rt['runtime_files']) | {row['binary']['path']} and
                {row['NEO'],row['probe']['loader']['path'],row['probe']['library']['path']} <= maps, 'native provider differs')
            allowed = {p['realpath'] for p in manifest['inventory']['files'].values()}
            wrapper = owner_module.identity(claim['wrapper']['pid']); require(wrapper and wrapper['sid'] == leader['pid'], 'wrapper ownership differs')
            server = owner_module.identity(wrapper['ppid']); require(server and server['sid'] == leader['pid'], 'server ownership differs')
            port = plan['command'][plan['command'].index('--port')+1]
            python_runtime.check_running(leader['pid'], P/'tools/strata_bench.py', manifest['selected'], plan['environment'], allowed, plan['command'][2:])
            python_runtime.check_running(server['pid'], P/'serve/server.py', manifest['selected'], plan['environment'], allowed,
                ['--engine','strata','--config',str(root/'config.json'),'--port',port])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size == 0, 'parameter warning/error')
            continue_benchmark(io)
        finish_benchmark(io)
    try:
        entry, _, _ = owner.run('bench', plan['command'], plan['environment'], P, interaction=gate, **plan['budgets'])
        require(owner_module.closed(entry) and entry['exit_code'] == 0, 'benchmark owner/protocol failed')
        end = gen.native_end(root, dict(binary=row['binary'], native_validation_probe=row['probe']), dict(config=plan['config']), row['boot'], entry['pid'])
        result['native_launched'] = True
        path, = (root/'bench').glob('*/results.json'); data = json.loads(path.read_text())
        cohort = data['results']['prefill_only']; require(cohort['complete'] and len(cohort['requests']) == 1, 'one complete request required')
        sample = cohort['requests'][0]
        require(sample['eligible'] and sample['ok'] and sample['controller_eligible'] and
            sample['request_sha256'] == plan['request_sha256'] and
            sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == plan['expected_tokens'] and
            sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24, 'request work differs')
        sys.path.insert(0, str(P/'tools')); from strata_bench import verify_request_journal
        journal = verify_request_journal(path.parent)
        require(journal['valid'] and journal['complete'], 'request journal invalid')
        events = [json.loads(line) for line in (path.parent/'requests.jsonl').read_text().splitlines()]
        require([dict(v['sample'], run=v['run']) for v in events if v['kind'] == 'result' and
            v['phase'] == 'prefill-only'] == cohort['requests'], 'cohort differs from durable request journal')
        text = (root/'engine.stderr').read_text()
        result['validation'] = runtime.validation.preload_evidence(text, end['native']['pid'])
        require(result['validation']['loader'] == row['probe']['loader']['path'], 'validation provider differs')
        require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|retirement failed', text), 'native runtime error')
        footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', text, re.M))
        require(len(footers) == 2 and 0 < int(footers[0][1]) < 32768 and int(footers[0][2]) == 0 and
            tuple(map(int,footers[1].groups())) == (plan['expected_tokens'],0,plan['expected_tokens'],sample['generated_tokens']), 'native request interval differs')
        part = text[footers[0].end():footers[1].start()]
        maximum_rows = max(rows for _, rows in plan['expected_window_pairs'])
        require(part.count('strata prefill: layer-major candidate, tokens='+str(plan['expected_tokens']-1)+' chunk=8192 residual_dtype=bf16') == 1 and
            part.count('strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+str(maximum_rows)) == 1, 'selected prefill route differs')
        work = gen.common.extract_planned_moe_work(part, plan['expected_window_pairs'])
        result.update(census=gen.common.compact_window_census(work['planned_moe_windows']),
            request={k:sample[k] for k in ('request_sha256','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms')},
            native_identity=end['native'])
        admission(row, full_models=(mode == 'long'))
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
        if (root/'engine.stderr').exists(): result['error_excerpt'] = (root/'engine.stderr').read_text()[-4096:]
    finally:
        result['ownership_closed'] = owner.active is None and all(c.get('direct_child_reaped') is True and
            c.get('session_empty') is True and c.get('survivors') == [] and
            c.get('ownership_status') == 'observed' and type(c.get('exit_code')) is int for c in owner.commands)
        result['protocol_qualified'] = bool(owner.commands) and result['passed'] and all(owner_module.closed(c) for c in owner.commands)
        result['passed'] = result['passed'] and result['protocol_qualified']
        postchecks(result, row['boot'], cursor)
        result['complete'] = bool(result['complete'] and result['passed'])
        persist()
        row['checks'].append(result); row['active'] = bool(owner and owner.active)
        row['ownership_closed'] = result.get('ownership_closed', False)
        row['stage'] = 'running' if result['passed'] else 'failed'
        if mode == 'long' and result['passed']: row.update(stage='closed',complete=True,passed=True)
        write(ROOT/'record.json',row);write(RESULT,row)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([]);runtime.guards.driver_guard()
        if sys.argv[1:] == ['prepare']: prepare()
        elif sys.argv[1:] in (['run','smoke'],['run','long']): sys.exit(run(sys.argv[2]))
        else: raise RuntimeError('prepare|run smoke|run long')
